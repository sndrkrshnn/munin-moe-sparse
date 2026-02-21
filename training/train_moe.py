#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml
from transformers import AutoTokenizer

from src.munin_moe.model import MuninMoEModel


class JsonlTextDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text") or obj.get("prompt", "")
                expert = obj.get("expert", "general")
                self.rows.append((text, expert))
        self.tok = tokenizer
        self.max_len = max_len
        self.expert_map = {"linux": 0, "toolcalling": 1, "general": -1}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text, expert = self.rows[idx]
        enc = self.tok(text, truncation=True, max_length=self.max_len + 1, return_tensors="pt")
        ids = enc["input_ids"][0]
        if ids.size(0) < 2:
            ids = torch.tensor([self.tok.bos_token_id or 1, self.tok.eos_token_id or 2])
        x = ids[:-1]
        y = ids[1:]
        return x, y, self.expert_map.get(expert, -1)


def collate(batch, pad_id=0):
    xs, ys, es = zip(*batch)
    max_t = max(x.size(0) for x in xs)
    bx = torch.full((len(xs), max_t), pad_id, dtype=torch.long)
    by = torch.full((len(xs), max_t), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        bx[i, : x.size(0)] = x
        by[i, : y.size(0)] = y
    return bx, by, torch.tensor(es, dtype=torch.long)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def router_supervision_loss(router_aux, expert_labels):
    if not router_aux:
        return torch.tensor(0.0, device=expert_labels.device)
    valid = expert_labels >= 0
    if not valid.any():
        return torch.tensor(0.0, device=expert_labels.device)

    losses = []
    for aux in router_aux:
        logits = aux["router_logits"]  # [B,T,E]
        cls = logits[:, 0, :]  # sample-level label supervision on first token
        losses.append(F.cross_entropy(cls[valid], expert_labels[valid]))
    return torch.stack(losses).mean()


def router_load_balance_loss(router_aux):
    if not router_aux:
        return torch.tensor(0.0)

    losses = []
    for aux in router_aux:
        probs = aux["router_probs"]  # [B,T,E]
        mean_probs = probs.mean(dim=(0, 1))  # [E]
        target = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
        losses.append(F.mse_loss(mean_probs, target))
    return torch.stack(losses).mean()


def expert_utilization(router_aux):
    if not router_aux:
        return {}
    counts = None
    for aux in router_aux:
        sel = aux["selected"]  # [B,T,K]
        e = aux["router_probs"].size(-1)
        flat = sel.reshape(-1)
        c = torch.bincount(flat, minlength=e).float()
        counts = c if counts is None else counts + c
    counts = counts / max(counts.sum().item(), 1.0)
    return {f"expert_{i}": counts[i].item() for i in range(counts.numel())}


def evaluate(model, dataloader, device, top_k):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y, _ in dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x, top_k=top_k)
            l = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            losses.append(l.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-config", required=True)
    args = ap.parse_args()

    model_cfg = load_yaml(args.config)
    train_cfg = load_yaml(args.train_config)

    tokenizer_name = train_cfg.get("tokenizer", {}).get("name", "gpt2")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    max_len = train_cfg["data"].get("max_length", 512)
    ds_train = JsonlTextDataset(train_cfg["data"]["train_path"], tok, max_len=max_len)
    ds_val = JsonlTextDataset(train_cfg["data"]["val_path"], tok, max_len=max_len)

    micro_bs = train_cfg["batching"].get("micro_batch_size", 1)
    train_dl = DataLoader(ds_train, batch_size=micro_bs, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))
    val_dl = DataLoader(ds_val, batch_size=1, shuffle=False, collate_fn=lambda b: collate(b, tok.pad_token_id))

    m = model_cfg["model"]
    moe = model_cfg["moe"]
    model = MuninMoEModel(
        vocab_size=m["vocab_size"],
        max_seq_len=m["max_seq_len"],
        dim=m["hidden_size"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        moe_layers=moe.get("moe_layers", []),
        num_experts=moe.get("num_experts", 2),
        expert_mult=moe.get("expert_ffn_mult", 2.5),
    )

    requested = train_cfg["hardware"].get("device", "cpu")
    device = "mps" if requested == "mps" and torch.backends.mps.is_available() else "cpu"
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["training"]["learning_rate"]))
    epochs = int(train_cfg["training"].get("epochs", 1))
    grad_acc = int(train_cfg["batching"].get("gradient_accumulation_steps", 1))

    loss_cfg = train_cfg.get("loss", {})
    router_w = float(loss_cfg.get("router_supervision_weight", 0.2))
    lb_w = float(loss_cfg.get("load_balance_weight", 0.02))

    logging_steps = int(train_cfg.get("outputs", {}).get("logging_steps", 10))

    out_dir = Path(train_cfg["outputs"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Training Configuration ===")
    print(f"device={device} tokenizer={tokenizer_name}")
    print(f"train_samples={len(ds_train)} val_samples={len(ds_val)}")
    print(f"epochs={epochs} micro_batch={micro_bs} grad_acc={grad_acc}")
    print(f"router_w={router_w} load_balance_w={lb_w}")
    print(f"logging_steps={logging_steps} top_k={moe.get('top_k_default', 1)}")

    global_step = 0
    model.train()
    for ep in range(epochs):
        ep_start = time.time()
        optim.zero_grad(set_to_none=True)
        running_loss = 0.0

        for i, (x, y, e) in enumerate(train_dl):
            step_start = time.time()
            x, y, e = x.to(device), y.to(device), e.to(device)
            logits, aux = model(x, top_k=moe.get("top_k_default", 1))
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            r_loss = router_supervision_loss(aux, e)
            lb_loss = router_load_balance_loss(aux).to(device)
            loss = lm_loss + router_w * r_loss + lb_w * lb_loss
            (loss / grad_acc).backward()
            running_loss += loss.item()

            if (i + 1) % grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % logging_steps == 0:
                    util = expert_utilization(aux)
                    util_s = " ".join(f"{k}={v:.2f}" for k, v in util.items()) if util else "expert_util=NA"
                    tok_count = int((y != -100).sum().item())
                    dt = time.time() - step_start
                    print(
                        f"[train] ep={ep+1}/{epochs} batch={i+1}/{len(train_dl)} step={global_step} "
                        f"loss={loss.item():.4f} lm={lm_loss.item():.4f} router={r_loss.item():.4f} lb={lb_loss.item():.4f} "
                        f"tokens={tok_count} step_sec={dt:.3f} {util_s}"
                    )

        train_avg = running_loss / max(1, len(train_dl))
        val_loss = evaluate(model, val_dl, device, moe.get("top_k_default", 1))

        ckpt = out_dir / f"epoch-{ep+1}.pt"
        torch.save({"model": model.state_dict(), "tokenizer": tok.name_or_path}, ckpt)
        print(
            f"[epoch_end] ep={ep+1}/{epochs} train_loss={train_avg:.4f} val_loss={val_loss:.4f} "
            f"elapsed_sec={time.time()-ep_start:.1f} checkpoint={ckpt}"
        )

    latest = out_dir / "latest"
    latest.mkdir(exist_ok=True)
    torch.save(model.state_dict(), latest / "model.pt")
    tok.save_pretrained(latest / "tokenizer")
    print(f"wrote final artifact to {latest}")


if __name__ == "__main__":
    main()

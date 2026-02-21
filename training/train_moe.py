#!/usr/bin/env python3
import argparse
import json
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
        cls = logits[:, 0, :]  # supervise first token as sample-level label
        losses.append(F.cross_entropy(cls[valid], expert_labels[valid]))
    return torch.stack(losses).mean()


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

    device = "mps" if train_cfg["hardware"].get("device") == "mps" and torch.backends.mps.is_available() else "cpu"
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["training"]["learning_rate"]))
    epochs = int(train_cfg["training"].get("epochs", 1))
    grad_acc = int(train_cfg["batching"].get("gradient_accumulation_steps", 1))
    router_w = float(train_cfg.get("loss", {}).get("router_supervision_weight", 0.2))

    out_dir = Path(train_cfg["outputs"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    model.train()
    for ep in range(epochs):
        optim.zero_grad(set_to_none=True)
        for i, (x, y, e) in enumerate(train_dl):
            x, y, e = x.to(device), y.to(device), e.to(device)
            logits, aux = model(x, top_k=moe.get("top_k_default", 1))
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            r_loss = router_supervision_loss(aux, e)
            loss = lm_loss + router_w * r_loss
            (loss / grad_acc).backward()

            if (i + 1) % grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0:
                    print(f"ep={ep} step={global_step} lm={lm_loss.item():.4f} router={r_loss.item():.4f}")

        ckpt = out_dir / f"epoch-{ep+1}.pt"
        torch.save({"model": model.state_dict(), "tokenizer": tok.name_or_path}, ckpt)
        print(f"saved {ckpt}")

    # quick val perplexity proxy
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y, _ in val_dl:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x, top_k=moe.get("top_k_default", 1))
            l = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            losses.append(l.item())
    if losses:
        print(f"val_loss={sum(losses)/len(losses):.4f}")

    latest = out_dir / "latest"
    latest.mkdir(exist_ok=True)
    torch.save(model.state_dict(), latest / "model.pt")
    tok.save_pretrained(latest / "tokenizer")
    print(f"wrote final artifact to {latest}")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.hd = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.hd).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(y)


class ExpertFFN(nn.Module):
    def __init__(self, dim: int, mult: float = 2.5):
        super().__init__()
        hidden = int(dim * mult)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SparseMoE(nn.Module):
    def __init__(self, dim: int, num_experts: int = 2, expert_mult: float = 2.5):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(dim, expert_mult) for _ in range(num_experts)])

    def forward(self, x, top_k: int = 1):
        # x: [B,T,C]
        logits = self.router(x)
        probs = torch.softmax(logits, dim=-1)

        k = min(top_k, self.num_experts)
        topv, topi = torch.topk(probs, k=k, dim=-1)

        out = torch.zeros_like(x)
        for ei, expert in enumerate(self.experts):
            mask = (topi == ei).any(dim=-1)  # [B,T]
            if not mask.any():
                continue
            ex = expert(x)
            w = probs[..., ei].unsqueeze(-1)
            out = out + ex * w * mask.unsqueeze(-1)

        aux = {
            "router_logits": logits,
            "router_probs": probs,
            "selected": topi,
        }
        return out, aux


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, use_moe: bool, num_experts: int, expert_mult: float):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, n_heads)
        self.n2 = RMSNorm(dim)
        self.use_moe = use_moe
        if use_moe:
            self.ff = SparseMoE(dim, num_experts=num_experts, expert_mult=expert_mult)
        else:
            self.ff = ExpertFFN(dim, mult=expert_mult)

    def forward(self, x, top_k=1):
        x = x + self.attn(self.n1(x))
        if self.use_moe:
            ff, aux = self.ff(self.n2(x), top_k=top_k)
        else:
            ff, aux = self.ff(self.n2(x)), None
        x = x + ff
        return x, aux


class MuninMoEModel(nn.Module):
    def __init__(self, vocab_size=32000, max_seq_len=2048, dim=768, n_layers=16, n_heads=12, moe_layers=None, num_experts=2, expert_mult=2.5):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_seq_len, dim)
        moe_layers = set(moe_layers or [])
        self.blocks = nn.ModuleList([
            Block(dim, n_heads, i in moe_layers, num_experts, expert_mult) for i in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, top_k=1):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok(input_ids) + self.pos(pos)

        router_aux = []
        for b in self.blocks:
            x, aux = b(x, top_k=top_k)
            if aux is not None:
                router_aux.append(aux)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, router_aux

from dataclasses import dataclass
import math


@dataclass
class RoutingDecision:
    top_k: int
    selected_experts: list[int]
    entropy: float


def softmax(logits):
    m = max(logits)
    ex = [math.exp(x - m) for x in logits]
    s = sum(ex)
    return [v / s for v in ex]


def route(logits, top_k_default=1, entropy_fallback_threshold=1.15, top_k_fallback=2):
    probs = softmax(logits)
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    k = top_k_fallback if entropy >= entropy_fallback_threshold else top_k_default
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    return RoutingDecision(top_k=k, selected_experts=order[:k], entropy=entropy)

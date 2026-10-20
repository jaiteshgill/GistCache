def cost_usd(pricing: dict, which: str, in_tokens: int, out_tokens: int) -> float:
    p = pricing.get(which, {})
    return (in_tokens * p.get("input", 0.0) + out_tokens * p.get("output", 0.0)) / 1_000_000

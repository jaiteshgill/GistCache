import yaml

DEFAULTS = {
    "verifier": {"hit_threshold": 0.90, "near_threshold": 0.70, "top_k": 5, "strict_content_check": True,
                 "contradiction_threshold": 0.5, "borderline_judge": False, "borderline_low": 0.3, "nli_guard": "cross-encoder/nli-deberta-v3-small"},
    "patcher": {"enabled": True, "max_length_ratio": 1.5, "cache_patched_answers": False, "prompt": "v1"},
    "cache": {"path": "./data/gistcache.db", "max_entries": 100000, "default_ttl_seconds": None, "volatility_check": False},
    "bypass": {"max_temperature": 0.7, "with_tools": True},
    "multi_turn": {"context_turns": 4},
    "pricing": {},
}


def load_config(path: str | None) -> dict:
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg

import argparse
import re
import time

from .config import load_config
from .store import Store


def _parse_age(s: str) -> int:
    m = re.fullmatch(r"(\d+)([smhd])", s)
    if not m:
        raise SystemExit("--older-than must look like 30m, 12h or 7d")
    return int(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def main(argv=None):
    p = argparse.ArgumentParser(prog="gistcache")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8100); s.add_argument("--host", default="127.0.0.1")
    l = sub.add_parser("log"); l.add_argument("--last", type=int, default=10)
    sub.add_parser("stats")
    pu = sub.add_parser("purge"); pu.add_argument("--scope"); pu.add_argument("--older-than")
    a = p.parse_args(argv)
    cfg = load_config(a.config)

    if a.cmd == "serve":
        import uvicorn
        from .server import build_pipeline, create_app
        uvicorn.run(create_app(build_pipeline(cfg)), host=a.host, port=a.port)
        return
    store = Store(cfg["cache"]["path"], cfg["cache"]["max_entries"])
    if a.cmd == "log":
        print(f"{'TIME':9} {'TIER':9} {'REASON':40} {'COS':5} {'XENC':5} {'MS':>6} COST")
        for r in store.last_decisions(a.last):
            f = lambda v: f"{v:.2f}" if v is not None else "-"
            print(f"{time.strftime('%H:%M:%S', time.localtime(r['request_at'])):9} {r['tier']:9} {r['reason'][:40]:40} "
                  f"{f(r['cosine']):5} {f(r['cross_score']):5} {r['latency_ms']:6.0f} ${r['cost_usd']:.4f}")
    elif a.cmd == "stats":
        for tier, v in store.stats().items():
            print(f"{tier:9} n={v['count']:<5} cost=${v['cost_usd']:.4f} mean_latency={v['mean_latency_ms']:.0f}ms")
    elif a.cmd == "purge":
        print(f"purged {store.purge(a.scope, _parse_age(a.older_than) if a.older_than else None)} entries")


if __name__ == "__main__":
    main()

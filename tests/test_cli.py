import time

import pytest

from gistcache import cli
from gistcache.config import load_config
from gistcache.providers.base import Anthropic, OpenAICompatible, build_provider
from gistcache.store import Store


@pytest.fixture
def cfg_path(tmp_path):
    db = tmp_path / "g.db"
    p = tmp_path / "config.yaml"
    p.write_text(f"cache:\n  path: {db}\n  max_entries: 1000\n")
    return str(p), db


def seeded(db):
    s = Store(str(db))
    s.add_entry("a", None, "q1", "q1", "ans", "m", b"")
    s.add_entry("b", None, "q2", "q2", "ans", "m", b"")
    s.log_decision("a", "q1", "hit", "hit: paraphrase", None, 0.93, 0.97, [], False, 12.0, 0.0)
    s.log_decision("a", "q2", "miss", "miss: negation mismatch", None, 0.9, 0.2, [], False, 2000.0, 0.006)
    return s


def run(cfg, *args):
    cli.main(["--config", cfg, *args])


def test_log_prints_recent_decisions(cfg_path, capsys):
    cfg, db = cfg_path
    seeded(db)
    run(cfg, "log", "--last", "5")
    out = capsys.readouterr().out
    assert "TIER" in out and "hit: paraphrase" in out and "miss: negation mismatch" in out and "$0.0060" in out


def test_stats_summarizes_tiers(cfg_path, capsys):
    cfg, db = cfg_path
    seeded(db)
    run(cfg, "stats")
    out = capsys.readouterr().out
    assert "hit" in out and "miss" in out and "n=1" in out


def test_purge_by_scope_only_removes_that_scope(cfg_path, capsys):
    cfg, db = cfg_path
    s = seeded(db)
    run(cfg, "purge", "--scope", "a")
    assert "purged 1 entries" in capsys.readouterr().out
    assert s.candidates("a", None) == [] and len(s.candidates("b", None)) == 1


def test_purge_older_than_keeps_fresh_entries(cfg_path, capsys):
    cfg, db = cfg_path
    s = seeded(db)
    s.db.execute("UPDATE entries SET created_at=? WHERE scope='a'", (int(time.time()) - 10 * 86400,))
    s.db.commit()
    run(cfg, "purge", "--older-than", "7d")
    assert "purged 1 entries" in capsys.readouterr().out
    assert s.candidates("a", None) == [] and len(s.candidates("b", None)) == 1


@pytest.mark.parametrize("text,seconds", [("30s", 30), ("5m", 300), ("12h", 43200), ("7d", 604800)])
def test_parse_age(text, seconds):
    assert cli._parse_age(text) == seconds


@pytest.mark.parametrize("bad", ["7", "d", "7 days", "-1d", "1w"])
def test_parse_age_rejects_bad_input(bad):
    with pytest.raises(SystemExit):
        cli._parse_age(bad)


def test_serve_starts_uvicorn_on_the_requested_port(cfg_path, monkeypatch):
    import uvicorn
    from gistcache import server
    cfg, _ = cfg_path
    seen = {}
    monkeypatch.setattr(server, "build_pipeline", lambda c: "pipeline")
    monkeypatch.setattr(server, "create_app", lambda p: ("app", p))
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: seen.update(app=app, host=host, port=port))
    run(cfg, "serve", "--port", "9123")
    assert seen == {"app": ("app", "pipeline"), "host": "127.0.0.1", "port": 9123}


def test_shipped_configs_load_and_build_providers(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    for path in ("config.example.yaml", "configs/anthropic.yaml", "configs/ollama.yaml"):
        c = load_config(path)
        for role in ("large", "small"):
            assert build_provider(c["providers"][role]).model


def test_ollama_config_uses_the_local_endpoint_and_leaves_unvalidated_judges_off():
    c = load_config("configs/ollama.yaml")
    p = build_provider(c["providers"]["large"])
    assert isinstance(p, OpenAICompatible) and p.base_url == "http://localhost:11434/v1" and p.api_key == ""
    assert c["verifier"]["borderline_judge"] is False and c["cache"]["volatility_check"] is False


def test_anthropic_config_builds_the_anthropic_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert isinstance(build_provider(load_config("configs/anthropic.yaml")["providers"]["large"]), Anthropic)


def test_config_flag_works_before_or_after_the_subcommand(cfg_path, capsys):
    cfg, db = cfg_path
    seeded(db)
    cli.main(["--config", cfg, "stats"])          # before
    before = capsys.readouterr().out
    cli.main(["stats", "--config", cfg])          # after (what the README shows)
    after = capsys.readouterr().out
    assert before == after and "hit" in after


def test_serve_accepts_config_after_the_subcommand(cfg_path, monkeypatch):
    import uvicorn
    from gistcache import server
    cfg, _ = cfg_path
    got = {}
    monkeypatch.setattr(server, "build_pipeline", lambda c: got.update(path=c["cache"]["path"]) or "pipeline")
    monkeypatch.setattr(server, "create_app", lambda p: p)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    cli.main(["serve", "--config", cfg, "--port", "9001"])
    assert got["path"].endswith("g.db")  # the config file was actually read

"""SQLite store for cache entries and the decision log."""
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id TEXT PRIMARY KEY, scope TEXT NOT NULL, context_key TEXT,
    query TEXT NOT NULL, query_norm TEXT NOT NULL, answer TEXT NOT NULL,
    model TEXT NOT NULL, created_at INTEGER NOT NULL, expires_at INTEGER,
    hit_count INTEGER DEFAULT 0, last_hit_at INTEGER, embedding BLOB
);
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY, request_at INTEGER NOT NULL, scope TEXT NOT NULL,
    query TEXT NOT NULL, tier TEXT NOT NULL, reason TEXT NOT NULL,
    candidate_id TEXT, cosine REAL, cross_score REAL, differences TEXT,
    fell_back INTEGER DEFAULT 0, latency_ms REAL NOT NULL, cost_usd REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entries_scope ON entries(scope, context_key);
"""


@dataclass
class Entry:
    id: str
    scope: str
    context_key: str | None
    query: str
    query_norm: str
    answer: str
    model: str
    created_at: int
    expires_at: int | None
    embedding: bytes | None = None


class Store:
    def __init__(self, path: str = ":memory:", max_entries: int = 100_000):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.max_entries = max_entries

    def add_entry(self, scope, context_key, query, query_norm, answer, model,
                  embedding: bytes | None = None, ttl: int | None = None) -> str:
        eid, now = uuid.uuid4().hex, int(time.time())
        self.db.execute(
            "INSERT INTO entries (id,scope,context_key,query,query_norm,answer,model,created_at,expires_at,embedding)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (eid, scope, context_key, query, query_norm, answer, model, now, now + ttl if ttl else None, embedding))
        self.db.commit()
        self._evict()
        return eid

    def candidates(self, scope: str, context_key: str | None) -> list[Entry]:
        now = int(time.time())
        rows = self.db.execute(
            "SELECT * FROM entries WHERE scope=? AND context_key IS ? AND (expires_at IS NULL OR expires_at>?)",
            (scope, context_key, now)).fetchall()
        return [Entry(**{k: r[k] for k in Entry.__dataclass_fields__}) for r in rows]

    def get(self, eid: str) -> Entry | None:
        r = self.db.execute("SELECT * FROM entries WHERE id=?", (eid,)).fetchone()
        return Entry(**{k: r[k] for k in Entry.__dataclass_fields__}) if r else None

    def touch(self, eid: str) -> None:
        self.db.execute("UPDATE entries SET hit_count=hit_count+1, last_hit_at=? WHERE id=?", (int(time.time()), eid))
        self.db.commit()

    def _evict(self) -> None:
        n = self.db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        if n > self.max_entries:
            self.db.execute(
                "DELETE FROM entries WHERE id IN (SELECT id FROM entries "
                "ORDER BY COALESCE(last_hit_at, created_at) ASC LIMIT ?)", (n - self.max_entries,))
            self.db.commit()

    def log_decision(self, scope, query, tier, reason, candidate_id, cosine, cross_score,
                     differences, fell_back, latency_ms, cost_usd) -> None:
        self.db.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, int(time.time()), scope, query, tier, reason, candidate_id, cosine, cross_score,
             json.dumps([str(d) for d in differences]), int(fell_back), latency_ms, cost_usd))
        self.db.commit()

    def last_decisions(self, n: int = 10):
        return self.db.execute("SELECT * FROM decisions ORDER BY request_at DESC, rowid DESC LIMIT ?", (n,)).fetchall()

    def stats(self) -> dict:
        rows = self.db.execute("SELECT tier, COUNT(*) c, SUM(cost_usd) cost, AVG(latency_ms) lat "
                               "FROM decisions GROUP BY tier").fetchall()
        return {r["tier"]: {"count": r["c"], "cost_usd": r["cost"] or 0.0, "mean_latency_ms": r["lat"]} for r in rows}

    def purge(self, scope: str | None = None, older_than_s: int | None = None) -> int:
        q, args = "DELETE FROM entries WHERE 1=1", []
        if scope:
            q += " AND scope=?"; args.append(scope)
        if older_than_s:
            q += " AND created_at<?"; args.append(int(time.time()) - older_than_s)
        cur = self.db.execute(q, args)
        self.db.commit()
        return cur.rowcount

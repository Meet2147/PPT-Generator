import json
import sqlite3
import time
from typing import Optional, Dict, Any

DB_PATH = "quiz_cache.sqlite3"

def init_cache():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS quiz_cache (
      cache_key TEXT PRIMARY KEY,
      payload_json TEXT NOT NULL,
      created_at INTEGER NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def get_cached(cache_key: str, ttl_seconds: int) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT payload_json, created_at FROM quiz_cache WHERE cache_key=?", (cache_key,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    payload_json, created_at = row
    if int(time.time()) - int(created_at) > ttl_seconds:
        return None

    return json.loads(payload_json)

def set_cached(cache_key: str, payload: Dict[str, Any]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO quiz_cache(cache_key, payload_json, created_at) VALUES (?, ?, ?)",
        (cache_key, json.dumps(payload), int(time.time()))
    )
    conn.commit()
    conn.close()

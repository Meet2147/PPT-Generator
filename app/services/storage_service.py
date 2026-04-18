import sqlite3
from pathlib import Path
from typing import Any

from app.config import settings


DB_PATH = settings.cache_dir / "deckmint.db"


def init_db() -> None:
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS deck_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                topic TEXT NOT NULL,
                audience TEXT NOT NULL,
                tone TEXT NOT NULL,
                slide_count INTEGER NOT NULL,
                pptx_filename TEXT NOT NULL,
                pdf_filename TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lifetime_claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                razorpay_payment_link_id TEXT,
                razorpay_payment_id TEXT UNIQUE,
                email TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def create_user(name: str, email: str, password_hash: str) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, password_hash),
        )
        return int(cursor.lastrowid)


def get_user_by_email(email: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, name, email, password_hash, created_at FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, name, email, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def record_deck_history(
    user_id: int,
    title: str,
    topic: str,
    audience: str,
    tone: str,
    slide_count: int,
    pptx_filename: str,
    pdf_filename: str | None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO deck_history
            (user_id, title, topic, audience, tone, slide_count, pptx_filename, pdf_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, title, topic, audience, tone, slide_count, pptx_filename, pdf_filename),
        )


def list_user_decks(user_id: int) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, title, topic, audience, tone, slide_count, pptx_filename, pdf_filename, created_at
            FROM deck_history
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def count_lifetime_claims() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM lifetime_claims").fetchone()
    return int(row["count"] if row else 0)


def record_lifetime_claim(
    razorpay_payment_link_id: str | None,
    razorpay_payment_id: str | None,
    email: str | None,
) -> bool:
    if not razorpay_payment_id:
        return False
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM lifetime_claims WHERE razorpay_payment_id = ?",
            (razorpay_payment_id,),
        ).fetchone()
        if existing:
            return False
        conn.execute(
            """
            INSERT INTO lifetime_claims (razorpay_payment_link_id, razorpay_payment_id, email)
            VALUES (?, ?, ?)
            """,
            (razorpay_payment_link_id, razorpay_payment_id, email),
        )
    return True


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

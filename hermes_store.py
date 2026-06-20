"""SQLite persistence for Hermes Relay.

The GitHub Actions runner filesystem is ephemeral, so the workflow restores/saves
this database via actions/cache. Locally, it defaults to ./hermes_relay.db.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = "hermes_relay.db"
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db_path() -> Path:
    return Path(os.getenv("HERMES_RELAY_DB", DEFAULT_DB_PATH))


def article_hash(title: str, link: str) -> str:
    key = f"{title.strip()}-{link.strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_hash TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            link TEXT NOT NULL,
            source_feed TEXT,
            published TEXT,
            summary TEXT,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            times_seen INTEGER NOT NULL DEFAULT 1,
            used_in_briefing INTEGER NOT NULL DEFAULT 0
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_link ON articles(link);
        CREATE INDEX IF NOT EXISTS idx_articles_first_seen_at ON articles(first_seen_at);
        CREATE INDEX IF NOT EXISTS idx_articles_used_in_briefing ON articles(used_in_briefing);

        CREATE TABLE IF NOT EXISTS briefings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT NOT NULL,
            lens TEXT,
            model TEXT,
            json_path TEXT,
            html_path TEXT,
            top_articles TEXT,
            email_attempted INTEGER NOT NULL DEFAULT 0,
            email_sent INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def record_article(
    conn: sqlite3.Connection,
    *,
    title: str,
    link: str,
    source_feed: str | None = None,
    published: str | None = None,
    summary: str | None = None,
    seen_at: str | None = None,
) -> tuple[bool, str]:
    """Insert or update an article.

    Returns (is_new, article_hash). Dedupe uses both the historical title+link hash
    and a unique link index so minor title changes do not create repeat posts.
    """

    now = seen_at or utc_now()
    uid = article_hash(title, link)
    try:
        conn.execute(
            """
            INSERT INTO articles (
                article_hash, title, link, source_feed, published, summary,
                first_seen_at, last_seen_at, times_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (uid, title, link, source_feed, published, summary, now, now),
        )
        conn.commit()
        return True, uid
    except sqlite3.IntegrityError:
        conn.execute(
            """
            UPDATE articles
            SET last_seen_at = ?,
                times_seen = times_seen + 1,
                source_feed = COALESCE(source_feed, ?),
                published = COALESCE(published, ?),
                summary = CASE
                    WHEN summary IS NULL OR summary = '' THEN COALESCE(?, summary)
                    ELSE summary
                END
            WHERE article_hash = ? OR link = ?
            """,
            (now, source_feed, published, summary, uid, link),
        )
        conn.commit()
        return False, uid


def article_exists(conn: sqlite3.Connection, *, title: str, link: str) -> bool:
    uid = article_hash(title, link)
    row = conn.execute(
        "SELECT 1 FROM articles WHERE article_hash = ? OR link = ? LIMIT 1",
        (uid, link),
    ).fetchone()
    return row is not None


def import_legacy_signal_files(conn: sqlite3.Connection, pattern: str = "hermes_signal_*.json") -> int:
    """Backfill the database from checked-in/generated JSON signal files once."""

    already_done = conn.execute(
        "SELECT value FROM metadata WHERE key = 'legacy_signal_import_completed'"
    ).fetchone()
    if already_done:
        return 0

    imported = 0
    for file_path in sorted(glob.glob(pattern)):
        try:
            articles = json.loads(Path(file_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: could not import {file_path}: {exc}")
            continue

        for article in articles:
            title = str(article.get("title", "")).strip()
            link = str(article.get("link", "")).strip()
            if not title or not link:
                continue
            is_new, _ = record_article(
                conn,
                title=title,
                link=link,
                source_feed=article.get("source_feed") or f"legacy:{file_path}",
                published=article.get("published"),
                summary=article.get("summary"),
            )
            if is_new:
                imported += 1
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('legacy_signal_import_completed', ?)",
        (utc_now(),),
    )
    conn.commit()
    return imported


def mark_articles_used(conn: sqlite3.Connection, articles: list[dict[str, Any]]) -> None:
    for article in articles:
        title = str(article.get("title", "")).strip()
        link = str(article.get("link", "")).strip()
        if not title or not link:
            continue
        uid = article_hash(title, link)
        conn.execute(
            "UPDATE articles SET used_in_briefing = 1 WHERE article_hash = ? OR link = ?",
            (uid, link),
        )
    conn.commit()


def record_briefing(
    conn: sqlite3.Connection,
    *,
    run_date: str,
    lens: str | None,
    model: str | None,
    json_path: str | None,
    html_path: str | None,
    top_articles: str | None,
    email_attempted: bool = False,
    email_sent: bool = False,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO briefings (
            run_date, lens, model, json_path, html_path, top_articles,
            email_attempted, email_sent, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_date,
            lens,
            model,
            json_path,
            html_path,
            top_articles,
            int(email_attempted),
            int(email_sent),
            utc_now(),
        ),
    )
    briefing_id = cur.lastrowid
    conn.commit()
    if briefing_id is None:
        raise RuntimeError("SQLite did not return a briefing row id")
    return int(briefing_id)


def stats(conn: sqlite3.Connection) -> dict[str, int]:
    article_count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    used_count = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE used_in_briefing = 1"
    ).fetchone()[0]
    briefing_count = conn.execute("SELECT COUNT(*) FROM briefings").fetchone()[0]
    return {
        "articles": int(article_count),
        "articles_used_in_briefings": int(used_count),
        "briefings": int(briefing_count),
    }

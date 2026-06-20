import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from hermes_store import (
    article_exists,
    connect,
    import_legacy_signal_files,
    record_article,
    record_briefing,
    stats,
)


class HermesStoreTests(unittest.TestCase):
    def test_record_article_dedupes_by_hash_and_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "relay.db")
            try:
                is_new, first_hash = record_article(
                    conn,
                    title="Important breach",
                    link="https://example.com/breach",
                    source_feed="feed-a",
                    published="today",
                    summary="first",
                )
                self.assertTrue(is_new)

                is_new, second_hash = record_article(
                    conn,
                    title="Important breach",
                    link="https://example.com/breach",
                    source_feed="feed-a",
                    published="today",
                    summary="duplicate",
                )
                self.assertFalse(is_new)
                self.assertEqual(first_hash, second_hash)
                self.assertTrue(
                    article_exists(
                        conn,
                        title="Important breach",
                        link="https://example.com/breach",
                    )
                )

                row = conn.execute(
                    "SELECT times_seen FROM articles WHERE link = ?",
                    ("https://example.com/breach",),
                ).fetchone()
                self.assertEqual(row[0], 2)
                self.assertEqual(stats(conn)["articles"], 1)
            finally:
                conn.close()

    def test_import_legacy_signal_files_backfills_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                Path("hermes_signal_2026-01-01.json").write_text(
                    json.dumps(
                        [
                            {
                                "title": "One",
                                "link": "https://example.com/one",
                                "published": "2026-01-01",
                                "summary": "summary",
                            },
                            {
                                "title": "One",
                                "link": "https://example.com/one",
                                "published": "2026-01-01",
                                "summary": "summary",
                            },
                        ]
                    ),
                    encoding="utf-8",
                )
                conn = connect("relay.db")
                try:
                    imported = import_legacy_signal_files(conn)
                    self.assertEqual(imported, 1)
                    self.assertEqual(stats(conn)["articles"], 1)
                finally:
                    conn.close()
            finally:
                os.chdir(cwd)

    def test_record_briefing_persists_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "relay.db")
            try:
                briefing_id = record_briefing(
                    conn,
                    run_date="2026-01-01",
                    lens="Detection engineering",
                    model="gemini-test",
                    json_path="json_output/test.json",
                    html_path="json_output/test.html",
                    top_articles="draft",
                    email_attempted=True,
                    email_sent=False,
                )
                self.assertGreater(briefing_id, 0)
                self.assertEqual(stats(conn)["briefings"], 1)
                row = conn.execute("SELECT lens, email_attempted, email_sent FROM briefings").fetchone()
                self.assertEqual(row["lens"], "Detection engineering")
                self.assertEqual(row["email_attempted"], 1)
                self.assertEqual(row["email_sent"], 0)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()

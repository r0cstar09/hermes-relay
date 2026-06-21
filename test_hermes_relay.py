import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

from hermes_store import connect, mark_articles_used, record_article


MODULE_PATH = Path(__file__).with_name("hermes-relay.py")


def load_relay_module():
    fake_feedparser = types.ModuleType("feedparser")
    setattr(fake_feedparser, "parse", lambda url: None)
    sys.modules.setdefault("feedparser", fake_feedparser)
    spec = importlib.util.spec_from_file_location("hermes_relay_script", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HermesRelayCollectorTests(unittest.TestCase):
    def test_load_unbriefed_backlog_excludes_already_briefed_articles(self):
        relay = load_relay_module()
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "relay.db")
            try:
                record_article(
                    conn,
                    title="Unused but still relevant",
                    link="https://example.com/unused",
                    source_feed="feed",
                    published="today",
                    summary="summary",
                    seen_at="2026-06-21T10:00:00+00:00",
                )
                record_article(
                    conn,
                    title="Already briefed",
                    link="https://example.com/used",
                    source_feed="feed",
                    published="today",
                    summary="summary",
                    seen_at="2026-06-21T10:00:00+00:00",
                )
                mark_articles_used(
                    conn,
                    [{"title": "Already briefed", "link": "https://example.com/used"}],
                )

                backlog = relay.load_unbriefed_backlog(
                    conn, since="2026-06-21T00:00:00+00:00", limit=10
                )

                self.assertEqual([item["title"] for item in backlog], ["Unused but still relevant"])
                self.assertEqual(backlog[0]["source_feed"], "feed")
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()

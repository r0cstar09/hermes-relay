import json
import os
import tempfile
import unittest
from pathlib import Path

from publish_blog_post import parse_articles, choose_top_article, slugify, main


SAMPLE_LLM = """1) Critical VPN Bug Now Exploited
Score: 9/10

Key Takeaways:
- Active exploitation changes this from patch hygiene to incident triage.
- Internet-facing VPNs should be checked before normal maintenance windows.

Angle for this story:
Active exploitation vs. theoretical risk

One-Line Board Take:
Treat exposed VPN appliances as incident-response scope, not routine patch backlog.

Article Summary:
A vendor disclosed a critical VPN vulnerability that is now being exploited. The affected systems sit at the network edge and can expose privileged access paths when defenders respond slowly.

Briefing - Variant A:
The important shift is exploitation. Once a network-edge vulnerability moves from advisory language to active use, the operational question changes from whether to patch to whether anyone already touched the environment.

Briefing - Variant B:
Network-edge bugs compress response timelines because they sit in front of identity, remote access, and monitoring assumptions.

---

2) Lower Priority Malware Report
Score: 6/10

Key Takeaways:
- Useful context, but less urgent.

Article Summary:
A malware report described commodity tradecraft.
"""


class PublishBlogPostTests(unittest.TestCase):
    def test_slugify_keeps_safe_human_readable_slug(self):
        self.assertEqual(slugify("Critical VPN Bug: Now Exploited!"), "critical-vpn-bug-now-exploited")

    def test_parse_articles_matches_source_and_selects_highest_score(self):
        articles = [{"title": "Critical VPN Bug Now Exploited", "link": "https://example.com/vpn"}]
        parsed = parse_articles(SAMPLE_LLM, articles)
        self.assertEqual(len(parsed), 2)
        selected = choose_top_article(parsed)
        self.assertEqual(selected.title, "Critical VPN Bug Now Exploited")
        self.assertEqual(selected.score, 9)
        self.assertEqual(selected.source_url, "https://example.com/vpn")
        self.assertIsNotNone(selected.board_take)
        self.assertIn("incident-response", selected.board_take or "")

    def test_main_writes_schema_compatible_astro_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = root / "opposite-osiris"
            blog = site / "src/content/blog"
            blog.mkdir(parents=True)
            briefing_dir = root / "json_output/2026-06-20"
            briefing_dir.mkdir(parents=True)
            briefing = briefing_dir / "hermes_llm_top3_2026-06-20.json"
            briefing.write_text(
                json.dumps({"date": "2026-06-20", "lens": "What I would do next", "top_articles": SAMPLE_LLM}),
                encoding="utf-8",
            )
            signal = root / "hermes_signal_2026-06-20.json"
            signal.write_text(
                json.dumps([{"title": "Critical VPN Bug Now Exploited", "link": "https://example.com/vpn"}]),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            try:
                os.chdir(root)
                rc = main(["--briefing-json", str(briefing), "--site-dir", str(site)])
            finally:
                os.chdir(cwd)
            self.assertEqual(rc, 0)
            posts = list(blog.glob("2026-06-20-critical-vpn-bug-now-exploited.md"))
            self.assertEqual(len(posts), 1)
            text = posts[0].read_text(encoding="utf-8")
            self.assertIn('title: "Critical VPN Bug Now Exploited"', text)
            self.assertIn('publishDate: "2026-06-20"', text)
            self.assertIn("Source: [Critical VPN Bug Now Exploited](https://example.com/vpn)", text)


if __name__ == "__main__":
    unittest.main()

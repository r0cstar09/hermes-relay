import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from publish_blog_post import (
    _image_bytes_from_generated_image,
    build_hero_image_prompt,
    build_markdown,
    build_editor_prompt,
    choose_top_article,
    main,
    parse_articles,
    slugify,
    split_frontmatter,
)


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

    def test_split_frontmatter_keeps_schema_separate_from_editor_body(self):
        markdown = '---\ntitle: "A"\npublishDate: "2026-06-20"\n---\n\nBody text'
        frontmatter, body = split_frontmatter(markdown)
        self.assertIn('title: "A"', frontmatter)
        self.assertEqual(body, "\nBody text")

    def test_editor_prompt_uses_voice_profile_and_forbids_new_facts(self):
        selected = choose_top_article(parse_articles(SAMPLE_LLM, [{"title": "Critical VPN Bug Now Exploited", "link": "https://example.com/vpn"}]))
        prompt = build_editor_prompt(body="## Draft\nSource: https://example.com/vpn", voice_profile="Tony says the dashboard is not the control.", article=selected)
        self.assertIn("Tony says the dashboard is not the control", prompt)
        self.assertIn("Do not add facts", prompt)
        self.assertIn("https://example.com/vpn", prompt)

    def test_markdown_can_include_generated_hero_image_frontmatter(self):
        selected = choose_top_article(parse_articles(SAMPLE_LLM, []))
        markdown = build_markdown(
            selected,
            run_date="2026-06-20",
            lens=None,
            model="gemini-2.5-flash",
            hero_image="/assets/blog/hermes-relay/2026-06-20-critical-vpn-bug-now-exploited.png",
            hero_image_alt="Abstract cyber defense illustration for Critical VPN Bug Now Exploited",
        )
        self.assertIn('img: "/assets/blog/hermes-relay/2026-06-20-critical-vpn-bug-now-exploited.png"', markdown)
        self.assertIn('img_alt: "Abstract cyber defense illustration for Critical VPN Bug Now Exploited"', markdown)

    def test_hero_image_prompt_avoids_text_logos_and_people(self):
        selected = choose_top_article(parse_articles(SAMPLE_LLM, []))
        prompt = build_hero_image_prompt(selected)
        self.assertIn("16:9", prompt)
        self.assertIn("no humans", prompt)
        self.assertIn("no company logos", prompt)
        self.assertIn("no readable text", prompt)

    def test_image_bytes_helper_accepts_bytes_and_base64(self):
        raw = b"fake-image-bytes"
        self.assertEqual(_image_bytes_from_generated_image(SimpleNamespace(image=SimpleNamespace(image_bytes=raw))), raw)
        self.assertEqual(_image_bytes_from_generated_image(SimpleNamespace(image=SimpleNamespace(image_bytes="ZmFrZS1pbWFnZS1ieXRlcw=="))), raw)

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

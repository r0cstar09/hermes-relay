import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from repair_blog_images import AD_PROJECT, AD_CAPTION, AD_SCREENSHOT, LEGACY_MARKER, legacy_posts, replace_image_fields, stage_one
from verify_publication import frontmatter


POST = '''---
title: "Supermarket refrigeration controls disrupted"
description: "Remote freezer controls and physical remediation."
publishDate: "2026-11-02"
tags: ["cyber"]
img: "/assets/blog/hermes-relay/2026-11-02-freezers.svg"
img_alt: "Old generic cover"
---
The refrigeration controller needs a physical modification, not just a patch.
Source: [Original report](https://example.com/report)
'''


class RepairBlogImagesTests(unittest.TestCase):
    def make_site(self, root):
        site = root / 'site'
        post = site / 'src/content/blog/2026-11-02-freezers.md'
        post.parent.mkdir(parents=True)
        post.write_text(POST)
        asset = site / 'public/assets/blog/hermes-relay/2026-11-02-freezers.svg'
        asset.parent.mkdir(parents=True)
        asset.write_text('<svg><desc>' + LEGACY_MARKER + '</desc></svg>')
        return site, post

    def test_selects_only_legacy_shield_covers(self):
        with tempfile.TemporaryDirectory() as tmp:
            site, post = self.make_site(Path(tmp))
            other = post.with_name('unrelated.md')
            other.write_text(POST.replace('.svg', '.png'))
            draft = post.with_name('draft.md')
            draft.write_text(POST.replace('tags: ', 'draft: true\ntags: '))
            self.assertEqual(legacy_posts(site), [post])

    def test_preserves_prose_and_non_image_metadata(self):
        before, body = frontmatter(POST)
        updated = replace_image_fields(POST, '/assets/new.png', 'A freezer controller')
        after, new_body = frontmatter(updated)
        self.assertEqual(body, new_body)
        for key in before.keys() - {'img', 'img_alt'}:
            self.assertEqual(before[key], after[key])
        self.assertEqual(after['img'], '/assets/new.png')

    def test_stages_manifest_data_and_full_article_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site, post = self.make_site(root)
            # Unit fixture: provider decoding/format checks belong to the image-generator tests.
            fixture = b'unit-test-image-output'
            def fake_generate(**kwargs):
                self.assertIn('physical modification', kwargs['article'].summary)
                self.assertEqual(kwargs['article'].title, 'Supermarket refrigeration controls disrupted')
                kwargs['output_path'].write_bytes(fixture)
            with patch('repair_blog_images.generate_hero_image_with_vertex', side_effect=fake_generate):
                row = stage_one(site, post, root / 'output', 'test-model')
            self.assertEqual(row['before_markdown_sha256'], hashlib.sha256(POST.encode()).hexdigest())
            self.assertEqual(row['image_sha256'], hashlib.sha256(fixture).hexdigest())
            self.assertEqual((root/'output/site'/row['post']).read_bytes(), post.read_bytes())
            self.assertEqual((root/'output/site/public'/row['image'].lstrip('/')).read_bytes(), fixture)
            self.assertEqual(frontmatter(post.read_text())[1], frontmatter(POST)[1])

    def test_ad_cover_keeps_dashboard_inside_the_case_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = root / 'site'
            post = site / AD_PROJECT
            post.parent.mkdir(parents=True)
            original = POST.replace('/assets/blog/hermes-relay/2026-11-02-freezers.svg', '/images/ad-account-hygiene/02-dashboard.png') + '\n' + AD_CAPTION + '\n'
            post.write_text(original)
            def fake_generate(**kwargs):
                kwargs['output_path'].parent.mkdir(parents=True)
                kwargs['output_path'].write_bytes(b'unit-test-ad-cover')
            with patch('repair_blog_images.generate_hero_image_with_vertex', side_effect=fake_generate):
                row = stage_one(site, post, root / 'output', 'test-model')
            self.assertTrue(row['image'].startswith('/images/ad-account-hygiene/'))
            self.assertTrue(row['restored_dashboard_screenshot'])
            new_body = frontmatter(post.read_text())[1]
            self.assertEqual(new_body.replace(AD_SCREENSHOT, ''), frontmatter(original)[1])
            self.assertEqual(new_body.count(AD_SCREENSHOT), 1)

    def test_generation_failure_does_not_rewrite_article(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site, post = self.make_site(root)
            with patch('repair_blog_images.generate_hero_image_with_vertex', side_effect=RuntimeError('unavailable')):
                with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                    stage_one(site, post, root / 'output', 'test-model')
            self.assertEqual(post.read_text(), POST)
            self.assertFalse((root / 'output').exists())


if __name__ == '__main__':
    unittest.main()

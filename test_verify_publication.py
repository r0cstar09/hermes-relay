import json
from pathlib import Path
import tempfile
import unittest
from verify_publication import daily_post, verify_once


class PublicationGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.site = Path(self.temp.name)
        self.blog = self.site/'src/content/blog'; self.blog.mkdir(parents=True)
        self.asset = self.site/'public/assets/cover.svg'; self.asset.parent.mkdir(parents=True)
        self.asset.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"/>')
        self.post = self.blog/'2026-10-26-example.md'
        self.text = '---\ntitle: "A & B"\npublishDate: "2026-10-26"\nimg: "/assets/cover.svg"\n---\nRead [the source](https://example.org/source).\n'
        self.post.write_text(self.text)
        self.html = b'<h1>A &amp; B</h1><a href="https://example.org/source">Source</a>'
    def getter(self,url):
        return self.asset.read_bytes() if url.endswith('.svg') else self.html
    def test_preflight_finds_existing_post(self):
        self.assertEqual(daily_post(self.site,'2026-10-26'),self.post)
        self.assertIsNone(daily_post(self.site,'2026-10-27'))
    def test_drafts_do_not_count_as_published(self):
        self.post.write_text(self.text.replace('publishDate:', 'draft: true\npublishDate:'))
        self.assertIsNone(daily_post(self.site,'2026-10-26'))
    def test_duplicate_day_refuses_another_publication(self):
        (self.blog/'2026-10-26-other.md').write_text(self.text)
        with self.assertRaisesRegex(ValueError,'Multiple'): daily_post(self.site,'2026-10-26')
    def test_verifies_title_source_and_exact_image(self):
        receipt=verify_once(self.site,self.post,'https://www.tonymuzo.dev',self.getter)
        self.assertEqual(receipt['title'],'A & B')
        self.assertEqual(receipt['source_urls'],['https://example.org/source'])
        self.assertEqual(len(receipt['image_sha256']),64)
    def test_rejects_stale_or_missing_article(self):
        self.html=b'<h1>August article</h1>'
        with self.assertRaisesRegex(ValueError,'heading'): verify_once(self.site,self.post,'https://www.tonymuzo.dev',self.getter)
    def test_rejects_missing_source(self):
        self.html=b'<h1>A &amp; B</h1>'
        with self.assertRaisesRegex(ValueError,'Source links'): verify_once(self.site,self.post,'https://www.tonymuzo.dev',self.getter)
    def test_rejects_missing_or_changed_image(self):
        with self.assertRaisesRegex(ValueError,'differs'):
            verify_once(self.site,self.post,'https://www.tonymuzo.dev',lambda url:b'wrong' if url.endswith('.svg') else self.html)
    def test_rejects_image_path_traversal(self):
        self.post.write_text(self.text.replace('/assets/cover.svg','/../private.svg'))
        with self.assertRaisesRegex(ValueError,'Invalid image'):
            verify_once(self.site,self.post,'https://www.tonymuzo.dev',self.getter)

if __name__=='__main__': unittest.main()

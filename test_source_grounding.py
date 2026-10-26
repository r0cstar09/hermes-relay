import json
import unittest
from typing import Any
from source_grounding import SourceText, check_source_url, validate_verdict, assert_supported_identifiers, validate_reader_copy

class GroundingTests(unittest.TestCase):
    def test_html_extraction_excludes_active_and_navigation_text(self):
        parser=SourceText(); parser.feed('<nav>Menu</nav><script>attack()</script><article><h1>Title</h1><p>Real &amp; useful evidence.</p></article>')
        self.assertIn('Real & useful evidence.',parser.text())
        self.assertNotIn('attack',parser.text()); self.assertNotIn('Menu',parser.text())
    def test_source_urls_fail_closed(self):
        check_source_url('https://openai.com/index/story/')
        for url in ('http://openai.com/','https://127.0.0.1/','https://openai.com.evil.test/','https://a@openai.com/','https://openai.com:8443/'):
            with self.subTest(url=url), self.assertRaises(ValueError): check_source_url(url)
    def test_review_requires_all_boolean_approvals(self):
        good=dict(publishable=True,grounded=True,specific=True,non_generic=True,reasons=[])
        validate_verdict(json.dumps(good))
        for key in ('publishable','grounded','specific','non_generic'):
            for invalid in (False,'true',None):
                bad: dict[str, Any] = dict(good); bad[key]=invalid
                with self.subTest(key=key,value=invalid), self.assertRaises(ValueError): validate_verdict(json.dumps(bad))
    def test_invented_cve_is_rejected(self):
        assert_supported_identifiers('CVE-2026-12345','Source CVE-2026-12345')
        with self.assertRaises(ValueError): assert_supported_identifiers('CVE-2026-99999','Source CVE-2026-12345')

    def test_reader_rejects_wrong_source_and_interstitials(self):
        url='https://openai.com/index/story'
        text='Title: Actual story\nURL Source: '+url+'\nMarkdown Content:\n'+('Evidence '*110)
        validate_reader_copy(text,url)
        for bad in (text.replace(url,'https://example.com/'),text.replace('Actual story','Just a moment'),text.replace('Evidence '*110,'Too short')):
            with self.assertRaises(ValueError): validate_reader_copy(bad,url)

if __name__=='__main__': unittest.main()

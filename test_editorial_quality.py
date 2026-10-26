import unittest
from editorial_quality import validate_article

class EditorialQualityTests(unittest.TestCase):
    def setUp(self):
        self.url='https://example.com/source'
        self.body=('Concrete evidence guides a bounded operational decision. '*40)+'\nSource: '+self.url
    def test_structurally_valid_copy_passes(self):
        validate_article(self.body,source_url=self.url,score=8)
    def test_missing_source_low_score_or_thin_copy_fails(self):
        for body,score in ((self.body.replace(self.url,''),8),(self.body,5),(self.url,9)):
            with self.subTest(score=score,body_length=len(body)), self.assertRaises(ValueError):
                validate_article(body,source_url=self.url,score=score)
    def test_canned_template_and_active_html_fail(self):
        for suffix in ('The headline is the easy part.','The headline about a critical PHP vulnerability is the easy part.','This is really an ownership problem.','<script>alert(1)</script>','\nPipeline note: generated'):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                validate_article(self.body+'\n'+suffix,source_url=self.url,score=8)

if __name__=='__main__': unittest.main()

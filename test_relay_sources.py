from datetime import datetime, timedelta, timezone
import unittest
from relay_sources import AI_FEEDS, balanced_candidates, is_current

class SourceGateTests(unittest.TestCase):
    def setUp(self): self.now=datetime(2026,10,26,18,tzinfo=timezone.utc)
    def test_rss_and_atom_dates(self):
        self.assertTrue(is_current('Mon, 26 Oct 2026 13:45:00 GMT',self.now))
        self.assertTrue(is_current('2026-10-25T15:10:19+00:00',self.now))
    def test_old_undated_future_items_are_rejected(self):
        for value in ('',None,'today','2026-08-19','2026-10-28T00:00:00Z'):
            with self.subTest(value=value): self.assertFalse(is_current(value,self.now))
    def test_seven_day_boundary(self):
        self.assertTrue(is_current((self.now-timedelta(days=7)).isoformat(),self.now))
        self.assertFalse(is_current((self.now-timedelta(days=7,seconds=1)).isoformat(),self.now))
    def test_candidate_cap_balances_sources_and_deduplicates(self):
        rows=[{'link':f'https://example.com/{i}','source_feed':'incident'} for i in range(30)]
        ai={'link':'https://example.com/ai','source_feed':AI_FEEDS[0]}
        rows += [ai,ai]
        selected=balanced_candidates(rows,limit=8)
        self.assertEqual(selected[0],ai)
        self.assertEqual(len(selected),8)
        self.assertEqual(len({r['link'] for r in selected}),8)

if __name__=='__main__': unittest.main()

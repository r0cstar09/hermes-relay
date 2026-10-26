"""Bounded source retrieval and freshness gates for the daily blog."""
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.request import Request, urlopen
from collections import defaultdict, deque

AI_FEEDS = [
    'https://openai.com/news/rss.xml',
    'https://simonwillison.net/atom/everything/',
    'https://huggingface.co/blog/feed.xml',
]
MAX_PER_SOURCE = 8
MAX_CANDIDATES = 40


def is_current(published, now=None):
    now = now or datetime.now(timezone.utc)
    value = str(published or '').strip()
    if not value: return False
    try:
        try: stamp = parsedate_to_datetime(value)
        except (ValueError, TypeError): stamp = datetime.fromisoformat(value.replace('Z','+00:00'))
        if stamp.tzinfo is None: stamp = stamp.replace(tzinfo=timezone.utc)
        return now-timedelta(days=7) <= stamp <= now+timedelta(hours=1)
    except (ValueError, TypeError, OverflowError):
        return False


def fetch_feed_bytes(url):
    request = Request(url,headers={'User-Agent':'Mozilla/5.0 (Hermes Relay; daily editorial RSS reader)'})
    with urlopen(request,timeout=20) as response:
        if response.status != 200: raise ValueError('Feed did not return HTTP 200')
        raw = response.read(3_000_001)
        if len(raw)>3_000_000: raise ValueError('Feed exceeds the bounded response budget')
        return raw


def balanced_candidates(articles, limit=MAX_CANDIDATES):
    queues = defaultdict(deque)
    seen = set()
    for article in articles:
        link = article.get('link')
        if link in seen: continue
        seen.add(link)
        queues[article.get('source_feed','unknown')].append(article)
    # Give fresh primary AI / builder material representation before incident feeds.
    order = sorted(queues, key=lambda key:(key not in AI_FEEDS,key))
    result = []
    while len(result)<limit and any(queues.values()):
        for key in order:
            if queues[key] and len(result)<limit: result.append(queues[key].popleft())
    return result

import feedparser
import hashlib
import json
from datetime import datetime

# List of RSS feeds
FEEDS = [
    "https://krebsonsecurity.com/feed/",
    "https://www.bleepingcomputer.com/feed/",
    "https://www.mandiant.com/resources/rss",
    "https://www.microsoft.com/en-us/security/blog/feed/",
    "https://blog.google/threat-analysis-group/rss/",
    "https://unit42.paloaltonetworks.com/feed/",
    "https://www.cisa.gov/alerts.xml",
    "https://www.cisa.gov/advisories.xml",
    "https://www.darkreading.com/rss.xml"
]

# Output file
TODAY = datetime.utcnow().strftime("%Y-%m-%d")
OUTPUT_JSON = f"hermes_signal_{TODAY}.json"

# Holds unique items
seen_hashes = set()
all_items = []

def hash_item(title, link):
    """Unique hash per entry to dedupe."""
    key = f"{title}-{link}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def fetch_and_parse():
    """Fetch all feeds and parse items."""
    for url in FEEDS:
        print(f"Fetching: {url}")
        feed = feedparser.parse(url)
        for item in feed.entries:
            title = item.get("title", "").strip()
            link = item.get("link", "").strip()
            published = item.get("published", item.get("updated", ""))
            summary = item.get("summary", "")

            uid = hash_item(title, link)
            if uid in seen_hashes:
                continue

            seen_hashes.add(uid)

            all_items.append({
                "title": title,
                "link": link,
                "published": published,
                "summary": summary
            })

def save_output():
    """Save the combined results to JSON."""
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_items)} items to {OUTPUT_JSON}")

if __name__ == "__main__":
    fetch_and_parse()
    save_output()
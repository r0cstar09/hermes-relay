import feedparser
import hashlib
import json
import glob
from datetime import datetime
from pathlib import Path

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

def load_previous_articles():
    """Load all previously seen articles from past JSON files."""
    previous_hashes = set()
    previous_count = 0
    
    # Find all previous hermes_signal JSON files (excluding today's)
    previous_files = sorted(glob.glob("hermes_signal_*.json"), reverse=True)
    
    for file_path in previous_files:
        # Skip today's file if it exists
        if file_path == OUTPUT_JSON:
            continue
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                articles = json.load(f)
                for article in articles:
                    title = article.get("title", "").strip()
                    link = article.get("link", "").strip()
                    if title and link:
                        uid = hash_item(title, link)
                        previous_hashes.add(uid)
                        previous_count += 1
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue
    
    print(f"Loaded {previous_count} previously seen articles from {len(previous_files)} file(s)")
    return previous_hashes

def fetch_and_parse():
    """Fetch all feeds and parse only new items."""
    # Load previously seen articles
    previous_hashes = load_previous_articles()
    seen_hashes.update(previous_hashes)
    
    new_count = 0
    skipped_count = 0
    
    for url in FEEDS:
        print(f"Fetching: {url}")
        try:
            feed = feedparser.parse(url)
            for item in feed.entries:
                title = item.get("title", "").strip()
                link = item.get("link", "").strip()
                published = item.get("published", item.get("updated", ""))
                summary = item.get("summary", "")

                if not title or not link:
                    continue

                uid = hash_item(title, link)
                if uid in seen_hashes:
                    skipped_count += 1
                    continue

                seen_hashes.add(uid)
                new_count += 1

                all_items.append({
                    "title": title,
                    "link": link,
                    "published": published,
                    "summary": summary
                })
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            continue
    
    print(f"\nFound {new_count} new articles, skipped {skipped_count} previously seen articles")

def save_output():
    """Save the combined results to JSON."""
    if not all_items:
        print(f"No new articles to save. Output file will not be created.")
        return
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_items)} new items to {OUTPUT_JSON}")

if __name__ == "__main__":
    fetch_and_parse()
    save_output()
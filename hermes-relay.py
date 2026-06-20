import json
from datetime import datetime

import feedparser

from hermes_store import connect, import_legacy_signal_files, record_article, stats

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
    "https://www.darkreading.com/rss.xml",
]

# Output file
TODAY = datetime.utcnow().strftime("%Y-%m-%d")
OUTPUT_JSON = f"hermes_signal_{TODAY}.json"

# Holds current-run new items only
all_items = []


def bootstrap_database():
    """Open the persistent SQLite DB and import any legacy JSON signals once."""
    conn = connect()
    imported = import_legacy_signal_files(conn)
    current_stats = stats(conn)
    print(
        "SQLite persistence ready: "
        f"{current_stats['articles']} article(s), "
        f"{current_stats['briefings']} briefing(s). "
        f"Imported {imported} legacy article(s) from hermes_signal_*.json."
    )
    return conn


def fetch_and_parse():
    """Fetch all feeds, persist every seen item, and output only first-seen items."""
    conn = bootstrap_database()
    new_count = 0
    skipped_count = 0

    try:
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

                    is_new, _ = record_article(
                        conn,
                        title=title,
                        link=link,
                        source_feed=url,
                        published=published,
                        summary=summary,
                    )
                    if not is_new:
                        skipped_count += 1
                        continue

                    new_count += 1
                    all_items.append(
                        {
                            "title": title,
                            "link": link,
                            "published": published,
                            "summary": summary,
                            "source_feed": url,
                        }
                    )
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                continue
    finally:
        conn.close()

    print(
        f"\nFound {new_count} first-seen article(s), "
        f"skipped {skipped_count} article(s) already in SQLite history"
    )


def save_output():
    """Save current-run first-seen articles to JSON for the LLM step/artifacts."""
    if not all_items:
        print("No first-seen articles to save. Output file will not be created.")
        return

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_items)} first-seen item(s) to {OUTPUT_JSON}")


if __name__ == "__main__":
    fetch_and_parse()
    save_output()

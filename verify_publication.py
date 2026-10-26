"""Idempotency and live-deployment gate for the daily website publisher."""
from __future__ import annotations
import argparse
from datetime import date, datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen


class PageHeadings(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_h1 = False
        self.headings = []
        self.current = []
    def handle_starttag(self, tag, attrs):
        if tag == 'h1': self.in_h1 = True; self.current = []
    def handle_endtag(self, tag):
        if tag == 'h1' and self.in_h1:
            self.headings.append(' '.join(''.join(self.current).split()))
            self.in_h1 = False
    def handle_data(self, data):
        if self.in_h1: self.current.append(data)


def frontmatter(text):
    if not text.startswith('---\n'): raise ValueError('Missing frontmatter')
    header, body = text[4:].split('\n---\n', 1)
    fields = {}
    for line in header.splitlines():
        key, sep, value = line.partition(':')
        if not sep: continue
        value = value.strip()
        try: fields[key] = json.loads(value)
        except json.JSONDecodeError: fields[key] = value.strip("'\"")
    return fields, body


def daily_post(site_dir, run_date):
    date.fromisoformat(run_date)
    matches = []
    for path in sorted((Path(site_dir)/'src/content/blog').glob(run_date+'-*.md')):
        fields, _ = frontmatter(path.read_text())
        if str(fields.get('publishDate')) == run_date and fields.get('draft') not in (True, 'true'):
            matches.append(path)
    if len(matches) > 1: raise ValueError('Multiple posts already exist for this date; refusing to choose or add another')
    return matches[0] if matches else None


def fetch(url):
    with urlopen(Request(url, headers={'User-Agent':'Mozilla/5.0 (Hermes publication verifier)', 'Cache-Control':'no-cache'}), timeout=20) as response:
        if response.status != 200: raise ValueError('Expected HTTP 200')
        return response.read()


def verify_once(site_dir, post_path, base_url, getter=fetch):
    fields, body = frontmatter(post_path.read_text())
    title = ' '.join(str(fields['title']).split())
    url = base_url.rstrip('/') + '/blog/' + post_path.stem + '/'
    html = getter(url).decode('utf-8')
    parser = PageHeadings(); parser.feed(html)
    if title not in parser.headings: raise ValueError('Live article heading does not match the committed post')
    sources = sorted(set(re.findall(r'https://[^\s)<>"\]]+', body)))
    if not sources: raise ValueError('No source links in the committed article')
    if not any(source.replace('&','&amp;') in html or source in html for source in sources):
        raise ValueError('Source links are missing from the live article')
    image = fields.get('img')
    image_hash = None
    if image:
        if not isinstance(image,str) or not image.startswith('/') or image.startswith('//'):
            raise ValueError('Hero image must be a site-local asset')
        public = (Path(site_dir)/'public').resolve()
        local = (public/image.lstrip('/')).resolve()
        if not local.is_relative_to(public): raise ValueError('Invalid image path')
        expected = local.read_bytes()
        actual = getter(urljoin(base_url, image))
        if actual != expected: raise ValueError('Live hero image differs from committed bytes')
        image_hash = hashlib.sha256(actual).hexdigest()
    return {'title':fields['title'], 'date':str(fields['publishDate']), 'url':url,
            'source_urls':sources, 'image_sha256':image_hash}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight','verify'])
    parser.add_argument('--site-dir', required=True, type=Path)
    parser.add_argument('--date', default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument('--base-url', default='https://www.tonymuzo.dev')
    parser.add_argument('--receipt', type=Path, default=Path('publication-status.json'))
    parser.add_argument('--attempts', type=int, default=24)
    parser.add_argument('--interval', type=float, default=20)
    args = parser.parse_args(argv)
    parts = urlsplit(args.base_url)
    if parts.scheme != 'https' or not parts.netloc: raise ValueError('An HTTPS base URL is required')
    post = daily_post(args.site_dir, args.date)
    if args.mode == 'preflight':
        value = 'true' if post else 'false'
        if os.getenv('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'],'a') as out: out.write('already_published='+value+'\n')
        print(json.dumps({'already_published':bool(post),'post_path':str(post) if post else None}))
        return 0
    if not post: raise ValueError('No current-day published Markdown exists')
    if args.attempts < 1 or args.interval < 0: raise ValueError('Invalid retry budget')
    receipt = None
    for attempt in range(args.attempts):
        try:
            receipt = verify_once(args.site_dir,post,args.base_url)
            break
        except Exception as exc:
            print(f'Live verification {attempt+1}/{args.attempts}: {type(exc).__name__}: {exc}', flush=True)
            if attempt+1 == args.attempts: raise
            time.sleep(args.interval)
    if receipt is None: raise RuntimeError('Live verification did not produce a receipt')
    receipt['blog_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.site_dir,text=True).strip()
    receipt['verified_at'] = datetime.now(timezone.utc).isoformat()
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

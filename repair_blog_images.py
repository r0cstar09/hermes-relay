"""Stage story-specific replacements for legacy shield covers; never commit or push."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from publish_blog_post import ArticleBlock, generate_hero_image_with_vertex, yaml_quote
from verify_publication import frontmatter

LEGACY_MARKER = 'Editorial cyber defense illustration with a protected network edge and telemetry paths.'
DEFAULT_MODELS = 'gemini-3.1-flash-image,gemini-3-pro-image'
AD_PROJECT = Path('src/content/work/ad-account-hygiene-powershell-python.md')
AD_DASHBOARD = '/images/ad-account-hygiene/02-dashboard.png'
AD_CAPTION = '*The dashboard gives you a quick summary. The individual sheets give you the accounts to go through.*'
AD_SCREENSHOT = '![The dashboard showing the account groups and priorities.](' + AD_DASHBOARD + ')\n\n'


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def legacy_posts(site_dir: Path) -> list[Path]:
    public = (site_dir / 'public').resolve()
    matches = []
    for post in sorted((site_dir / 'src/content/blog').glob('*.md')):
        fields, _ = frontmatter(post.read_text(encoding='utf-8'))
        image = fields.get('img', '')
        if fields.get('draft') in (True, 'true') or not isinstance(image, str):
            continue
        if not image.startswith('/assets/blog/hermes-relay/') or not image.endswith('.svg'):
            continue
        asset = (public / image.lstrip('/')).resolve()
        if not asset.is_relative_to(public):
            raise ValueError('Image path escapes the public directory')
        if LEGACY_MARKER in asset.read_text(encoding='utf-8'):
            matches.append(post)
    return matches


def replace_image_fields(markdown: str, image: str, alt: str) -> str:
    header, body = markdown[4:].split('\n---\n', 1)
    for key, value in [('img', image), ('img_alt', alt)]:
        header, count = re.subn(r'^' + key + r':.*$', lambda _: key + ': ' + yaml_quote(value), header, flags=re.M)
        if count != 1:
            raise ValueError(f'Expected exactly one {key} frontmatter field')
    return '---\n' + header + '\n---\n' + body


def stage_one(site_dir: Path, post: Path, output_dir: Path, image_model: str) -> dict:
    before = post.read_bytes()
    markdown = before.decode('utf-8')
    fields, body = frontmatter(markdown)
    article = ArticleBlock(
        title=str(fields['title']), score=None, key_takeaways=[], angle=None,
        board_take=None, summary=body, variant_a=None, variant_b=None, source_url=None,
    )
    is_ad_project = post.relative_to(site_dir) == AD_PROJECT
    image_dir = site_dir / ('public/images/ad-account-hygiene' if is_ad_project else 'public/assets/blog/hermes-relay')
    temporary = image_dir / f'{post.stem}-story-new.png'
    if temporary.exists():
        raise FileExistsError(f'Refusing to overwrite an existing staging image: {temporary}')
    generate_hero_image_with_vertex(article=article, image_model=image_model, output_path=temporary)
    generated = temporary.read_bytes()
    image_hash = sha256(generated)
    final = temporary.with_name(f'{post.stem}-{image_hash[:12]}.png')
    if final.exists() and final.read_bytes() != generated:
        raise ValueError('Content-addressed image filename collision')
    temporary.replace(final)
    image = '/' + final.relative_to(site_dir / 'public').as_posix()
    updated = replace_image_fields(markdown, image, f'Editorial illustration for {fields["title"]}')
    expected_body = body
    if is_ad_project and AD_DASHBOARD not in body:
        if body.count(AD_CAPTION) != 1:
            raise ValueError('Cannot safely restore the AD dashboard screenshot at its caption')
        expected_body = body.replace(AD_CAPTION, AD_SCREENSHOT + AD_CAPTION)
        updated = updated.replace(AD_CAPTION, AD_SCREENSHOT + AD_CAPTION)
    updated_fields, updated_body = frontmatter(updated)
    protected_before = {k:v for k,v in fields.items() if k not in ('img', 'img_alt')}
    protected_after = {k:v for k,v in updated_fields.items() if k not in ('img', 'img_alt')}
    if updated_body != expected_body or protected_before != protected_after:
        raise ValueError('Image repair must preserve article text and all other metadata')
    post.write_text(updated, encoding='utf-8')
    for path in (post, final):
        staged = output_dir / 'site' / path.relative_to(site_dir)
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, staged)
    return {
        'post': post.relative_to(site_dir).as_posix(), 'title': fields['title'],
        'before_markdown_sha256': sha256(before), 'after_markdown_sha256': sha256(post.read_bytes()),
        'body_sha256': sha256(body.encode()), 'after_body_sha256': sha256(updated_body.encode()),
        'restored_dashboard_screenshot': is_ad_project and body != expected_body,
        'old_image': fields['img'], 'image': image,
        'image_sha256': image_hash, 'model_chain': image_model,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected-count', required=True, type=int)
    parser.add_argument('--include-ad-project', action='store_true', help='Also create an AD portfolio cover and restore its dashboard screenshot inline')
    parser.add_argument('--image-model', default=os.getenv('VERTEX_BLOG_IMAGE_MODEL') or DEFAULT_MODELS)
    args = parser.parse_args(argv)
    site = args.site_dir.resolve()
    output = args.output_dir.resolve()
    if not 1 <= args.expected_count <= 12:
        raise ValueError('Explicit expected count must be between 1 and 12')
    if output.exists():
        raise FileExistsError('Use a fresh output directory for each repair run')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=site, text=True).strip():
        raise ValueError('Refusing to stage repairs in a dirty website checkout')
    posts = legacy_posts(site)
    if len(posts) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} legacy covers, found {len(posts)}')
    if args.include_ad_project:
        portfolio = site / AD_PROJECT
        if not portfolio.is_file():
            raise FileNotFoundError('The authorized AD portfolio project is missing')
        posts.append(portfolio)
    output.mkdir(parents=True)
    report = {
        'status': 'in_progress', 'expected_count': len(posts), 'legacy_cover_count': args.expected_count,
        'site_base_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=site, text=True).strip(),
        'publisher_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'posts': [],
    }
    manifest = output / 'manifest.json'
    for post in posts:
        report['posts'].append(stage_one(site, post, output, args.image_model))
        manifest.write_text(json.dumps(report, indent=2) + '\n')
        print(f'Staged story-specific image: {post.name}', flush=True)
    hashes = [item['image_sha256'] for item in report['posts']]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate images found in repair batch')
    report['status'] = 'staged_not_published'
    manifest.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Staged {len(posts)} covers; no commit, push, publication, email, or article rewrite performed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

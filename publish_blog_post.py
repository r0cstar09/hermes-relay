#!/usr/bin/env python3
"""Publish the top Hermes Relay briefing item as a blog post.

The script consumes the JSON produced by llm_score_and_summarize.py and the
matching hermes_signal_YYYY-MM-DD.json source article list, writes one Astro
content Markdown file into opposite-osiris, optionally verifies the Astro build,
and optionally commits/pushes the result.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


DEFAULT_SITE_DIR = Path(os.getenv("OPPOSITE_OSIRIS_DIR", "/mnt/c/Users/antho/opposite-osiris"))
DEFAULT_BLOG_DIR = Path("src/content/blog")


@dataclass
class ArticleBlock:
    title: str
    score: int | None
    key_takeaways: list[str]
    angle: str | None
    board_take: str | None
    summary: str
    variant_a: str | None
    variant_b: str | None
    source_url: str | None


def slugify(value: str, max_len: int = 72) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return (value[:max_len].rstrip("-") or "hermes-relay-briefing")


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_briefing_json(base_dir: Path) -> Path:
    files = sorted(base_dir.glob("json_output/*/hermes_llm_top3_*.json"), reverse=True)
    if not files:
        raise FileNotFoundError("No json_output/*/hermes_llm_top3_*.json files found")
    return files[0]


def date_from_briefing_path(path: Path, data: dict) -> str:
    if data.get("date"):
        return str(data["date"])
    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(path))
    return match.group(1) if match else date.today().isoformat()


def split_article_blocks(text: str) -> list[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*---\s*\n", text) if b.strip()]
    return blocks or [text.strip()]


def extract_section(block: str, label: str, stop_labels: Iterable[str]) -> str | None:
    labels = [re.escape(label)] + [re.escape(s) for s in stop_labels]
    pattern = rf"(?is){re.escape(label)}\s*:?\s*(.*?)(?=\n\s*(?:{'|'.join(labels[1:])})\s*:?|\Z)"
    match = re.search(pattern, block)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def extract_title(block: str) -> str:
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\d+\)\s*(.+?)\s*$", line)
        if match:
            return match.group(1).strip(" -*#")
        return line.strip(" -*#")
    return "Hermes Relay Cyber Briefing"


def extract_score(block: str) -> int | None:
    match = re.search(r"(?im)^Score:\s*(\d{1,2})\s*/\s*10", block)
    if not match:
        return None
    return int(match.group(1))


def extract_bullets(text: str | None) -> list[str]:
    if not text:
        return []
    bullets = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(('-', '*')):
            bullets.append(line.lstrip('-* ').strip())
        elif line:
            bullets.append(line)
    return bullets


def normalize_title(value: str) -> str:
    return re.sub(r"\W+", " ", value).strip().lower()


def find_source_url(title: str, signal_articles: list[dict]) -> str | None:
    norm = normalize_title(title)
    for article in signal_articles:
        if normalize_title(str(article.get("title", ""))) == norm:
            return article.get("link") or article.get("url")
    for article in signal_articles:
        article_title = normalize_title(str(article.get("title", "")))
        if article_title and (article_title in norm or norm in article_title):
            return article.get("link") or article.get("url")
    return None


def parse_articles(llm_text: str, signal_articles: list[dict]) -> list[ArticleBlock]:
    stop_labels = [
        "Score",
        "Key Takeaways",
        "Angle for this story",
        "One-Line Board Take",
        "Board-Level Impact",
        "Article Summary",
        "Briefing - Variant A",
        "Briefing - Variant B",
        "LinkedIn Post",
    ]
    parsed: list[ArticleBlock] = []
    for raw in split_article_blocks(llm_text):
        title = extract_title(raw)
        key_takeaways = extract_bullets(extract_section(raw, "Key Takeaways", stop_labels))
        summary = (
            extract_section(raw, "Article Summary", stop_labels)
            or extract_section(raw, "Board-Level Impact", stop_labels)
            or ""
        )
        variant_a = extract_section(raw, "Briefing - Variant A", stop_labels)
        variant_b = extract_section(raw, "Briefing - Variant B", stop_labels)
        if not summary:
            # Fallback for older briefing formats.
            summary = re.sub(r"(?im)^Score:.*$", "", raw).strip()
        parsed.append(
            ArticleBlock(
                title=title,
                score=extract_score(raw),
                key_takeaways=key_takeaways,
                angle=extract_section(raw, "Angle for this story", stop_labels),
                board_take=extract_section(raw, "One-Line Board Take", stop_labels),
                summary=summary.strip(),
                variant_a=variant_a.strip() if variant_a else None,
                variant_b=variant_b.strip() if variant_b else None,
                source_url=find_source_url(title, signal_articles),
            )
        )
    return parsed


def choose_top_article(blocks: list[ArticleBlock]) -> ArticleBlock:
    if not blocks:
        raise ValueError("No article blocks found in briefing JSON")
    return sorted(blocks, key=lambda b: b.score if b.score is not None else -1, reverse=True)[0]


def build_markdown(article: ArticleBlock, run_date: str, lens: str | None, model: str | None) -> str:
    description = article.board_take or (article.key_takeaways[0] if article.key_takeaways else article.summary[:150])
    tags = ["cyber", "threat-intelligence", "defense"]
    body: list[str] = []
    body.append("---")
    body.append(f"title: {yaml_quote(article.title)}")
    body.append(f"description: {yaml_quote(description[:180])}")
    body.append(f"publishDate: {yaml_quote(run_date)}")
    body.append(f"tags: [{', '.join(yaml_quote(t) for t in tags)}]")
    body.append("---")
    body.append("")
    if article.board_take:
        body.append(f"> {article.board_take}")
        body.append("")
    body.append("Hermes Relay surfaced this as the highest-priority cyber story in today's feed. The useful question is not just what happened, but what a defender should do with the signal.")
    body.append("")
    body.append("## What happened")
    body.append("")
    body.append(article.summary.strip())
    body.append("")
    if article.key_takeaways:
        body.append("## Why it matters")
        body.append("")
        for takeaway in article.key_takeaways[:4]:
            body.append(f"- {takeaway}")
        body.append("")
    commentary = article.variant_a or article.variant_b
    if commentary:
        body.append("## Practitioner take")
        body.append("")
        body.append(commentary.strip())
        body.append("")
    body.append("## What I would watch next")
    body.append("")
    body.append("The next useful signal is whether this turns into repeatable attacker tradecraft, a one-off disclosure, or a control-validation problem for teams that assumed they already had coverage. I would use it as a prompt to verify exposure, confirm logging, and make sure the response path is owned before the story fades from the feed.")
    body.append("")
    body.append("---")
    body.append("")
    body.append("*Generated from Hermes Relay's daily cyber briefing and reviewed through a practitioner lens before publishing to this blog.*")
    if article.source_url:
        body.append(f"\nSource: [{article.title}]({article.source_url})")
    if lens or model:
        details = []
        if lens:
            details.append(f"lens: {lens}")
        if model:
            details.append(f"model: {model}")
        body.append(f"\nPipeline note: {'; '.join(details)}.")
    body.append("")
    return "\n".join(body)


def run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}  # cwd={cwd}")
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc


def maybe_commit_and_push(site_dir: Path, post_path: Path, message: str, push: bool) -> None:
    rel = post_path.relative_to(site_dir)
    run(["git", "add", str(rel)], cwd=site_dir)
    diff = run(["git", "diff", "--cached", "--quiet"], cwd=site_dir, check=False)
    if diff.returncode == 0:
        print("No blog changes to commit.")
        return
    run(["git", "commit", "-m", message], cwd=site_dir)
    if push:
        run(["git", "push", "origin", "main"], cwd=site_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--briefing-json", type=Path, help="Path to hermes_llm_top3_YYYY-MM-DD.json")
    parser.add_argument("--signal-json", type=Path, help="Path to matching hermes_signal_YYYY-MM-DD.json")
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE_DIR)
    parser.add_argument("--blog-dir", type=Path, default=DEFAULT_BLOG_DIR)
    parser.add_argument("--verify-build", action="store_true", help="Run npm run build in the Astro site")
    parser.add_argument("--commit", action="store_true", help="Commit the generated post in the Astro site repo")
    parser.add_argument("--push", action="store_true", help="Push the commit to origin main; implies --commit")
    args = parser.parse_args(argv)

    repo_dir = Path.cwd()
    briefing_json = args.briefing_json or find_latest_briefing_json(repo_dir)
    briefing_data = load_json(briefing_json)
    run_date = date_from_briefing_path(briefing_json, briefing_data)
    signal_json = args.signal_json or repo_dir / f"hermes_signal_{run_date}.json"
    signal_articles = load_json(signal_json) if signal_json.exists() else []

    llm_text = briefing_data.get("top_articles")
    if not isinstance(llm_text, str) or not llm_text.strip():
        raise ValueError(f"{briefing_json} does not contain non-empty top_articles text")

    selected = choose_top_article(parse_articles(llm_text, signal_articles))
    slug = f"{run_date}-{slugify(selected.title)}.md"
    site_dir = args.site_dir.resolve()
    blog_dir = site_dir / args.blog_dir
    blog_dir.mkdir(parents=True, exist_ok=True)
    post_path = blog_dir / slug
    markdown = build_markdown(
        selected,
        run_date=run_date,
        lens=briefing_data.get("lens"),
        model=os.getenv("VERTEX_MODEL_RESOURCE") or os.getenv("VERTEX_MODEL"),
    )
    post_path.write_text(markdown, encoding="utf-8")
    print(f"Published blog markdown -> {post_path}")

    if args.verify_build:
        run(["npm", "run", "build"], cwd=site_dir)

    if args.push:
        args.commit = True
    if args.commit:
        maybe_commit_and_push(
            site_dir,
            post_path,
            message=f"Publish Hermes Relay blog post for {run_date}",
            push=args.push,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

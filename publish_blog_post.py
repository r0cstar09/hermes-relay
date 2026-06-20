#!/usr/bin/env python3
"""Publish the top Hermes Relay briefing item as a blog post.

The script consumes the JSON produced by llm_score_and_summarize.py and the
matching hermes_signal_YYYY-MM-DD.json source article list, writes one Astro
content Markdown file into opposite-osiris, optionally verifies the Astro build,
and optionally commits/pushes the result.
"""

from __future__ import annotations

import argparse
import base64
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
DEFAULT_VOICE_PROFILE = Path("prompts/tony_voice.md")
DEFAULT_VERTEX_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1") or "us-central1"
DEFAULT_IMAGE_MODEL = os.getenv("VERTEX_BLOG_IMAGE_MODEL")


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


def build_markdown(
    article: ArticleBlock,
    run_date: str,
    lens: str | None,
    model: str | None,
    hero_image: str | None = None,
    hero_image_alt: str | None = None,
) -> str:
    description = article.board_take or (article.key_takeaways[0] if article.key_takeaways else article.summary[:150])
    tags = ["cyber", "threat-intelligence", "defense"]
    body: list[str] = []
    body.append("---")
    body.append(f"title: {yaml_quote(article.title)}")
    body.append(f"description: {yaml_quote(description[:180])}")
    body.append(f"publishDate: {yaml_quote(run_date)}")
    body.append(f"tags: [{', '.join(yaml_quote(t) for t in tags)}]")
    if hero_image:
        body.append(f"img: {yaml_quote(hero_image)}")
        body.append(f"img_alt: {yaml_quote(hero_image_alt or f'Abstract cyber defense illustration for {article.title}')}")
    body.append("---")
    body.append("")
    if article.board_take:
        body.append(f"> {article.board_take}")
        body.append("")
    body.append("The headline is the easy part. The useful question is what this story exposes about how security programs actually work under pressure.")
    body.append("")
    body.append("## What happened")
    body.append("")
    body.append(article.summary.strip())
    body.append("")
    if article.key_takeaways:
        body.append("## What people will get wrong")
        body.append("")
        body.append("The common mistake is treating this as a standalone news item instead of a test of ownership, visibility, and response discipline.")
        body.append("")
        for takeaway in article.key_takeaways[:4]:
            body.append(f"- {takeaway}")
        body.append("")
    commentary = article.variant_a or article.variant_b
    if commentary:
        body.append("## Practitioner lens")
        body.append("")
        body.append(commentary.strip())
        body.append("")
    body.append("## What I would watch next")
    body.append("")
    body.append("The next useful signal is whether this becomes repeatable attacker tradecraft, a one-off disclosure, or a control-validation problem for teams that assumed they already had coverage. I would use it as a prompt to verify exposure, confirm logging, and make sure the response path is owned before the story fades from the feed.")
    body.append("")
    body.append("---")
    body.append("")
    body.append("*Generated from Hermes Relay's daily cyber briefing and edited through Tony's practitioner voice profile before publishing to this blog.*")
    if article.source_url:
        body.append(f"\nSource: [{article.title}]({article.source_url})")
    if lens or model:
        details = []
        if lens:
            details.append(f"lens: {lens}")
        if model:
            details.append(f"draft model: {model}")
        body.append(f"\nPipeline note: {'; '.join(details)}.")
    body.append("")
    return "\n".join(body)


def build_hero_image_prompt(article: ArticleBlock) -> str:
    context = article.board_take or article.angle or article.summary[:220]
    return f"""Create a 16:9 hero image for a cybersecurity practitioner's blog post.

Article title: {article.title}
Editorial angle: {context}

Visual direction:
- abstract cyber defense operations, not a literal news screenshot
- network edge, telemetry streams, detection logic, incident-response pressure
- dark navy/charcoal background with restrained cyan/teal/amber accents
- premium editorial illustration, clean composition, high contrast
- no humans, no faces, no company logos, no vendor names, no UI screenshots
- no readable text, no fake CVE strings, no lock/key clichés
- suitable for a professional portfolio/blog hero image
"""


def _image_bytes_from_generated_image(generated) -> bytes | None:
    image = getattr(generated, "image", None)
    if not image:
        return None
    raw = getattr(image, "image_bytes", None)
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, str):
        return base64.b64decode(raw)
    return None


def generate_hero_image_with_vertex(*, article: ArticleBlock, image_model: str, output_path: Path) -> str:
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = DEFAULT_VERTEX_LOCATION
    if not project:
        raise EnvironmentError("GOOGLE_CLOUD_PROJECT is required for the Vertex blog hero image pass")

    from google import genai
    from google.genai import types

    prompt = build_hero_image_prompt(article)
    client = genai.Client(vertexai=True, project=project, location=location)
    print(f"Running Vertex blog hero image pass with model: {image_model}")
    response = client.models.generate_images(
        model=image_model,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="16:9",
            output_mime_type="image/png",
            add_watermark=False,
            enhance_prompt=True,
        ),
    )
    generated_images = getattr(response, "generated_images", None) or []
    for generated in generated_images:
        data = _image_bytes_from_generated_image(generated)
        if data:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(data)
            return output_path.as_posix()
    reasons = [getattr(item, "rai_filtered_reason", None) for item in generated_images]
    reasons = [reason for reason in reasons if reason]
    suffix = f": {'; '.join(reasons)}" if reasons else ""
    raise ValueError(f"Vertex image generation did not return image bytes{suffix}")


def split_frontmatter(markdown: str) -> tuple[str, str]:
    if not markdown.startswith("---\n"):
        return "", markdown
    end = markdown.find("\n---\n", 4)
    if end == -1:
        return "", markdown
    return markdown[: end + len("\n---\n")], markdown[end + len("\n---\n") :]


def extract_response_text(response) -> str:
    if getattr(response, "text", None):
        return response.text.strip()
    parts: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def parse_model_fallbacks(value: str) -> list[str]:
    return [model.strip() for model in value.split(",") if model.strip()]


def build_editor_prompt(*, body: str, voice_profile: str, article: ArticleBlock) -> str:
    source_line = f"Source URL that must remain present if already in the draft: {article.source_url}" if article.source_url else "No source URL was matched for this article."
    return f"""You are editing a cybersecurity blog post for Tony.

Use the voice profile below as the target. Do not copy the profile phrases mechanically; use it to match directness, skepticism, rhythm, and word choice.

{voice_profile}

Editing task:
- Rewrite the draft body so it sounds like Tony, not an AI-generated cyber summary.
- Make one clear practitioner argument.
- Keep the factual recap short.
- Preserve Markdown headings unless changing a heading makes the argument clearer.
- Preserve source links, the disclosure note, and pipeline note if present.
- Do not add facts, CVEs, vendors, dates, numbers, breach details, exploit mechanics, quotes, or source claims that are not already in the draft.
- Keep it publishable as a blog post, not LinkedIn.
- Return Markdown body only. Do not return YAML frontmatter. Do not wrap in code fences.

{source_line}

Draft body:

{body}
"""


def edit_markdown_with_vertex(markdown: str, *, editor_model: str, voice_profile_path: Path, article: ArticleBlock) -> str:
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = DEFAULT_VERTEX_LOCATION
    if not project:
        raise EnvironmentError("GOOGLE_CLOUD_PROJECT is required for the Vertex blog editor pass")
    if not voice_profile_path.exists():
        raise FileNotFoundError(f"Voice profile not found: {voice_profile_path}")

    from google import genai
    from google.genai import types

    frontmatter, body = split_frontmatter(markdown)
    voice_profile = voice_profile_path.read_text(encoding="utf-8")
    prompt = build_editor_prompt(body=body, voice_profile=voice_profile, article=article)
    client = genai.Client(vertexai=True, project=project, location=location)
    response = None
    last_error: Exception | None = None
    for model in parse_model_fallbacks(editor_model):
        try:
            print(f"Running Vertex blog editor pass with model: {model}")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.6,
                    system_instruction=(
                        "You are a careful human editor for a cybersecurity practitioner's blog. "
                        "Preserve facts and links. Remove generic AI prose. Return only Markdown body."
                    ),
                ),
            )
            break
        except Exception as exc:
            last_error = exc
            message = str(exc)
            if "404" in message or "NOT_FOUND" in message or "not found" in message.lower():
                print(f"Vertex editor model unavailable: {model}; trying next fallback if configured.")
                continue
            raise
    if response is None:
        raise RuntimeError(f"No configured Vertex editor model succeeded: {last_error}")
    edited_body = extract_response_text(response)
    if not edited_body:
        raise ValueError("Vertex blog editor response did not contain text output")
    edited_body = re.sub(r"^```(?:markdown|md)?\s*", "", edited_body.strip(), flags=re.I)
    edited_body = re.sub(r"\s*```$", "", edited_body.strip())
    if article.source_url and article.source_url not in edited_body and article.source_url in body:
        print("Editor removed source URL; restoring source/footer from draft body.")
        for line in body.splitlines():
            if article.source_url in line:
                edited_body = edited_body.rstrip() + "\n\n" + line.strip() + "\n"
                break
    return frontmatter + edited_body.strip() + "\n"


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


def maybe_commit_and_push(site_dir: Path, paths: Path | list[Path], message: str, push: bool) -> None:
    if isinstance(paths, Path):
        paths = [paths]
    rel_paths = [str(path.relative_to(site_dir)) for path in paths]
    run(["git", "add", *rel_paths], cwd=site_dir)
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
    parser.add_argument("--editor-model", default=os.getenv("VERTEX_BLOG_EDITOR_MODEL"), help="Optional Vertex model for final Tony-voice editor pass")
    parser.add_argument("--voice-profile", type=Path, default=DEFAULT_VOICE_PROFILE, help="Markdown voice profile for the editor pass")
    parser.add_argument("--skip-editor", action="store_true", help="Disable the Vertex final editor pass even if --editor-model/env is set")
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL, help="Optional Vertex image model for blog hero image generation")
    parser.add_argument("--image-dir", type=Path, default=Path("public/assets/blog/hermes-relay"), help="Site-relative public directory for generated hero images")
    parser.add_argument("--skip-image", action="store_true", help="Disable the Vertex hero image pass even if --image-model/env is set")
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
    slug_stem = slug[:-3]
    site_dir = args.site_dir.resolve()
    blog_dir = site_dir / args.blog_dir
    blog_dir.mkdir(parents=True, exist_ok=True)
    post_path = blog_dir / slug
    hero_image = None
    hero_image_alt = None
    hero_image_path = None
    if args.image_model and not args.skip_image:
        hero_image_path = site_dir / args.image_dir / f"{slug_stem}.png"
        generate_hero_image_with_vertex(article=selected, image_model=args.image_model, output_path=hero_image_path)
        public_dir = site_dir / "public"
        hero_image = "/" + hero_image_path.relative_to(public_dir).as_posix()
        hero_image_alt = f"Abstract cyber defense illustration for {selected.title}"
    markdown = build_markdown(
        selected,
        run_date=run_date,
        lens=briefing_data.get("lens"),
        model=os.getenv("VERTEX_MODEL_RESOURCE") or os.getenv("VERTEX_MODEL"),
        hero_image=hero_image,
        hero_image_alt=hero_image_alt,
    )
    if args.editor_model and not args.skip_editor:
        markdown = edit_markdown_with_vertex(
            markdown,
            editor_model=args.editor_model,
            voice_profile_path=args.voice_profile,
            article=selected,
        )
    post_path.write_text(markdown, encoding="utf-8")
    print(f"Published blog markdown -> {post_path}")

    if args.verify_build:
        run(["npm", "run", "build"], cwd=site_dir)

    if args.push:
        args.commit = True
    if args.commit:
        commit_paths = [post_path]
        if hero_image_path:
            commit_paths.append(hero_image_path)
        maybe_commit_and_push(
            site_dir,
            commit_paths,
            message=f"Publish Hermes Relay blog post for {run_date}",
            push=args.push,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

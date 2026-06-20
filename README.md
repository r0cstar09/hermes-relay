# Hermes Relay

Hermes Relay curates high-impact cybersecurity news and produces a daily executive-ready briefing.

## What It Does
- Pulls fresh stories from selected cybersecurity RSS feeds
- Persists seen articles and briefing metadata in SQLite
- Deduplicates against the persistent article history
- Uses Google Vertex AI Gemini to score and summarize the top stories
- Writes JSON + HTML briefing outputs and optionally emails the result
- Publishes the highest-priority story as a blog post in `opposite-osiris` when a cross-repo GitHub token is configured

## Runtime Requirements
- Python 3.10+
- Google Cloud CLI (`gcloud`)
- Vertex AI API enabled in your Google Cloud project
- Application Default Credentials (ADC)

## Environment Variables
Required:
- `GOOGLE_CLOUD_PROJECT` - Google Cloud project ID used by Vertex AI

Optional:
- `GOOGLE_CLOUD_LOCATION` - Vertex region (default: `us-central1`)
- `VERTEX_MODEL` - draft/ranking model ID (default: `gemini-2.5-flash`)
- `VERTEX_MODEL_RESOURCE` - full Vertex model resource name override (if set, this takes precedence over `VERTEX_MODEL`)
- `VERTEX_BLOG_EDITOR_MODEL` - stronger Vertex model used for final Tony-voice blog editing (GitHub Actions default: `gemini-3.5-flash`)
- `ICLOUD_EMAIL`, `ICLOUD_PASSWORD`, `EMAIL_RECIPIENT` - only required for SMTP email delivery
- `OPPOSITE_OSIRIS_DIR` - local path to the Astro site when running `publish_blog_post.py` manually (default: `/mnt/c/Users/antho/opposite-osiris`)
- GitHub secret `OPPOSITE_OSIRIS_PAT` - fine-grained token with contents read/write on `r0cstar09/opposite-osiris`; required for scheduled cross-repo blog publishing

## ADC Setup
Run the bootstrap script:

```bash
bash <(curl -sSL https://storage.googleapis.com/cloud-samples-data/adc/setup_adc.sh)
```

If you need to complete setup manually:

```bash
gcloud auth application-default login
gcloud auth login --update-adc
gcloud config set project <your-project-id>
gcloud auth application-default set-quota-project <your-project-id>
gcloud services enable aiplatform.googleapis.com --project <your-project-id>
```

## Install and Run
```bash
python -m pip install -r requirements.txt
python orchestrator.py
```

## Pipeline Steps
1. `hermes-relay.py` fetches RSS articles, stores them in SQLite, and writes only first-seen articles for today's run
2. `llm_score_and_summarize.py` builds a prompt and calls Vertex Gemini
3. The script saves JSON/HTML output, records briefing metadata in SQLite, and sends email if SMTP vars are configured
4. `publish_blog_post.py` selects the top-scored story, writes a published Astro Markdown post to `opposite-osiris/src/content/blog/`, runs a stronger Vertex editor pass against `prompts/tony_voice.md`, verifies the Astro build, commits, and pushes to `main` when `OPPOSITE_OSIRIS_PAT` is configured in GitHub Actions

## Blog Publishing
Manual local publish after a briefing exists:

```bash
python publish_blog_post.py \
  --site-dir /mnt/c/Users/antho/opposite-osiris \
  --editor-model "${VERTEX_BLOG_EDITOR_MODEL:-gemini-3.5-flash}" \
  --voice-profile prompts/tony_voice.md \
  --verify-build \
  --commit \
  --push
```

Use `--skip-editor` for local dry runs without Vertex credentials.

The generated post is published, not drafted. LinkedIn is not touched.

## Persistence
Hermes Relay uses SQLite for durable history:

- default path: `hermes_relay.db`
- override path: `HERMES_RELAY_DB=/path/to/hermes_relay.db`
- tables: `articles`, `briefings`, `metadata`

On GitHub Actions, `hermes_relay.db*` is restored/saved with `actions/cache`, then uploaded as a workflow artifact with the JSON/HTML outputs. Locally, the DB file stays in the repo working directory but is ignored by git.

This fixes the old artifact-only issue: daily runners can now remember articles that were already seen and avoid repeatedly drafting around the same stories.

## Tests
```bash
python -m unittest discover -v
```

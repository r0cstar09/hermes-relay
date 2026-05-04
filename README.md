# Hermes Relay

Hermes Relay curates high-impact cybersecurity news and produces a daily executive-ready briefing.

## What It Does
- Pulls fresh stories from selected cybersecurity RSS feeds
- Deduplicates against previously seen articles
- Uses Google Vertex AI Gemini to score and summarize the top stories
- Writes JSON + HTML briefing outputs and optionally emails the result

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
- `VERTEX_MODEL` - model ID (default: `gemini-2.5-flash`)
- `VERTEX_MODEL_RESOURCE` - full Vertex model resource name override (if set, this takes precedence over `VERTEX_MODEL`)
- `ICLOUD_EMAIL`, `ICLOUD_PASSWORD`, `EMAIL_RECIPIENT` - only required for SMTP email delivery

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
1. `hermes-relay.py` fetches and stores today's new articles
2. `llm_score_and_summarize.py` builds a prompt and calls Vertex Gemini
3. The script saves JSON/HTML output and sends email if SMTP vars are configured
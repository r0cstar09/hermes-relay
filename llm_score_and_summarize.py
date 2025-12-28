import os
import json
import glob
import requests
from datetime import date

# -----------------------------
# CONFIG
# -----------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")   # change if needed

OUTPUT_FILE = f"hermes_llm_top3_{date.today().isoformat()}.json"

# -----------------------------
# VALIDATION
# -----------------------------
if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise EnvironmentError(
        "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY environment variables"
    )

# -----------------------------
# HELPERS
# -----------------------------
def load_articles():
    files = sorted(glob.glob("hermes_signal_*.json"), reverse=True)

    if not files:
        raise FileNotFoundError("No hermes_relay_*.json files found")

    latest_file = files[0]
    print(f"Loading articles from: {latest_file}")

    with open(latest_file, "r") as f:
        return json.load(f)


def build_prompt(articles):
    return f"""
You are a senior cybersecurity analyst advising executives.

Task:
- Score each article from 1–10 based on business risk, reputational damage, and executive concern.
- Select the top 3.
- For each, provide:
  • Headline
  • Score
  • 3–4 blunt, plain-language bullets
  • One sentence on why executives should care

Articles:
{json.dumps(articles, indent=2)}
"""


def call_llm(prompt):
    base = AZURE_OPENAI_ENDPOINT.rstrip("/")
    url = f"{base}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are concise, skeptical, and executive-focused."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }

    response = requests.post(url, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(f"LLM request failed: {response.status_code} {response.text}")
        # Write a helpful error file for debugging
        with open(OUTPUT_FILE, "w") as f:
            json.dump({"error": "LLM request failed", "status": response.status_code, "response": response.text}, f, indent=2)
        import sys
        sys.exit(1)

    return response.json()["choices"][0]["message"]["content"]


# -----------------------------
# MAIN
# -----------------------------
def main():
    articles = load_articles()
    prompt = build_prompt(articles)

    print("Calling Azure OpenAI…")
    result = call_llm(prompt)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(
            {
                "date": date.today().isoformat(),
                "top_articles": result,
            },
            f,
            indent=2,
        )

    print(f"Saved output → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
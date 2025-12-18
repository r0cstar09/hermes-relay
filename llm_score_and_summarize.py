import os
import json
import requests
from datetime import date

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = "gpt-5.2-chat"  # change to your deployed model name

INPUT_FILE = f"hermes_relay_{date.today()}.json"
OUTPUT_FILE = f"hermes_llm_top3_{date.today()}.json"

headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_API_KEY
}

def load_articles():
    with open(INPUT_FILE, "r") as f:
        return json.load(f)

def build_prompt(articles):
    return f"""
You are a senior cybersecurity analyst advising executives.

From the list below:
1. Score each article (1–10) based on business risk, reputational damage, and likelihood of executive concern.
2. Select the top 3.
3. Summarize each in 3–4 blunt, plain-language bullets.
4. Explain in one sentence why executives should care.

Articles:
{json.dumps(articles, indent=2)}
"""

def call_llm(prompt):
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"

    payload = {
        "messages": [
            {"role": "system", "content": "You are concise, skeptical, and executive-focused."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 1
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def main():
    articles = load_articles()
    prompt = build_prompt(articles)
    result = call_llm(prompt)

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"top_articles": result}, f, indent=2)

    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
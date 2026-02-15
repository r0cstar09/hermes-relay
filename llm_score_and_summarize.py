import os
import json
import glob
import requests
import smtplib
import re
from datetime import date
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# -----------------------------
# CONFIG
# -----------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Deployment name - change this to match your Azure OpenAI deployment name
# If not set in environment, defaults to gpt-4o-mini
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
# Ensure deployment name is not empty (handle case where secret exists but is empty)
if not DEPLOYMENT_NAME or DEPLOYMENT_NAME.strip() == "":
    DEPLOYMENT_NAME = "gpt-4o-mini"
    print(f"Warning: AZURE_OPENAI_DEPLOYMENT was empty, using default: {DEPLOYMENT_NAME}")

# API version - required for Azure OpenAI
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
# Ensure API version is not empty (handle case where secret exists but is empty)
if not API_VERSION or API_VERSION.strip() == "":
    API_VERSION = "2024-02-15-preview"
    print(f"Warning: AZURE_OPENAI_API_VERSION was empty, using default: {API_VERSION}")

# Email configuration
ICLOUD_EMAIL = os.getenv("ICLOUD_EMAIL")
ICLOUD_PASSWORD = os.getenv("ICLOUD_PASSWORD")  # App-specific password
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", ICLOUD_EMAIL)  # Default to sender if not set

# Output directory structure: json_output/YYYY-MM-DD/hermes_llm_top3_YYYY-MM-DD.json
today = date.today().isoformat()
OUTPUT_DIR = Path("json_output") / today
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
OUTPUT_FILE = OUTPUT_DIR / f"hermes_llm_top3_{today}.json"

# -----------------------------
# VALIDATION
# -----------------------------
if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise EnvironmentError(
        "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY environment variables"
    )

# Final validation - ensure values are set
if not DEPLOYMENT_NAME or DEPLOYMENT_NAME.strip() == "":
    raise EnvironmentError(
        f"DEPLOYMENT_NAME is empty. Please set AZURE_OPENAI_DEPLOYMENT environment variable."
    )
if not API_VERSION or API_VERSION.strip() == "":
    raise EnvironmentError(
        f"API_VERSION is empty. Please set AZURE_OPENAI_API_VERSION environment variable."
    )

print(f"=== Configuration Loaded ===")
print(f"  Deployment: '{DEPLOYMENT_NAME}'")
print(f"  API Version: '{API_VERSION}'")
print(f"  Endpoint: {AZURE_OPENAI_ENDPOINT[:50]}..." if AZURE_OPENAI_ENDPOINT else "  Endpoint: NOT SET")
print(f"============================")

# -----------------------------
# HELPERS
# -----------------------------
def load_articles():
    """Load articles from today's file (new articles only)."""
    # Look for today's specific file first
    today_file = f"hermes_signal_{today}.json"
    
    if not Path(today_file).exists():
        # Fallback: get the latest file if today's doesn't exist
        files = sorted(glob.glob("hermes_signal_*.json"), reverse=True)
        if not files:
            raise FileNotFoundError(f"No hermes_signal_*.json files found. Expected today's file: {today_file}")
        latest_file = files[0]
        print(f"Warning: Today's file ({today_file}) not found. Using latest file: {latest_file}")
        file_to_load = latest_file
    else:
        file_to_load = today_file
        print(f"Loading new articles from today's file: {file_to_load}")
    
    with open(file_to_load, "r", encoding="utf-8") as f:
        articles = json.load(f)
    
    if not articles:
        raise ValueError(f"No articles found in {file_to_load}. The file may be empty.")
    
    print(f"Loaded {len(articles)} new article(s) from today")
    return articles


# Lens of the day: applied to Article Summary and Briefing. Rotates by day of year.
LENSES = [
    {
        "name": "First 24 hours",
        "description": "Focus on what the board and security teams should know or do in the first 24 hours after this kind of incident or disclosure.",
    },
    {
        "name": "Supply chain and third-party risk",
        "description": "Focus on supply chain, vendor, or third-party implications—how this affects or depends on partners, suppliers, or software supply chain.",
    },
    {
        "name": "Explain to non-security leadership",
        "description": "Frame everything so a CFO, COO, or general counsel can understand why it matters and what the organization should do.",
    },
    {
        "name": "One concrete detection or mitigation step",
        "description": "Emphasize at least one specific, actionable detection or mitigation step a team can take (e.g. a control, a query, a config change).",
    },
    {
        "name": "Trend and threat landscape",
        "description": "Place this incident or vulnerability in the broader trend and threat landscape—what’s shifting and what it signals.",
    },
    {
        "name": "Regulatory and compliance",
        "description": "Focus on regulatory, legal, or compliance implications and what boards or security leaders should consider from that angle.",
    },
    {
        "name": "What I would do next",
        "description": "Write from experience: what you would actually do next as a practitioner or leader—priorities, order of operations, and trade-offs.",
    },
    {
        "name": "The number that matters",
        "description": "Lead with or center the one number that matters (e.g. impact, scope, cost, timeline) and build the narrative around it.",
    },
]


def get_lens_for_date(d: date) -> tuple[str, str]:
    """Return (lens_name, lens_description) for the given date. Deterministic by day of year."""
    idx = (d.toordinal() % len(LENSES))
    lens = LENSES[idx]
    return lens["name"], lens["description"]


# Angles the LLM can choose from per article (judgment call: which aspect matters most for this story).
# Model picks ONE per article and writes everything through that angle. Encouraged to vary across the 3 articles.
ARTICLE_ANGLES = [
    "First 24 hours — what to know or do immediately",
    "Supply chain or third-party risk",
    "Explain to non-security leadership (CFO, board, legal)",
    "One concrete detection or mitigation step",
    "Trend and threat landscape — what this signals",
    "Regulatory or compliance implications",
    "What I would do next (practitioner priorities)",
    "The number that matters (impact, scope, cost, timeline)",
    "Executive or privileged access risk",
    "Active exploitation vs. theoretical risk",
]


def build_prompt(articles, lens_name: str, lens_description: str):
    angles_list = "\n".join(f'- "{a}"' for a in ARTICLE_ANGLES)
    return f"""
You are a senior cybersecurity analyst advising executives.

Today's lens (optional nudge): "{lens_name}". {lens_description}

Task:
- You are analyzing NEW articles from today ({today}). Score each from 1–10, select the top 3.
- For each article you MUST make a judgment call: which aspect of this story is most important? Pick ONE angle from the list below and write the entire article block through that angle.
- Prefer different angles for each of the 3 articles when it makes sense (so the briefing has variety). You may use today's lens or any angle from the list.

For each article, use this EXACT format (use "---" to separate each article):

1) [Headline - use exact title from article]
Score: [X]/10

Key Takeaways:
- [2-3 short, critical bullet points only]

Angle for this story:
[Pick exactly ONE from this list and write it on the next line. This is your judgment of what matters most for this story. Then write everything below through this angle.]
{angles_list}

One-Line Board Take:
[One line, under 15 words, through the angle you chose. Board-level so-what.]

Article Summary:
[2-3 short paragraphs that summarize what the article is actually about. Give you and your readers a clear, factual overview: what happened, who or what is affected, and the main points. Plain language, no board jargon. This is the "what this article says" summary.]

Briefing - Variant A:
[ONE paragraph, 90–140 words. LinkedIn-ready. Through your chosen angle. Lead with the so-what. No bullets. Natural tone. End with a forward-looking insight, not a question.]

Briefing - Variant B:
[ONE paragraph, 90–140 words. LinkedIn-ready. Through your chosen angle. Lead with one concrete detail (number, technique, or fact), then expand. Different opening from A. End with a forward-looking insight, not a question.]

---

[Repeat for article 2 — choose an angle, ideally different from article 1 when appropriate]

---

[Repeat for article 3 — choose an angle, ideally different from articles 1 and 2 when appropriate]

IMPORTANT:
- Use the exact article title/headline as in the articles list below.
- For each article output "Angle for this story:" then the exact angle text from the list. Then write One-Line Board Take, Article Summary, Briefing - Variant A, and Briefing - Variant B through that angle.
- Vary the chosen angle across the three articles when it fits the stories.

Articles:
{json.dumps(articles, indent=2)}
"""


def call_llm(prompt):
    # Validate environment variables
    if not AZURE_OPENAI_ENDPOINT:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
    if not AZURE_OPENAI_API_KEY:
        raise ValueError("AZURE_OPENAI_API_KEY is not set")
    if not DEPLOYMENT_NAME:
        raise ValueError("DEPLOYMENT_NAME is not set")
    
    # Clean the endpoint URL
    base = AZURE_OPENAI_ENDPOINT.rstrip("/")
    
    # Construct the API URL with API version parameter (required for Azure OpenAI)
    url = f"{base}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"
    
    print(f"=== Azure OpenAI Configuration ===")
    print(f"Endpoint base: {base}")
    print(f"Deployment name: {DEPLOYMENT_NAME}")
    print(f"API version: {API_VERSION}")
    print(f"Full URL: {url}")
    print(f"API key present: {'Yes' if AZURE_OPENAI_API_KEY else 'No'}")
    print(f"API key length: {len(AZURE_OPENAI_API_KEY) if AZURE_OPENAI_API_KEY else 0} characters")
    print(f"================================")

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are a senior cybersecurity practitioner writing for LinkedIn. Your audience is security leaders, practitioners, and technical managers. You write in a calm, confident, practical tone with an executive-technical style. You explain why things matter and what you would do next based on experience, not theory. You avoid bullet points, lists, headings, and generic advice. You end with a subtle forward-looking insight, not a question."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 1,
    }

    response = requests.post(url, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(f"\n❌ LLM request failed!")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        print(f"\nDebugging info:")
        print(f"  - Endpoint: {AZURE_OPENAI_ENDPOINT}")
        print(f"  - Deployment: {DEPLOYMENT_NAME}")
        print(f"  - API Version: {API_VERSION}")
        print(f"  - URL called: {url}")
        
        # Write a helpful error file for debugging
        with open(OUTPUT_FILE, "w") as f:
            json.dump({
                "error": "LLM request failed", 
                "status": response.status_code, 
                "response": response.text,
                "url": url,
                "deployment": DEPLOYMENT_NAME,
                "endpoint": AZURE_OPENAI_ENDPOINT,
                "api_version": API_VERSION
            }, f, indent=2)
        
        # Provide helpful error message
        if response.status_code == 404:
            print(f"\n💡 Troubleshooting 404 'Resource not found' error:")
            print(f"   1. Verify DEPLOYMENT_NAME matches your Azure deployment exactly")
            print(f"   2. Check that the deployment exists in your Azure OpenAI resource")
            print(f"   3. Ensure the deployment name in GitHub secrets is: {DEPLOYMENT_NAME}")
            print(f"   4. Verify the endpoint URL is correct: {AZURE_OPENAI_ENDPOINT}")
        
        import sys
        sys.exit(1)

    return response.json()["choices"][0]["message"]["content"]


def match_headline_to_article(headline, articles):
    """Match a headline from LLM response to the original article to get the link."""
    # Try exact match first
    for article in articles:
        if article["title"].strip() == headline.strip():
            return article["link"]
    
    # Try case-insensitive match
    for article in articles:
        if article["title"].strip().lower() == headline.strip().lower():
            return article["link"]
    
    # Try partial match (headline contains article title or vice versa)
    for article in articles:
        title_lower = article["title"].strip().lower()
        headline_lower = headline.strip().lower()
        if title_lower in headline_lower or headline_lower in title_lower:
            return article["link"]
    
    return None


def format_email_html(llm_response, articles, lens_name=None):
    """Convert LLM markdown response to HTML email with article links."""
    lens_line = f'<p><strong>Today\'s lens:</strong> {lens_name}</p>\n            ' if lens_name else ""
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }}
            .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #1a1a1a; border-bottom: 3px solid #007AFF; padding-bottom: 10px; margin-top: 0; }}
            h2 {{ color: #2c3e50; margin-top: 30px; margin-bottom: 15px; }}
            .article {{ background: #f8f9fa; border-left: 4px solid #007AFF; padding: 20px; margin: 25px 0; border-radius: 4px; }}
            .score {{ font-size: 20px; font-weight: bold; color: #007AFF; margin: 15px 0; }}
            ul {{ margin: 15px 0; padding-left: 25px; }}
            li {{ margin: 10px 0; }}
            .link {{ color: #007AFF; text-decoration: none; font-weight: 600; }}
            .link:hover {{ text-decoration: underline; }}
            .angle-tag {{ background: #e8eaf6; border-left: 4px solid #3f51b5; padding: 10px 15px; margin: 10px 0; border-radius: 4px; font-size: 14px; color: #283593; }}
            .board-one-liner {{ background: #f0f4f8; border-left: 4px solid #5c6bc0; padding: 12px 15px; margin: 12px 0; border-radius: 4px; font-style: italic; color: #37474f; }}
            .article-summary {{ background: #f5f5f5; border-left: 4px solid #616161; padding: 15px; margin: 15px 0; border-radius: 4px; }}
            .briefing-paragraph {{ background: #e3f2fd; border-left: 4px solid #1976d2; padding: 20px; margin: 20px 0; border-radius: 4px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
            .briefing-paragraph h3 {{ color: #1565c0; margin-top: 0; font-size: 16px; }}
            .briefing-paragraph .briefing-text {{ background: white; padding: 15px; border-radius: 4px; font-size: 15px; line-height: 1.7; color: #333; }}
            .briefing-variant {{ margin: 12px 0; }}
            .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 14px; text-align: center; }}
            p {{ margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔒 Daily Cybersecurity Briefing</h1>
            <p><strong>Date:</strong> {date.today().strftime('%B %d, %Y')}</p>
            {lens_line}
    """
    
    # Split response into sections by "---" (on its own line)
    # Handle both "\n---\n" and "\n---" patterns
    sections = re.split(r'\n---\s*\n', llm_response)
    if len(sections) == 1:
        # Try alternative splitting
        sections = re.split(r'\n---\n', llm_response)
    
    article_count = 0
    for section in sections:
        if not section.strip():
            continue
        
        # Try multiple headline patterns
        headline = None
        
        # Pattern 1: Numbered format "1) Title" or "1. Title" at start
        headline_match = re.search(r'^\d+[\)\.]\s*(.+?)(?:\n|$)', section, re.MULTILINE)
        if headline_match:
            headline = headline_match.group(1).strip()
        else:
            # Pattern 2: "Headline: **text**" or "Headline: text"
            headline_match = re.search(r'Headline:\s*\*\*(.+?)\*\*|Headline:\s*(.+?)(?:\n|$)', section, re.IGNORECASE | re.DOTALL)
            if headline_match:
                headline = (headline_match.group(1) or headline_match.group(2)).strip()
            else:
                # Pattern 3: Bold text at start of section
                headline_match = re.search(r'^\*\*(.+?)\*\*', section, re.MULTILINE)
                if headline_match:
                    headline = headline_match.group(1).strip()
        
        if headline:
            headline = headline.strip('*').strip()
            article_count += 1
            
            # Find matching article link
            article_link = match_headline_to_article(headline, articles)
            
            # Start article div
            if article_link:
                html_content += f'<div class="article"><h2><a href="{article_link}" class="link" target="_blank">{headline}</a></h2>\n'
            else:
                html_content += f'<div class="article"><h2>{headline}</h2>\n'
            
            # Extract score - try multiple patterns
            score = None
            score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)\s*/?\s*10|Score:\s*\*\*(\d+(?:\.\d+)?)/10\*\*|\*\*(\d+(?:\.\d+)?)/10\*\*', section)
            if score_match:
                score = score_match.group(1) or score_match.group(2) or score_match.group(3)
            
            if score:
                html_content += f'<div class="score">Score: {score}/10</div>\n'
            
            # Extract Key Takeaways bullets (limit to 2-3)
            takeaways_match = re.search(r'Key Takeaways?:(.+?)(?=Angle for this story|One-Line Board Take|Article Summary|Board-Level Impact|Briefing|$)', section, re.IGNORECASE | re.DOTALL)
            bullets = []
            if takeaways_match:
                takeaways_text = takeaways_match.group(1)
                bullets = re.findall(r'[-•]\s*(.+?)(?=\n|$)', takeaways_text, re.MULTILINE)
            else:
                bullets_match = re.search(r'^[-•]\s*(.+?)$', section, re.MULTILINE)
                if bullets_match:
                    all_bullets = re.findall(r'^[-•]\s*(.+?)$', section, re.MULTILINE)
                    bullets = [b.strip() for b in all_bullets[:3] if b.strip() and len(b.strip()) > 5]
            
            if bullets:
                html_content += '<ul>\n'
                for bullet in bullets[:3]:
                    bullet = bullet.strip()
                    if bullet and len(bullet) > 5:
                        html_content += f'<li>{bullet}</li>\n'
                html_content += '</ul>\n'
            
            # Extract Angle for this story (LLM's chosen angle for this article)
            angle_match = re.search(r'Angle for this story[^:]*:\s*(?:\n\s*)?([^\n]+)', section, re.IGNORECASE)
            if angle_match:
                angle_line = angle_match.group(1).strip().replace('**', '').replace('*', '').strip()[:120]
                if angle_line:
                    html_content += f'<div class="angle-tag"><strong>Angle for this story:</strong> {angle_line}</div>\n'
            
            # Extract One-Line Board Take (single line, under 15 words)
            one_liner_match = re.search(r'One-Line Board Take[^:]*:\s*([^\n]+)', section, re.IGNORECASE)
            if one_liner_match:
                one_liner = one_liner_match.group(1).strip().replace('**', '').replace('*', '').strip()
                if one_liner and len(one_liner) < 200:
                    html_content += f'<div class="board-one-liner"><strong>One-line board take:</strong> {one_liner}</div>\n'
            
            # Extract Article Summary section (stop at Briefing - Variant A)
            summary_match = re.search(r'Article Summary:(.+?)(?=Briefing - Variant A|Briefing - Variant B|Variant A|Variant B|$)', section, re.IGNORECASE | re.DOTALL)
            if not summary_match:
                summary_match = re.search(r'Board-Level Impact:(.+?)(?=Briefing - Variant A|Briefing - Variant B|Variant A|Variant B|$)', section, re.IGNORECASE | re.DOTALL)
            if summary_match:
                summary_text = summary_match.group(1).strip()
                summary_text_html = summary_text.replace('\n', '<br>')
                summary_text_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', summary_text_html)
                html_content += f'''
                <div class="article-summary">
                    <strong>Article Summary:</strong><br>
                    {summary_text_html}
                </div>
                '''
            
            # Extract Briefing Variant A and Variant B (LinkedIn-ready paragraphs)
            variant_a_text = None
            variant_b_text = None
            variant_a_match = re.search(r'Briefing - Variant A[^:]*:\s*(.+?)(?=Briefing - Variant B|Variant B[^:]*:|$)', section, re.IGNORECASE | re.DOTALL)
            if variant_a_match:
                variant_a_text = variant_a_match.group(1).strip()
            variant_b_match = re.search(r'Briefing - Variant B[^:]*:\s*(.+?)(?=\n---|\Z)', section, re.IGNORECASE | re.DOTALL)
            if variant_b_match:
                variant_b_text = variant_b_match.group(1).strip()
            # Fallback: legacy single "Briefing Paragraph" if present
            if not variant_a_text and not variant_b_text:
                legacy_match = re.search(r'Briefing Paragraph[^:]*:\s*(.+?)(?=\n---|\Z)', section, re.IGNORECASE | re.DOTALL)
                if legacy_match:
                    variant_a_text = legacy_match.group(1).strip()
            for label, text in [("Variant A (lead with so-what)", variant_a_text), ("Variant B (lead with concrete detail)", variant_b_text)]:
                if text:
                    clean = text.replace('**', '').replace('*', '').strip()
                    if len(clean) > 50:
                        clean_html = clean.replace('\n', '<br>')
                        html_content += f'''
                <div class="briefing-paragraph briefing-variant">
                    <h3>📱 {label} — LinkedIn-ready</h3>
                    <div class="briefing-text">{clean_html}</div>
                </div>
                '''
            
            html_content += '</div>\n'
    
    # Debug: print how many articles were found
    if article_count == 0:
        print(f"Warning: No articles found in response. Response length: {len(llm_response)}")
        print(f"First 500 chars: {llm_response[:500]}")
    
    html_content += """
            <div class="footer">
                <p>Generated by Hermes Relay - Your daily cybersecurity intelligence briefing</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content


def send_email(html_content, subject="Daily Cybersecurity Briefing"):
    """Send email via iCloud SMTP."""
    if not ICLOUD_EMAIL or not ICLOUD_PASSWORD:
        print("Email credentials not configured. Skipping email send.")
        print("Set ICLOUD_EMAIL and ICLOUD_PASSWORD environment variables to enable email.")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"{subject} - {date.today().strftime('%B %d, %Y')}"
        msg['From'] = ICLOUD_EMAIL
        msg['To'] = EMAIL_RECIPIENT
        
        # Create HTML part
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        # Send email via iCloud SMTP
        print(f"Sending email to {EMAIL_RECIPIENT}...")
        print(f"Using SMTP server: smtp.mail.me.com:587")
        print(f"Email address: {ICLOUD_EMAIL}")
        print(f"Password length: {len(ICLOUD_PASSWORD) if ICLOUD_PASSWORD else 0} characters")
        
        with smtplib.SMTP('smtp.mail.me.com', 587) as server:
            print("Connecting to SMTP server...")
            server.set_debuglevel(1)  # Enable debug output
            print("Starting TLS...")
            server.starttls()
            print(f"Attempting login with email: {ICLOUD_EMAIL}")
            server.login(ICLOUD_EMAIL, ICLOUD_PASSWORD)
            print("Login successful! Sending message...")
            server.send_message(msg)
        
        print("Email sent successfully!")
        return True
    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ SMTP Authentication Failed!")
        print(f"Error details: {e}")
        print(f"\nTroubleshooting steps:")
        print(f"1. Verify ICLOUD_EMAIL is correct: {ICLOUD_EMAIL}")
        print(f"2. Verify ICLOUD_PASSWORD is an app-specific password (not your regular password)")
        print(f"3. Check that 2FA is enabled on your Apple ID")
        print(f"4. Generate a new app-specific password at: https://appleid.apple.com")
        print(f"5. Make sure there are no extra spaces in your .env file")
        return False
    except smtplib.SMTPException as e:
        print(f"❌ SMTP Error: {e}")
        print(f"Error code: {e.smtp_code if hasattr(e, 'smtp_code') else 'N/A'}")
        print(f"Error message: {e.smtp_error if hasattr(e, 'smtp_error') else str(e)}")
        return False
    except Exception as e:
        print(f"❌ Failed to send email: {type(e).__name__}: {e}")
        import traceback
        print(f"Full traceback:")
        traceback.print_exc()
        return False


# -----------------------------
# MAIN
# -----------------------------
def main():
    html_file = OUTPUT_DIR / f"hermes_briefing_{today}.html"
    signal_file = Path(f"hermes_signal_{today}.json")
    
    # First, check if today's signal file exists and has new articles
    has_new_articles = False
    signal_file_time = 0
    output_file_time = 0
    
    if signal_file.exists():
        signal_file_time = signal_file.stat().st_mtime
        try:
            with open(signal_file, "r", encoding="utf-8") as f:
                signal_articles = json.load(f)
                if signal_articles and len(signal_articles) > 0:
                    has_new_articles = True
                    print(f"Found {len(signal_articles)} new article(s) in today's signal file")
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not read {signal_file}")
    
    # Check if output files exist and get their modification time
    if OUTPUT_FILE.exists():
        output_file_time = OUTPUT_FILE.stat().st_mtime
    
    # Determine if we need to regenerate
    should_regenerate = False
    
    if not has_new_articles:
        print("No new articles found in today's signal file.")
        if html_file.exists():
            print("Using existing HTML file...")
            with open(html_file, "r", encoding="utf-8") as f:
                html_email = f.read()
            send_email(html_email)
            print("Done! Email sent from existing file.")
            return
        else:
            print("No HTML file exists and no new articles. Nothing to send.")
            return
    
    # If we have new articles, check if signal file is newer than output
    if has_new_articles:
        if not OUTPUT_FILE.exists() or signal_file_time > output_file_time:
            should_regenerate = True
            print("Signal file is newer than output files. Regenerating summaries...")
        else:
            print("Output files exist and are up to date. Using existing files...")
            if html_file.exists():
                with open(html_file, "r", encoding="utf-8") as f:
                    html_email = f.read()
                send_email(html_email)
                print("Done! Email sent from existing file.")
                return
            elif OUTPUT_FILE.exists():
                # Regenerate HTML from existing JSON
                with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                articles = load_articles()
                html_email = format_email_html(data["top_articles"], articles, lens_name=data.get("lens"))
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(html_email)
                send_email(html_email)
                print("Done! Email sent from regenerated HTML.")
                return
    
    # Generate new summaries from today's new articles
    if should_regenerate:
        print("Generating new summaries from today's new articles...")
        articles = load_articles()
        
        if not articles or len(articles) == 0:
            print("No articles to process. Exiting.")
            return
        
        lens_name, lens_description = get_lens_for_date(date.today())
        print(f"Today's lens: {lens_name}")

        prompt = build_prompt(articles, lens_name, lens_description)

        print("Calling Azure OpenAI…")
        result = call_llm(prompt)

        # Save JSON output (for backup/debugging)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "date": date.today().isoformat(),
                    "lens": lens_name,
                    "top_articles": result,
                },
                f,
                indent=2,
            )
        print(f"Saved output → {OUTPUT_FILE}")

        # Format as HTML email and send
        html_email = format_email_html(result, articles, lens_name=lens_name)
        
        # Save HTML to the same directory as JSON
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_email)
        print(f"Saved HTML → {html_file}")
        
        send_email(html_email)
        print("Done! New summaries generated and email sent.")


if __name__ == "__main__":
    main()
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


def build_prompt(articles):
    return f"""
You are a senior cybersecurity analyst advising executives.

Task:
- You are analyzing NEW articles from today ({today}) - these are fresh cybersecurity news items that have just been published.
- Score each article from 1–10 based on business risk, reputational damage, and executive concern.
- Select the top 3 most critical articles from these new items.
- For each article, provide in this EXACT format (use "---" to separate each article):

1) [Headline - use exact title from article]
Score: [X]/10

Key Takeaways:
- [2-3 short, critical bullet points only - most important points]

Board-Level Impact:
[Write 2-3 paragraphs explaining why the board should care. Focus on:
- Financial exposure (potential costs, fines, revenue impact)
- Reputational risk (customer trust, brand damage, media attention)
- Regulatory/legal exposure (compliance issues, lawsuits, regulatory fines)
- Strategic risk (competitive disadvantage, market position)
Write this as a board briefing - direct, factual, focused on business consequences]

Briefing Paragraph:
[REQUIRED: Write exactly ONE paragraph (90–140 words) on the next line. You are a senior cybersecurity practitioner writing for LinkedIn; audience is security leaders, practitioners, and technical managers. The paragraph must: contain NO bullet points, NO lists, NO headings; sound natural and human, not academic or AI-generated; focus on practical, real-world implications; give actionable guidance a security team could act on; avoid generic advice like "implement MFA" unless directly relevant; avoid restating the article headline. Explain "why this matters" and "what I would do next" from experience. Tone: calm, confident, practical. Style: executive-technical. End with a subtle forward-looking insight, not a question. Output the paragraph directly under "Briefing Paragraph:" with no subheadings.]

---

[Repeat for article 2]

---

[Repeat for article 3]

IMPORTANT: 
- Use the exact article title/headline as it appears in the articles list below
- Use "---" on its own line to separate each of the 3 articles
- Make sure you provide all 3 articles
- You MUST include "Briefing Paragraph:" followed by a single paragraph (90–140 words) for every article. Do not skip this section.

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


def format_email_html(llm_response, articles):
    """Convert LLM markdown response to HTML email with article links."""
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
            .executive-note {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 4px; }}
            .briefing-paragraph {{ background: #e3f2fd; border-left: 4px solid #1976d2; padding: 20px; margin: 20px 0; border-radius: 4px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
            .briefing-paragraph h3 {{ color: #1565c0; margin-top: 0; font-size: 16px; }}
            .briefing-paragraph .briefing-text {{ background: white; padding: 15px; border-radius: 4px; font-size: 15px; line-height: 1.7; color: #333; }}
            .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 14px; text-align: center; }}
            p {{ margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔒 Daily Cybersecurity Briefing</h1>
            <p><strong>Date:</strong> {date.today().strftime('%B %d, %Y')}</p>
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
            takeaways_match = re.search(r'Key Takeaways?:(.+?)(?=Board-Level Impact|Briefing Paragraph|$)', section, re.IGNORECASE | re.DOTALL)
            bullets = []
            if takeaways_match:
                takeaways_text = takeaways_match.group(1)
                bullets = re.findall(r'[-•]\s*(.+?)(?=\n|$)', takeaways_text, re.MULTILINE)
            else:
                # Fallback: look for bullets before "Board-Level Impact" or "Briefing Paragraph"
                bullets_match = re.search(r'^[-•]\s*(.+?)$', section, re.MULTILINE)
                if bullets_match:
                    all_bullets = re.findall(r'^[-•]\s*(.+?)$', section, re.MULTILINE)
                    # Take first 3 bullets
                    bullets = [b.strip() for b in all_bullets[:3] if b.strip() and len(b.strip()) > 5]
            
            if bullets:
                html_content += '<ul>\n'
                for bullet in bullets[:3]:  # Limit to 3 bullets
                    bullet = bullet.strip()
                    if bullet and len(bullet) > 5:
                        html_content += f'<li>{bullet}</li>\n'
                html_content += '</ul>\n'
            
            # Extract Board-Level Impact section
            board_match = re.search(r'Board-Level Impact:(.+?)(?=Briefing Paragraph|$)', section, re.IGNORECASE | re.DOTALL)
            if board_match:
                board_text = board_match.group(1).strip()
                # Convert newlines to <br> for HTML display
                board_text_html = board_text.replace('\n', '<br>')
                # Convert markdown bold to HTML
                board_text_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', board_text_html)
                
                html_content += f'''
                <div class="executive-note">
                    <strong>Board-Level Impact:</strong><br>
                    {board_text_html}
                </div>
                '''
            
            # Extract Briefing Paragraph section (LinkedIn-style one paragraph)
            briefing_text = None
            briefing_match = re.search(r'Briefing Paragraph[^:]*:\s*(.+?)(?=\n---|\Z)', section, re.IGNORECASE | re.DOTALL)
            if briefing_match:
                briefing_text = briefing_match.group(1).strip()
            # Fallback: if model skipped the label, take the last block of text after Board-Level Impact (often the briefing)
            if not briefing_text and board_match:
                after_board = section[board_match.end():].strip()
                # Remove "Briefing Paragraph" if present without content, then take first substantial block
                after_board = re.sub(r'^\s*Briefing Paragraph[^:]*:\s*', '', after_board, flags=re.IGNORECASE)
                if len(after_board) > 80 and not re.match(r'^\s*[-*]\s', after_board):
                    briefing_text = after_board.split('\n---')[0].strip()
            if briefing_text:
                briefing_clean = briefing_text.replace('**', '').replace('*', '').strip()
                if len(briefing_clean) > 50:
                    briefing_html = briefing_clean.replace('\n', '<br>')
                    html_content += f'''
                <div class="briefing-paragraph">
                    <h3>📱 Briefing (LinkedIn-ready)</h3>
                    <div class="briefing-text">{briefing_html}</div>
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
                html_email = format_email_html(data["top_articles"], articles)
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
        
        prompt = build_prompt(articles)

        print("Calling Azure OpenAI…")
        result = call_llm(prompt)

        # Save JSON output (for backup/debugging)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "date": date.today().isoformat(),
                    "top_articles": result,
                },
                f,
                indent=2,
            )
        print(f"Saved output → {OUTPUT_FILE}")

        # Format as HTML email and send
        html_email = format_email_html(result, articles)
        
        # Save HTML to the same directory as JSON
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_email)
        print(f"Saved HTML → {html_file}")
        
        send_email(html_email)
        print("Done! New summaries generated and email sent.")


if __name__ == "__main__":
    main()
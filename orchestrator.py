import os
import subprocess
import sys

def check_env():
    required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"Missing environment variables: {missing}")
        sys.exit(1)

def run_script(script_name):
    print(f"\nRunning {script_name}...")
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        sys.exit(1)

    print(result.stdout)

def main():
    check_env()
    run_script("hermes-relay.py")
    run_script("llm_score_and_summarize.py")
    print("\nHermes Relay pipeline completed successfully.")

if __name__ == "__main__":
    main()
"""Ask an OpenRouter model to assess candidate claim C-02.

The LangSmith @traceable wrapper records the claim row and the model's
assessment as a trace. Set OPENROUTER_API_KEY and LANGSMITH_API_KEY in .env.
"""

import csv
import os
from pathlib import Path
from datetime import date

try:
    from dotenv import load_dotenv
    from langsmith import traceable
    from openai import OpenAI
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Project dependencies are missing. From Ubuntu/WSL, run:\n"
        "  sudo apt update && sudo apt install -y python3-venv\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  python -m pip install -r requirements.txt\n"
        "Then run: python check_claim.py"
    ) from exc


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "aws_reinvent_candidate_items.csv"

TODAY = date.today().isoformat()

def load_claim(candidate_id: str = "C-02") -> dict[str, str]:
    with DATA_PATH.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["candidate_id"] == candidate_id:
                return row
    raise ValueError(f"Candidate {candidate_id!r} was not found in {DATA_PATH}")


@traceable(name="check_aws_launch_claim", run_type="chain")
def assess_claim(row: dict[str, str]) -> str:
    """Assess a candidate claim; the row and response are captured in LangSmith."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env before running this script.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": (
                   """You are an AWS re:Invent claim verifier.
                    Evaluate the claim using only the supplied or retrieved evidence.
                    Never invent facts or sources. Return exactly ONE concise briefing of 1 to 3 lines,
                    combining the verdict, key explanation, source URL (if verified), and confidence.
                    Mention the most important evidence gap only if relevant. Use plain language,
                    no headings, bullets, separate fields, or introductory text. Distinguish unsupported
                    claims from claims that remain unverified. Consider the evaluation date when assessing dates.Maximum 60 words."""
                ),
            },
            {
                "role": "user",
                "content": (
                    "Is this a real AWS launch? Assess the claim and briefly explain "
                    "your confidence and evidence limitations.\n\n"
                    f"Candidate ID: {row['candidate_id']}\n"
                    f"Claim: {row['claim_text']}\n"
                    f"Claim type: {row['claim_type']}\n"
                    f"Origin: {row['origin_source_name']} ({row['origin_source_type']})\n"
                    f"Source URL: {row['origin_source_url']}\n"
                    f"Published: {row['origin_published_date']}\n"
                    f"Today's Date: {TODAY}"
                ),
            },
        ],
    )
    return response.choices[0].message.content or "The model returned an empty response."


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise RuntimeError("Set LANGSMITH_API_KEY in .env to record the trace.")
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "reinvent")

    row = load_claim("C-02")
    print(f"Checking {row['candidate_id']}: {row['claim_text']}\n")
    print(assess_claim(row))
    print("\nAssessment generated.")
    print(
        "LangSmith tracing is enabled, but check the LangSmith dashboard "
        "to confirm that the trace was actually recorded."
    )

if __name__ == "__main__":
    main()

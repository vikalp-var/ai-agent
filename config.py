import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── User plan: set USER_PLAN=free in .env for free-tier behaviour ─────────────
USER_PLAN = os.getenv("USER_PLAN", "paid")   # "free" | "paid"

# ── Model names ───────────────────────────────────────────────────────────────
# Complexity classifier — cheapest possible, ~$0.0001/call
CLASSIFIER_MODEL = "gpt-4.1-nano"

# Main agent models (Anthropic Claude)
CHEAP_MODEL    = "claude-sonnet-4-6"   #  standard / medium / low complexity steps
POWERFUL_MODEL = "claude-opus-4-7"     # high-complexity steps (paid users only)

# Cost per 1K tokens (approximate, USD)
MODEL_COSTS = {
    "gpt-4.1-nano":      {"input": 0.0001,  "output": 0.0004},   # classifier
    "claude-sonnet-4-6": {"input": 0.003,   "output": 0.015},    # standard
    "claude-opus-4-7":   {"input": 0.015,   "output": 0.075},    # powerful
}

WORKSPACE_DIR = "workspace"
LOG_DIR        = "logs"
MEMORY_FILE    = "memory.json"
MAX_RETRIES    = 3

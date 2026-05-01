"""
Smart model router.

Selection logic (mirrors the reference architecture):
  Free tier  → always CHEAP_MODEL (Claude Sonnet 4.6)
  Paid tier  → per-step complexity assessment:
    • complexity == "high"                → POWERFUL_MODEL (Claude Opus 4.7)
    • complexity == "medium" / "low"      → CHEAP_MODEL    (Claude Sonnet 4.6)

Complexity is classified by the planner (which uses GPT-4.1-nano, ~$0.0001/call).
The router just reads the pre-classified complexity label and maps it to a model.

For dynamic runtime classification (e.g. retry path without planner),
classify_task_complexity() calls GPT-4.1-nano directly.
"""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI

from config import CLASSIFIER_MODEL, CHEAP_MODEL, POWERFUL_MODEL, USER_PLAN

# ── Complexity classifier prompt (same logic as reference) ───────────────────

_CLASSIFIER_PROMPT = """You are a task complexity classifier for a coding AI assistant.

Classify the coding step below as HIGH or STANDARD.

HIGH — use for:
- Full React/web app with multiple components, complex UI
- Complex algorithms, system design, security-critical flows
- REST APIs with multiple endpoints, database schema design
- Multi-file refactors, long files (>100 lines), deep debugging
- Tasks requiring sustained reasoning across multiple systems

STANDARD — use for:
- Single-file edits, simple Python scripts
- Reading/listing files, writing config or text files
- Simple bug fixes, UI tweaks, straightforward CRUD
- Installing packages, running commands
- Any well-scoped, execution-oriented task

Default to STANDARD. Only choose HIGH when genuinely needed.

Respond with exactly one word: HIGH or STANDARD

Task step:
{description}"""


def classify_task_complexity(description: str) -> str:
    """
    Classify a task step using GPT-4.1-nano.
    Returns "high" or "medium" (maps to POWERFUL_MODEL or CHEAP_MODEL).
    Cost: ~$0.0001/call.
    """
    prompt = _CLASSIFIER_PROMPT.format(description=description[:800])
    llm = ChatOpenAI(model=CLASSIFIER_MODEL, temperature=0, max_tokens=4)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip().upper()
        return "high" if "HIGH" in answer else "medium"
    except Exception:
        return "medium"  # Fail safe —  cheaper model


def route_model(task_type: str, complexity: str = "medium") -> str:
    """
    Map a planner-labelled complexity → model name.

    Priority:
      1. Free tier  → always CHEAP_MODEL
      2. High complexity + paid → POWERFUL_MODEL (Claude Opus 4.7)
      3. Everything else       → CHEAP_MODEL    (Claude Sonnet 4.6)
    """
    is_free = USER_PLAN.lower().strip() in ("free", "none", "")
    if is_free:
        return CHEAP_MODEL
    if complexity == "high":
        return POWERFUL_MODEL
    return CHEAP_MODEL


def get_model_router() -> RunnableBranch:
    """
    LangChain RunnableBranch for model selection.

    Input:  {"task_type": str, "complexity": str}
    Output: model name string
    """
    is_free = USER_PLAN.lower().strip() in ("free", "none", "")
    return RunnableBranch(
        # Free tier → always cheap (checked first — overrides everything)
        (
            lambda _: is_free,
            RunnableLambda(lambda _: CHEAP_MODEL),
        ),
        # High complexity → powerful model
        (
            lambda x: x.get("complexity") == "high",
            RunnableLambda(lambda _: POWERFUL_MODEL),
        ),
        # Default → cheap (Sonnet)
        RunnableLambda(lambda _: CHEAP_MODEL),
    )



def estimate_complexity(task: str) -> str:
    """Heuristic complexity score from task description."""
    complex_kw = [
        "algorithm", "optimize", "architecture", "debug", "fix", "complex",
        "advanced", "design pattern", "refactor", "async", "concurrent",
        "distributed", "machine learning", "neural", "recursion",
    ]
    simple_kw = [
        "simple", "basic", "hello world", "list", "format",
        "template", "boilerplate", "example", "print",
    ]

    task_lower = task.lower()
    if any(kw in task_lower for kw in complex_kw):
        return "high"
    if any(kw in task_lower for kw in simple_kw):
        return "low"
    return "medium"

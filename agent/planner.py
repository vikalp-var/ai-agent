import json
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import CLASSIFIER_MODEL


class _Step(BaseModel):
    step_id: int = Field(description="Sequential step number starting from 1")
    description: str = Field(description="What to do in this step")
    tool: str = Field(description="One of: write_file | read_file | execute_code | list_files | run_command | think")
    complexity: str = Field(default="medium", description="low | medium | high")
    requires_approval: bool = Field(default=False, description="True only for run_command steps")


class _TaskPlan(BaseModel):
    steps: list[_Step] = Field(description="Ordered list of steps to complete the task")


_SYSTEM_PROMPT = """You are a coding task planner. Decompose the user's coding task into clear, ordered steps.

Rules:
- If conversation history is provided, this is a FOLLOW-UP task. Read the history carefully to understand what was already done, then plan steps that continue from there.
- For follow-ups like "i want answer also" or "add answers" — look at what files were previously written and plan to write the answers/additions to those same files.
- requires_approval must be true ONLY for run_command steps.
- Do NOT create steps like "Test in browser", "Open in browser", or "Verify in browser" — the app is auto-served and previewed by the framework; manual browser steps are useless and must be omitted.
- For text/data tasks (writing questions, answers, lists): use write_file only — do NOT add execute_code steps.
- Only add execute_code when the task involves running Python code.
- Keep steps atomic — one action per step.
- Do not include more than 6 steps.

Complexity rules (CRITICAL — this controls which AI model is used):
- complexity = "high": writing a full React/web app, multi-component UI, complex algorithms, REST APIs with multiple endpoints, any file > ~100 lines of code
- complexity = "medium": standard Python scripts, moderate web pages, simple APIs, single-component apps
- complexity = "low": reading files, listing files, running commands, writing simple config/text files

Web/frontend app rules (CRITICAL):
- NEVER plan a step that uses npm, Create React App, vite, or any build tool.
- NEVER plan a step that creates a separate App.js file.
- For React apps: plan to write SEPARATE files using CDN React (no npm):
  Step 1: write style.css (styling)
  Step 2+: write components/ComponentName.js (one step per component)
  LAST step: write index.html (entry point that loads CDN + component files + inlines ReactDOM.render)
- index.html must contain an inline <script type="text/babel"> at the bottom that defines the App
  function and calls ReactDOM.render() — do NOT reference an external App.js.
- Components must NOT use ES module import/export. Declare as global React functions.
- For very simple apps (counter, calculator): single index.html with inline JS, complexity="medium"."""


def decompose_task(
    task: str,
    memory_context: Optional[dict] = None,
    conv_history: Optional[list] = None,
) -> tuple[list, object]:
    """Break a task into ordered steps using ChatOpenAI structured output.

    Returns (steps_list, usage_object) where usage_object has .prompt_tokens and
    .completion_tokens attributes.
    """
    context_str = ""
    if memory_context:
        learnings = memory_context.get("learnings", [])
        if learnings:
            context_str = f"\nPrevious learnings:\n{json.dumps(learnings[-3:], indent=2)}\n"

    conv_str = ""
    if conv_history:
        lines = []
        for m in conv_history:
            prefix = "User" if m.get("role") == "user" else "Assistant"
            limit = 800 if m.get("text", "").startswith("[Wrote file:") else 300
            lines.append(f"{prefix}: {m.get('text', '')[:limit]}")
        conv_str = "\nConversation so far (use this for context):\n" + "\n".join(lines) + "\n"

    llm = ChatOpenAI(model=CLASSIFIER_MODEL, temperature=0, max_tokens=700)
    structured_llm = llm.with_structured_output(_TaskPlan, include_raw=True)

    result = structured_llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"{conv_str}{context_str}Coding task:\n{task}"),
    ])

    plan: Optional[_TaskPlan] = result.get("parsed")
    raw_msg = result.get("raw")

    # Extract usage from the raw AIMessage
    class _Usage:
        def __init__(self, inp: int, out: int):
            self.prompt_tokens = inp
            self.completion_tokens = out

    usage = _Usage(0, 0)
    if raw_msg and hasattr(raw_msg, "usage_metadata") and raw_msg.usage_metadata:
        usage = _Usage(
            raw_msg.usage_metadata.get("input_tokens", 0),
            raw_msg.usage_metadata.get("output_tokens", 0),
        )

    if not plan:
        # Parsing error fallback — single write_file step
        return [
            {
                "step_id": 1,
                "description": task,
                "tool": "write_file",
                "complexity": "medium",
                "requires_approval": False,
            }
        ], usage

    steps = [s.model_dump() for s in plan.steps]
    return steps, usage


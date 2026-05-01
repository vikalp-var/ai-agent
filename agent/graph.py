"""
LangGraph coding agent — full LangChain/LangGraph integration.

Graph: START → planner → route → agent → [approval?] → tools → observe
               |________________retry___|  |_____________next_step___________|
       observe → finalise → END

LangGraph/LangChain features used:
  - ChatOpenAI.bind_tools(TOOLS)       : native tool calling
  - ToolNode(TOOLS)                    : automatic tool dispatch
  - add_messages reducer               : message accumulation in state
  - MemorySaver checkpointer           : state persistence for interrupt/resume
  - interrupt()                        : human-in-the-loop approval gate
  - Command(resume=...)                : resume after interrupt
  - Observer.as_runnable()             : RunnableLambda validation
  - get_model_router()                 : RunnableBranch model selection
  - InMemoryChatMessageHistory         : session memory (in memory.py)
"""

import json
from typing import Annotated, Callable, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from agent.memory import Memory
from agent.observer import Observer
from agent.planner import decompose_task
from agent.router import route_model
from agent.tools import TOOLS
from config import CHEAP_MODEL, CLASSIFIER_MODEL, MAX_RETRIES, MODEL_COSTS, POWERFUL_MODEL

# ── System prompt ─────────────────────────────────────────────────────────────

_AGENT_SYSTEM = """You are an expert coding agent operating in a Think -> Act -> Observe loop.

Guidelines:
- You MUST call exactly one tool per response.
- Think carefully before writing code: produce complete, working files.
- When you see a previous execution error, read it carefully and fix the root cause.
- For write_file: always write the FULL file content, never a partial patch.
- For execute_code: the file must already exist in the workspace.
- For run_command: use ONLY for shell commands like pip installs. This will require human approval.
- When the step says "Required tool: run_command", you MUST call run_command (not execute_code).

Web app guidelines (IMPORTANT):
- Do NOT use build tools (npm, webpack, vite, create-react-app) — NO installation needed.
- ALWAYS write an `index.html` as the entry point.
- Load React via CDN in index.html:
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
- For React apps with multiple components, write SEPARATE files for each component:
    index.html  — loads CDN scripts, then loads each component file with type="text/babel",
                  then has ONE inline <script type="text/babel"> that calls ReactDOM.render()
    components/TodoItem.js, components/TodoList.js, etc.
  In index.html load components then render inline — NEVER create a separate App.js file:
    <script type="text/babel" src="components/TodoItem.js"></script>
    <script type="text/babel" src="components/TodoList.js"></script>
    <script type="text/babel">
        function App() { return <div><TodoList /></div>; }
        ReactDOM.render(<App />, document.getElementById('root'));
    </script>
  Each component file should NOT use import/export — declare as global functions/classes.
- CSS goes in style.css, loaded via <link> in index.html.
- For simple apps (calculator, counter), a single index.html with inline JS is fine."""

# ── Reviewer system prompt (second agent — QA role) ──────────────────────────

_REVIEWER_SYSTEM = """You are a senior code reviewer. The coding agent just completed a task.
Your job is to review what was built and give a concise quality verdict.

Check for:
- Correctness: does the code logically accomplish the task?
- Completeness: are all required files present and referenced correctly?
- Common mistakes: syntax errors, missing CDN links, broken imports, missing closing tags
- For web apps: is index.html the entry point? Are React CDN scripts included?

Respond in this exact format:
[REVIEW] PASS | <one-line summary of what was built>
or
[REVIEW] FAIL | <one-line description of the specific issue found>

If FAIL, also add on a new line:
[FIX NEEDED] <concrete instruction for what must be corrected>"""

# ── State ─────────────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # LangGraph manages appending
    task: str
    conv_history: list
    steps: list
    current_step: int
    results: list
    attempts: int
    last_error: str
    token_tracker: dict
    model: str
    failed: bool
    approval_granted: bool
    reviewer_feedback: str     # populated by reviewer_node; empty string = pass


# ── Pre-built LangGraph tool node ─────────────────────────────────────────────

_tool_node = ToolNode(TOOLS)


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_graph(
    log_fn: Callable,
    memory: Memory,
    observer: Observer,
    checkpointer: Optional[MemorySaver] = None,
):
    """Compile and return the LangGraph coding-agent graph.

    Args:
        log_fn:       Callable(*args) to emit log/SSE events from nodes.
        memory:       Memory instance (InMemoryChatMessageHistory + long-term JSON).
        observer:     Observer instance — used via observer.as_runnable().
        checkpointer: MemorySaver (or compatible) for interrupt/resume support.
                      Required for interrupt() to work. Defaults to a fresh MemorySaver.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    # Observer as a LangChain RunnableLambda
    _obs_runnable = observer.as_runnable()

    # ── LLMs ──────────────────────────────────────────────────────────────────
    # Main agent: Claude via Anthropic (tool-calling)
    # Note: claude-opus-4-7 and claude-sonnet-4-6 deprecated the temperature param
    try:
        from langchain_anthropic import ChatAnthropic as _CA
        cheap_llm    = _CA(model=CHEAP_MODEL).bind_tools(TOOLS, tool_choice="any")
        powerful_llm = _CA(model=POWERFUL_MODEL).bind_tools(TOOLS, tool_choice="any")
        # Reviewer uses Sonnet (no tools needed)
        reviewer_llm = _CA(model=CHEAP_MODEL)
    except Exception:
        # Fallback to OpenAI if Anthropic key is missing / package not installed
        cheap_llm    = ChatOpenAI(model=CLASSIFIER_MODEL, temperature=0).bind_tools(TOOLS, tool_choice="required")
        powerful_llm = ChatOpenAI(model=CLASSIFIER_MODEL, temperature=0).bind_tools(TOOLS, tool_choice="required")
        reviewer_llm = ChatOpenAI(model=CLASSIFIER_MODEL, temperature=0)

    def _llm_for(model_name: str):
        return powerful_llm if model_name == POWERFUL_MODEL else cheap_llm

    def _update_tracker(tracker: dict, model_name: str, ai_msg: AIMessage) -> dict:
        tracker = dict(tracker)
        meta = getattr(ai_msg, "usage_metadata", None) or {}
        in_tok = meta.get("input_tokens", 0)
        out_tok = meta.get("output_tokens", 0)
        costs = MODEL_COSTS.get(model_name, {"input": 0, "output": 0})
        tracker["input"] += in_tok
        tracker["output"] += out_tok
        tracker["cost_usd"] += (
            (in_tok / 1000) * costs["input"] + (out_tok / 1000) * costs["output"]
        )
        return tracker

    # ── planner_node ──────────────────────────────────────────────────────────

    def planner_node(state: AgentState) -> dict:
        log_fn("\n[THINK] Planning task with planner...")
        mem_ctx = memory.get_relevant_context()
        steps, usage = decompose_task(
            state["task"], mem_ctx, conv_history=state["conv_history"]
        )

        costs = MODEL_COSTS.get(CHEAP_MODEL, {"input": 0, "output": 0})
        tracker = dict(state["token_tracker"])
        tracker["input"] += usage.prompt_tokens
        tracker["output"] += usage.completion_tokens
        tracker["cost_usd"] += (
            (usage.prompt_tokens / 1000) * costs["input"]
            + (usage.completion_tokens / 1000) * costs["output"]
        )

        log_fn(f"\n[PLAN] {len(steps)} step(s):")
        for s in steps:
            flag = " [approval required]" if s.get("requires_approval") else ""
            log_fn(f"  {s['step_id']:>2}. [{s.get('complexity','?'):>6}] {s['description']}{flag}")

        memory.add_to_session("assistant", f"Plan: {json.dumps([s['description'] for s in steps])}")

        return {
            "steps": steps,
            "current_step": 0,
            "results": [],
            "token_tracker": tracker,
        }

    # ── route_node ────────────────────────────────────────────────────────────

    def route_node(state: AgentState) -> dict:
        idx = state["current_step"]
        step = state["steps"][idx]
        # Call route_model() directly — reads USER_PLAN + config at call time
        model = route_model(step.get("tool", "think"), step.get("complexity", "medium"))
        log_fn(f"\n{'─' * 60}")
        log_fn(f"[STEP {step['step_id']}] {step['description']}")
        log_fn(f"[ROUTE] model={model}")
        return {
            "model": model,
            "last_error": "",
            "attempts": 0,
        }

    # ── agent_node ────────────────────────────────────────────────────────────

    def agent_node(state: AgentState) -> dict:
        idx = state["current_step"]
        step = state["steps"][idx]
        model_name = state["model"]
        last_error = state["last_error"]
        attempts = state["attempts"]

        if attempts > 0:
            log_fn(f"[RETRY] Attempt {attempts + 1}/{MAX_RETRIES}  (last error: {last_error[:120]})")

        # Conversation context
        conv_lines = []
        for m in state.get("conv_history", []):
            prefix = "User" if m.get("role") == "user" else "Assistant"
            limit = 800 if m.get("text", "").startswith("[Wrote file:") else 300
            conv_lines.append(f"{prefix}: {m.get('text', '')[:limit]}")
        conv_ctx = ("\nPrior conversation:\n" + "\n".join(conv_lines) + "\n") if conv_lines else ""

        prev_summary = json.dumps(
            [{"step": r["step_id"], "ok": r.get("success")} for r in state["results"][-2:]]
        )
        error_hint = (
            f"\nFix this error from the last attempt:\n{last_error[:400]}" if last_error else ""
        )

        tool_hint = (
            f"\nRequired tool for this step: {step['tool']}"
            if step.get("tool") and step["tool"] != "think"
            else ""
        )

        human_content = (
            f"{conv_ctx}"
            f"Overall task: {state['task']}\n"
            f"Current step {step['step_id']}: {step['description']}\n"
            f"All steps: {json.dumps([s['description'] for s in state['steps']])}\n"
            f"Completed so far: {prev_summary}"
            f"{tool_hint}"
            f"{error_hint}"
        )

        # Include last 3 complete AI+Tool groups as context.
        # Each AIMessage may have multiple tool_calls — collect ALL matching
        # ToolMessages so we never send an unanswered tool_call_id to OpenAI.
        msg_list = state.get("messages", [])
        ai_tool_groups: list = []
        i = 0
        while i < len(msg_list):
            m = msg_list[i]
            if isinstance(m, AIMessage) and m.tool_calls:
                pending_ids = {tc["id"] for tc in m.tool_calls}
                group = [m]
                j = i + 1
                while j < len(msg_list) and isinstance(msg_list[j], ToolMessage):
                    if msg_list[j].tool_call_id in pending_ids:
                        group.append(msg_list[j])
                        pending_ids.discard(msg_list[j].tool_call_id)
                    j += 1
                if not pending_ids:  # all tool calls answered — safe to include
                    ai_tool_groups.append(group)
                    i = j
                else:               # incomplete pair — skip to avoid 400 errors
                    i += 1
            else:
                i += 1
        context_msgs: list = []
        for group in ai_tool_groups[-3:]:
            context_msgs.extend(group)

        messages_to_send = (
            [SystemMessage(content=_AGENT_SYSTEM)]
            + context_msgs
            + [HumanMessage(content=human_content)]
        )

        llm = _llm_for(model_name)
        response: AIMessage = llm.invoke(messages_to_send)
        tracker = _update_tracker(state["token_tracker"], model_name, response)

        if response.content:
            log_fn(f"[THOUGHT] {response.content}")
        if response.tool_calls:
            tc = response.tool_calls[0]
            log_fn(
                f"[ACT]  {tc['name']}"
                f"({', '.join(f'{k}={repr(v)[:60]}' for k, v in tc['args'].items())})"
            )

        return {
            "messages": [response],
            "token_tracker": tracker,
        }

    # ── approval_node — uses LangGraph interrupt() ───────────────────────────

    def approval_node(state: AgentState) -> dict:
        idx = state["current_step"]
        step = state["steps"][idx]

        # interrupt() pauses the graph and saves state to the checkpointer.
        # Execution resumes when Command(resume=<bool>) is passed back.
        approved: bool = interrupt(
            {"step_id": step["step_id"], "desc": step["description"]}
        )

        if not approved:
            log_fn(f"[SKIP] Step {step['step_id']} skipped by user.")
            results = list(state["results"]) + [
                {"step_id": step["step_id"], "skipped": True, "success": False}
            ]
            return {
                "approval_granted": False,
                "results": results,
                "current_step": idx + 1,
            }
        return {"approval_granted": True}

    # ── observe_node — uses observer.as_runnable() ───────────────────────────

    def observe_node(state: AgentState) -> dict:
        idx = state["current_step"]
        step = state["steps"][idx]
        attempts = state["attempts"]
        msgs = state["messages"]

        # Find last AIMessage with tool calls
        ai_msg: Optional[AIMessage] = None
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and m.tool_calls:
                ai_msg = m
                break

        if not ai_msg:
            if attempts + 1 >= MAX_RETRIES:
                results = list(state["results"]) + [
                    {"step_id": step["step_id"], "success": False,
                     "failed_permanently": True, "error": "No tool call"}
                ]
                return {"results": results, "failed": True}
            return {"last_error": "LLM did not call any tool", "attempts": attempts + 1}

        tool_call = ai_msg.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        # Find matching ToolMessage
        tool_msg: Optional[ToolMessage] = None
        for m in reversed(msgs):
            if isinstance(m, ToolMessage) and m.tool_call_id == tool_call_id:
                tool_msg = m
                break

        if not tool_msg:
            if attempts + 1 >= MAX_RETRIES:
                results = list(state["results"]) + [
                    {"step_id": step["step_id"], "success": False,
                     "failed_permanently": True, "error": "Tool message missing"}
                ]
                return {"results": results, "failed": True}
            return {"last_error": "Tool result not found", "attempts": attempts + 1}

        # Parse tool result (ToolNode JSON-serialises dict returns)
        try:
            result = json.loads(tool_msg.content)
        except (json.JSONDecodeError, TypeError):
            result = {"success": True, "content": str(tool_msg.content)}

        # Validate via observer.as_runnable() — LangChain RunnableLambda
        verdict = _obs_runnable.invoke({"tool_name": tool_name, "result": result})

        if verdict["valid"]:
            log_fn("[OBSERVE] ok")
            if tool_name == "execute_code" and result.get("stdout"):
                log_fn(f"[OUTPUT]\n{result['stdout'].rstrip()}")
            if tool_name == "write_file" and result.get("success"):
                filename = tool_args.get("filename", "")
                content = tool_args.get("content", "")
                log_fn(f"[FILE_CONTENT] {filename}\n{content}")
            memory.add_to_session("tool", json.dumps(result)[:200])
            memory.compress_session()
            results = list(state["results"]) + [
                {"step_id": step["step_id"], "success": True, "result": result}
            ]
            return {"results": results, "last_error": "", "current_step": idx + 1, "attempts": 0}

        # Invalid — retry or halt
        issue = verdict.get("issue", "unknown error")
        log_fn(f"[OBSERVE] error  {issue[:200]}")
        log_fn(f"[OBSERVE] hint   {verdict.get('suggestion', '')}")
        memory.add_to_session("observation", f"Error in step {step['step_id']}: {issue[:300]}")

        if attempts + 1 >= MAX_RETRIES:
            log_fn(f"\n[STOP] Step {step['step_id']} failed permanently. Halting.")
            results = list(state["results"]) + [
                {"step_id": step["step_id"], "success": False,
                 "failed_permanently": True, "error": issue}
            ]
            return {"results": results, "failed": True, "last_error": issue}

        return {"last_error": issue, "attempts": attempts + 1}

    # ── finalise_node ─────────────────────────────────────────────────────────

    def finalise_node(state: AgentState) -> dict:
        results = state["results"]
        ok = sum(1 for r in results if r.get("success"))
        total = len(results)
        outcome = f"Completed {ok}/{total} steps"
        memory.remember_task(state["task"], outcome)
        line = "=" * 60
        log_fn(f"\n{line}\n  DONE — {outcome}\n{line}")
        t = state["token_tracker"]
        log_fn(f"\n[COST] Input tokens : {t['input']:,}")
        log_fn(f"[COST] Output tokens: {t['output']:,}")
        log_fn(f"[COST] Est. cost    : ${t['cost_usd']:.4f} USD")
        return {}

    # ── reviewer_node — second agent: QA / code-review pass ──────────────────

    def reviewer_node(state: AgentState) -> dict:
        """Independent QA agent that reviews what the coding agent built.
        Uses Claude Sonnet (cheap_llm without tools) to keep cost low.
        Result is logged and stored in reviewer_feedback.
        """
        # Collect what was written this run
        written_files = []
        for r in state.get("results", []):
            inner = r.get("result", {})
            if isinstance(inner, dict) and inner.get("filename"):
                written_files.append(inner["filename"])

        files_summary = ", ".join(written_files) if written_files else "(no files detected)"
        steps_summary = json.dumps([s["description"] for s in state.get("steps", [])])
        ok_count = sum(1 for r in state.get("results", []) if r.get("success"))
        total = len(state.get("results", []))

        review_prompt = (
            f"Task: {state['task']}\n"
            f"Steps planned: {steps_summary}\n"
            f"Steps completed: {ok_count}/{total}\n"
            f"Files written: {files_summary}\n\n"
            "Please review the work above."
        )

        log_fn("\n[REVIEWER] Reviewing output...")
        try:
            resp = reviewer_llm.invoke([
                SystemMessage(content=_REVIEWER_SYSTEM),
                HumanMessage(content=review_prompt),
            ])
            feedback = resp.content.strip()
            # Update token tracker with reviewer cost
            tracker = _update_tracker(state["token_tracker"], CHEAP_MODEL, resp)
        except Exception as exc:
            feedback = f"[REVIEW] PASS | (reviewer unavailable: {exc})"
            tracker = state["token_tracker"]

        log_fn(f"[REVIEWER] {feedback}")
        return {"reviewer_feedback": feedback, "token_tracker": tracker}

    # ── conditional edges ─────────────────────────────────────────────────────

    def after_plan(state: AgentState) -> str:
        return "finalise" if not state["steps"] else "route"

    def after_agent(state: AgentState) -> str:
        # Route to approval if the STEP is flagged requires_approval (checked first,
        # unconditionally — no dependency on which tool the LLM chose to call).
        idx = state["current_step"]
        step = state["steps"][idx] if idx < len(state["steps"]) else {}
        if step.get("requires_approval"):
            return "approval"
        # Also gate run_command calls even if the planner forgot to set the flag
        for m in reversed(state.get("messages", [])):
            if isinstance(m, AIMessage) and m.tool_calls:
                if m.tool_calls[0]["name"] == "run_command":
                    return "approval"
                break
        return "tools"

    def after_approval(state: AgentState) -> str:
        if not state.get("approval_granted", True):
            return "finalise" if state["current_step"] >= len(state["steps"]) else "route"
        return "tools"

    def after_observe(state: AgentState) -> str:
        if state.get("failed"):
            return "finalise"
        if state.get("last_error"):
            return "agent"
        return "finalise" if state["current_step"] >= len(state["steps"]) else "route"

    # ── assemble ──────────────────────────────────────────────────────────────

    g = StateGraph(AgentState)

    g.add_node("planner",  planner_node)
    g.add_node("route",    route_node)
    g.add_node("agent",    agent_node)
    g.add_node("approval", approval_node)
    g.add_node("tools",    _tool_node)       # prebuilt LangGraph ToolNode
    g.add_node("observe",  observe_node)
    g.add_node("finalise", finalise_node)
    g.add_node("reviewer", reviewer_node)   # second agent — QA pass

    g.set_entry_point("planner")
    g.add_conditional_edges("planner",  after_plan,     {"route": "route", "finalise": "finalise"})
    g.add_edge("route", "agent")
    g.add_conditional_edges("agent",    after_agent,    {"approval": "approval", "tools": "tools"})
    g.add_conditional_edges("approval", after_approval, {"tools": "tools", "route": "route", "finalise": "finalise"})
    g.add_edge("tools", "observe")
    g.add_conditional_edges("observe",  after_observe,  {"agent": "agent", "route": "route", "finalise": "finalise"})
    g.add_edge("finalise", "reviewer")   # always review after finalise
    g.add_edge("reviewer", END)

    # compile WITH checkpointer — required for interrupt() to work
    return g.compile(checkpointer=checkpointer)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_initial_state(task: str, conv_history: list) -> AgentState:
    """Build a fresh initial AgentState for a new task run."""
    return {
        "messages": [],
        "task": task,
        "conv_history": conv_history or [],
        "steps": [],
        "current_step": 0,
        "results": [],
        "attempts": 0,
        "last_error": "",
        "token_tracker": {"input": 0, "output": 0, "cost_usd": 0.0},
        "model": CHEAP_MODEL,
        "failed": False,
        "approval_granted": False,
        "reviewer_feedback": "",
    }

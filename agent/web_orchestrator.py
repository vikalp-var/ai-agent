import queue
import threading
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent.graph import build_graph, make_initial_state
from agent.memory import Memory
from agent.observer import Observer


class WebCodingAgent:
    """
    Web-aware LangGraph coding agent.

    - All log output is pushed to an SSE queue via _log().
    - Human-in-the-loop approvals use LangGraph interrupt() + Command(resume=...).
      The Flask /approve route fires the threading.Event that this class waits on.
    """

    def __init__(
        self,
        output_queue: queue.Queue,
        approval_event: threading.Event,
        approval_result: dict,
        clarification_event: Optional[threading.Event] = None,  # kept for compat, unused
        clarification_result: Optional[dict] = None,             # kept for compat, unused
        history: Optional[list] = None,
        session_id: Optional[str] = None,
    ):
        self._q = output_queue
        self._approval_event = approval_event
        self._approval_result = approval_result
        self._clarification_event = clarification_event or threading.Event()
        self._clarification_result = clarification_result or {"value": ""}
        self.history = history or []
        self._session_id = session_id or ""

        # Point tools at this session's subdirectory
        from agent.tools import set_session_subdir
        set_session_subdir(self._session_id)

        self._mem = Memory()
        self._obs = Observer()

        # Per-instance checkpointer so each WebCodingAgent run has isolated state.
        # Required for interrupt() to work.
        self._checkpointer = MemorySaver()

        # Pre-load conversation history into session memory
        for msg in self.history:
            role = "user" if msg.get("role") == "user" else "assistant"
            self._mem.add_to_session(role, msg.get("text", ""))

        # Build the LangGraph graph (with our log_fn and checkpointer)
        self._graph = build_graph(self._log, self._mem, self._obs, self._checkpointer)

    # -- log -> SSE queue --------------------------------------------------

    def _log(self, *args) -> None:
        message = " ".join(str(a) for a in args)

        # Inline file content block
        if message.startswith("[FILE_CONTENT] "):
            rest = message[len("[FILE_CONTENT] "):]
            nl_idx = rest.find("\n")
            if nl_idx != -1:
                self._q.put({
                    "type": "file_content",
                    "filename": rest[:nl_idx],
                    "content": rest[nl_idx + 1:],
                })
                return

        # Determine SSE event tag for UI styling
        tag = "log"
        if "[THINK]" in message or "[PLAN]" in message:
            tag = "think"
        elif "[STEP" in message or "[ROUTE]" in message:
            tag = "step"
        elif "[ACT]" in message:
            tag = "act"
        elif "[OBSERVE] ok" in message or "[OUTPUT]" in message:
            tag = "observe_ok"
        elif (
            "[OBSERVE] error" in message
            or "[OBSERVE] hint" in message
            or "[OBSERVE] \u2713" in message   # legacy ✓
            or "[OBSERVE] \u2717" in message   # legacy ✗
        ):
            tag = "observe_err"
        elif "[RETRY]" in message:
            tag = "retry"
        elif "[THOUGHT]" in message:
            tag = "thought"
        elif "[COST]" in message:
            tag = "cost"
        elif "[STOP]" in message or "[SKIP]" in message:
            tag = "warn"
        elif "[REVIEWER]" in message:
            tag = "reviewer"
        elif "===" in message:
            tag = "banner"

        self._q.put({"type": tag, "message": message})

    # -- public entry point ------------------------------------------------

    def run(self, task: str) -> None:
        """Run the agent. Blocking — call this in a background thread."""
        # recursion_limit: default is 25 which is too low for multi-step tasks.
        # Each step uses ~4 nodes (route→agent→tools→observe) + planner + finalise.
        # 200 handles up to ~45 steps safely.
        config = {"configurable": {"thread_id": "web-run"}, "recursion_limit": 200}
        input_val = make_initial_state(task, self.history)

        line = "=" * 60
        self._log(f"\n{line}\n  CODING AGENT  |  task: {task}\n{line}")

        while True:
            # Stream graph — nodes call _log() directly for real-time SSE output
            for _ in self._graph.stream(input_val, config, stream_mode="updates"):
                pass

            # Check whether the graph paused at an interrupt() node
            state = self._graph.get_state(config)
            if not state.next:
                break  # graph finished normally

            # Handle interrupt — human approval gate
            interrupted = False
            for task_item in state.tasks:
                for intr in task_item.interrupts:
                    data = intr.value  # {"step_id": int, "desc": str}
                    # Tell the UI approval is needed
                    self._q.put({
                        "type": "approval_needed",
                        "step_id": data["step_id"],
                        "desc": data["desc"],
                    })
                    # Wait for Flask /approve route to fire the event (5 min timeout)
                    self._approval_event.clear()
                    granted = self._approval_event.wait(timeout=300)
                    approved = (
                        bool(self._approval_result.get("value", False)) if granted else False
                    )
                    # Resume graph with user decision
                    input_val = Command(resume=approved)
                    interrupted = True
                    break
                if interrupted:
                    break

            if not interrupted:
                break  # No interrupts found — unexpected pause, exit safely

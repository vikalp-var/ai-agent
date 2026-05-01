"""
CLI entry point for the LangGraph coding agent.

Uses LangGraph's interrupt() + Command(resume=...) for human-in-the-loop approvals.
Run directly:  python -m agent.orchestrator  or  from agent.orchestrator import run_cli
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent.graph import AgentState, build_graph, make_initial_state
from agent.memory import Memory
from agent.observer import Observer


def run_cli(task: str, conv_history: list | None = None) -> None:
    """Run the LangGraph agent interactively in the terminal."""
    mem = Memory()
    obs = Observer()

    # Pre-load conversation history into session memory
    for msg in (conv_history or []):
        role = "user" if msg.get("role") == "user" else "assistant"
        mem.add_to_session(role, msg.get("text", ""))

    checkpointer = MemorySaver()
    graph = build_graph(print, mem, obs, checkpointer)
    # recursion_limit: default 25 is too low — each step needs ~4 graph nodes.
    config = {"configurable": {"thread_id": "cli"}, "recursion_limit": 200}

    # Build initial state
    input_val = make_initial_state(task, conv_history or [])

    line = "=" * 60
    print(f"\n{line}\n  CODING AGENT  |  task: {task}\n{line}")

    while True:
        # Invoke (or resume) the graph
        graph.invoke(input_val, config)

        # Check for pending interrupts (human approval gates)
        state = graph.get_state(config)
        if not state.next:
            break  # graph completed normally

        # Handle interrupt — collect input from user
        interrupted = False
        for task_item in state.tasks:
            for intr in task_item.interrupts:
                data = intr.value  # {"step_id": ..., "desc": ...}
                ans = input(
                    f'\n[HUMAN] Approve step {data["step_id"]} — "{data["desc"]}"? [y/n]: '
                ).strip().lower()
                approved = ans == "y"
                input_val = Command(resume=approved)
                interrupted = True
                break
            if interrupted:
                break

        if not interrupted:
            break  # no interrupt found — must be a different kind of pause


if __name__ == "__main__":
    import sys
    task_arg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not task_arg:
        task_arg = input("Enter task: ").strip()
    run_cli(task_arg)

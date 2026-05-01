import json
import os
from datetime import datetime

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage


class Memory:
    """
    Two-tier memory backed by LangChain chat history.
      - session_history : InMemoryChatMessageHistory (langchain_core)
      - long_term       : persisted JSON file (tasks + learnings)
    """

    def __init__(self, memory_file: str = "memory.json"):
        self.memory_file = memory_file
        self.session_history = InMemoryChatMessageHistory()
        self.long_term: dict = self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"tasks": [], "learnings": []}
        return {"tasks": [], "learnings": []}

    def save(self) -> None:
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.long_term, f, indent=2)

    # ── session (short-term) via  InMemoryChatMessageHistory ──────────────────

    def add_to_session(self, role: str, content: str) -> None:
        if role == "user":
            self.session_history.add_user_message(content)
        else:
            self.session_history.add_ai_message(content)

    def get_session_context(self, last_n: int = 10) -> list:
        msgs = self.session_history.messages[-last_n:]
        return [
            {
                "role": "assistant" if isinstance(m, AIMessage) else "user",
                "content": m.content,
            }
            for m in msgs
        ]

    def compress_session(self) -> None:
        """Drop middle messages when session grows large to save tokens."""
        msgs = self.session_history.messages
        if len(msgs) > 10:
            keep = msgs[:1] + msgs[-4:]
            self.session_history.clear()
            for m in keep:
                if isinstance(m, HumanMessage):
                    self.session_history.add_user_message(m.content)
                else:
                    self.session_history.add_ai_message(m.content)

    # ── long-term ─────────────────────────────────────────────────────────────

    def remember_task(self, task: str, outcome: str) -> None:
        self.long_term["tasks"].append(
            {"task": task, "outcome": outcome, "timestamp": datetime.now().isoformat()}
        )
        self.save()

    def add_learning(self, learning: str) -> None:
        self.long_term["learnings"].append(
            {"learning": learning, "timestamp": datetime.now().isoformat()}
        )
        self.save()

    def get_relevant_context(self) -> dict:
        """Return last 2 completed tasks + last 3 learnings for prompt context."""
        return {
            "recent_tasks": self.long_term["tasks"][-2:],
            "learnings": self.long_term["learnings"][-3:],
        }

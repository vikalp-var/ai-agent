import json
import os
from datetime import datetime
from pathlib import Path

from config import LOG_DIR


class AgentLogger:
    """
    Writes structured JSONL logs for every Think/Act/Observe cycle.
    Each line is a self-contained JSON object — easy to parse or pipe into
    any observability tool (Datadog, Grafana, etc.).
    """

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = Path(LOG_DIR) / f"session_{ts}.jsonl"

    # ── public API ────────────────────────────────────────────────────────────

    def log_step(
        self,
        step_id: int,
        thought: str,
        action: str,
        action_input: dict,
        model: str = "",
    ) -> None:
        self._write(
            {
                "type": "step",
                "step_id": step_id,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "model": model,
            }
        )

    def log_observation(
        self,
        step_id: int,
        result: dict,
        validation: dict,
        duration_ms: int,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self._write(
            {
                "type": "observation",
                "step_id": step_id,
                "success": result.get("success"),
                "validation_valid": validation.get("valid"),
                "needs_retry": validation.get("needs_retry", False),
                "duration_ms": duration_ms,
                "tokens_used": tokens_used,
                "cost_usd": round(cost_usd, 6),
            }
        )

    def log_plan(self, task: str, steps: list) -> None:
        self._write({"type": "plan", "task": task, "steps": steps})

    def log_cost_summary(self, tracker: dict) -> None:
        self._write({"type": "cost_summary", **tracker})

    # ── internal ──────────────────────────────────────────────────────────────

    def _write(self, payload: dict) -> None:
        payload["timestamp"] = datetime.now().isoformat()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

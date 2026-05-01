import re
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableLambda


class Observer:
    """
    Validates every tool result and decides whether to retry or escalate.
    Exposes validation logic as a LangChain RunnableLambda via as_runnable().
    """

    _ERROR_PATTERNS = (
        "traceback", "error:", "exception:", "syntaxerror",
        "nameerror", "typeerror", "valueerror", "importerror",
        "modulenotfounderror", "zerodivisionerror",
    )

    def validate_tool_result(self, tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        """Return a validation verdict for the caller."""
        if not result.get("success"):
            return {
                "valid": False,
                "needs_retry": True,
                "issue": result.get("error", "Unknown error"),
                "suggestion": self._suggest_fix(tool_name, result),
            }

        if tool_name == "execute_code":
            return self._validate_execution(result)

        if tool_name == "write_file":
            return self._validate_write_file(result)

        return {"valid": True, "needs_retry": False}

    def as_runnable(self) -> RunnableLambda:
        """Return a LangChain Runnable that validates tool results."""
        return RunnableLambda(
            lambda x: self.validate_tool_result(x["tool_name"], x["result"])
        )

    # -- private helpers   -------------------------------------------------------

    def _validate_execution(self, result: dict) -> dict:
        stderr = (result.get("stderr") or "").lower()
        returncode = result.get("returncode", 0)

        has_error = returncode != 0 or any(p in stderr for p in self._ERROR_PATTERNS)
        if has_error:
            issue = result.get("stderr") or result.get("stdout") or "Execution failed"
            return {
                "valid": False,
                "needs_retry": True,
                "issue": issue,
                "suggestion": "Read the error, fix the code in the file, then re-execute.",
            }

        return {"valid": True, "needs_retry": False, "output": result.get("stdout", "")}

    def _validate_write_file(self, result: dict) -> dict:
        """Check HTML files for <script src="..."> that reference missing local JS files.

        Catches the common case where the agent writes index.html referencing
        App.js (or other components) that were never created, breaking the preview.
        """
        filename = result.get("filename", "")
        content = result.get("content", "")

        if not filename.endswith(".html"):
            return {"valid": True, "needs_retry": False}

        # Find all <script ... src="foo.js"> attributes (skip CDN / absolute URLs)
        src_pattern = re.compile(
            r'<script[^>]+\bsrc=["\']([^"\'?#]+\.js)["\']',
            re.IGNORECASE,
        )
        srcs = src_pattern.findall(content)
        local_srcs = [s for s in srcs if not s.startswith(("http://", "https://", "//"))]

        if not local_srcs:
            return {"valid": True, "needs_retry": False}

        # Resolve paths relative to the HTML file location in the workspace
        from agent.tools import _workspace
        ws = _workspace()
        html_dir = (ws / filename).parent
        missing = [s for s in local_srcs if not (html_dir / s).exists()]

        if missing:
            return {
                "valid": False,
                "needs_retry": True,
                "issue": (
                    f"{filename} references JS files that do not exist: "
                    f"{', '.join(missing)}. "
                    "NEVER reference a separate App.js file. "
                    "Inline ALL React components and ReactDOM.render() directly inside "
                    'a single <script type="text/babel"> block at the bottom of index.html.'
                ),
                "suggestion": (
                    f"Rewrite {filename}: remove <script src=\"App.js\"> (or other missing "
                    "files) and put the App function + ReactDOM.render() inline."
                ),
            }

        return {"valid": True, "needs_retry": False}

    @staticmethod
    def _suggest_fix(tool_name: str, result: dict) -> str:
        error = (result.get("error") or "").lower()
        if "not found" in error:
            return "File may not exist yet - write it first with write_file."
        if "permission" in error:
            return "Permission denied - try a different filename or directory."
        if "syntax" in error:
            return "Fix the syntax error in the generated code."
        if "not in allowlist" in error:
            return "Use only whitelisted commands (pip install, pip list, etc.)."
        return "Retry with corrected parameters."

    @staticmethod
    def should_ask_human(action: str) -> bool:
        """Only run_command requires human approval (write_file is safe)."""
        return action == "run_command"

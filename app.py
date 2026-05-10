import json
import os
import queue
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path

CHAT_HISTORY_FILE = "chat_history.json"
MAX_HISTORY_SESSIONS = 50


def _load_chat_history() -> list:
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_chat_history(sessions: list) -> None:
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions[-MAX_HISTORY_SESSIONS:], f, indent=2)

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, send_from_directory, stream_with_context

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ── Shared agent state ────────────────────────────────────────────────────────
_output_queue: queue.Queue = queue.Queue()
_approval_event: threading.Event = threading.Event()
_approval_result: dict = {"value": None}
_agent_running: bool = False


# ── Conversational query detection & direct answering ────────────────────────

def _is_conversational_query(task: str, history: list) -> bool:
    """Return True if the query is conversational/general — not a coding task.
    Uses GPT-4.1-nano for cheap, fast classification."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        _SYS = (
            "You classify user messages for a coding AI agent.\n"
            "Reply with exactly one word: CONVERSATIONAL or CODING.\n\n"
            "CONVERSATIONAL — general chat, greetings, questions about previous work,\n"
            "  asking what was built, asking to explain or summarise, memory/history questions,\n"
            "  general knowledge questions not requiring code to be written.\n"
            "  Examples: 'hello', 'what did I build?', 'explain what you did',\n"
            "  'upar kya kiya', 'what was my last task', 'summarise my work',\n"
            "  'how are you', 'what is python', 'can you explain this'\n\n"
            "CODING — requests to write, create, fix, or run code / files / apps.\n"
            "  Examples: 'build a todo app', 'fix the bug', 'create a calculator',\n"
            "  'write a python script', 'add a dark mode button'\n\n"
            "Reply with ONE word only."
        )
        # Include last history message for context
        history_hint = ""
        if history:
            last = history[-1]
            history_hint = f"\nLast assistant output (summary): {str(last.get('text',''))[:200]}"
        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0, max_tokens=5)
        resp = llm.invoke([
            SystemMessage(content=_SYS),
            HumanMessage(content=f"User message: {task[:400]}{history_hint}"),
        ])
        return "CONVERSATIONAL" in resp.content.strip().upper()
    except Exception:
        return False


def _answer_conversational_query(task: str, history: list, output_queue: queue.Queue) -> None:
    """Answer a conversational query directly using the conversation history, push to SSE queue."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        _SYS = (
            "You are a helpful AI coding assistant. Answer the user's question in a friendly,\n"
            "concise way. If they ask about previous work, use the conversation history provided.\n"
            "If they greet you, greet back. If they ask what was built, summarise it from history.\n"
            "Always reply in English only, regardless of the language the user used."
        )
        # Build conversation context from history
        history_lines = []
        for m in history[-10:]:
            role = "User" if m.get("role") == "user" else "Assistant"
            text = m.get("text", "")[:500]
            history_lines.append(f"{role}: {text}")
        history_ctx = ("\nConversation history:\n" + "\n".join(history_lines)) if history_lines else ""

        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3, max_tokens=400)
        resp = llm.invoke([
            SystemMessage(content=_SYS),
            HumanMessage(content=f"{history_ctx}\n\nUser's current message: {task}"),
        ])
        answer = resp.content.strip()
        output_queue.put({"type": "log", "message": answer})
    except Exception as exc:
        output_queue.put({"type": "log", "message": f"Sorry, I couldn't answer that. ({exc})"})


# ── Pre-flight clarification (synchronous, before agent starts) ───────────────

def _generate_clarification_question(task: str) -> str:
    """Call gpt-4.1-nano synchronously and return ONE clarifying question.
    Returns "" on any error so the agent proceeds without clarification."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        _SYS = (
            "You are a friendly pre-flight assistant for a coding AI agent.\n"
            "The user wants to build something. Generate ONE short, targeted question\n"
            "(max 20 words) that would help the AI build exactly what the user wants.\n"
            "Focus on: key features, technology preferences, or important constraints.\n"
            "Output ONLY the question, nothing else."
        )
        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0, max_tokens=60)
        resp = llm.invoke([SystemMessage(content=_SYS), HumanMessage(content=task[:600])])
        q = resp.content.strip().strip('"').strip()
        return q if len(q) >= 5 else ""
    except Exception:
        return ""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    resp = app.make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/run", methods=["POST"])
def run():
    global _agent_running

    if _agent_running:
        return jsonify({"error": "Agent already running"}), 409

    data = request.get_json(force=True)
    task = (data.get("task") or "").strip()
    if not task:
        return jsonify({"error": "No task provided"}), 400
    history = data.get("history") or []
    # Sanitise: keep only expected keys, cap text length
    history = [
        {"role": str(m.get("role", ""))[:10], "text": str(m.get("text", ""))[:500]}
        for m in history
        if isinstance(m, dict)
    ][:20]
    session_id = str(data.get("session_id") or "")[:64].strip()

    # ── Conversational query check — answer directly without full agent ──────
    # If the user is asking a general question or asking about previous work,
    # answer immediately using LLM + history instead of running the coding agent.
    clarification_answer = str(data.get("clarification_answer") or "").strip()[:500]
    skip_clarification = bool(data.get("skip_clarification", False))

    if not clarification_answer and not skip_clarification:
        if _is_conversational_query(task, history):
            _agent_running = True

            def run_conversational():
                global _agent_running
                try:
                    _answer_conversational_query(task, history, _output_queue)
                finally:
                    _agent_running = False
                    _output_queue.put({"type": "done", "message": "Done."})

            threading.Thread(target=run_conversational, daemon=True).start()
            return jsonify({"status": "started"})

    # ── Synchronous pre-flight clarification ────────────────────────────────
    # On the FIRST call (no clarification_answer / skip flag), ask one question
    # and return immediately — the agent does NOT start yet.
    # On the second call (with clarification_answer or skip_clarification),
    # skip the question and enrich the task before running the agent.
    if not clarification_answer and not skip_clarification:
        question = _generate_clarification_question(task)
        if question:
            return jsonify({"status": "needs_clarification", "question": question})

    if clarification_answer:
        task = f"{task}\n\nAdditional context from user: {clarification_answer}"
    # ────────────────────────────────────────────────────────────────────────

    # Drain old queue
    while not _output_queue.empty():
        try:
            _output_queue.get_nowait()
        except queue.Empty:
            break

    _agent_running = True

    def run_agent():
        global _agent_running
        try:
            from agent.web_orchestrator import WebCodingAgent
            agent = WebCodingAgent(
                _output_queue, _approval_event, _approval_result,
                history=history, session_id=session_id,
            )
            agent.run(task)
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[AGENT ERROR]\n{tb}", flush=True)
            _output_queue.put({"type": "error", "message": f"[ERROR] {exc}\n\nDetails: {tb}"})
        finally:
            _agent_running = False
            _output_queue.put({"type": "done", "message": "Agent finished."})

    t = threading.Thread(target=run_agent, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                event = _output_queue.get(timeout=60)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "done":
                    break
            except queue.Empty:
                # Send heartbeat so connection stays alive
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/approve", methods=["POST"])
def approve():
    data = request.get_json(force=True)
    _approval_result["value"] = bool(data.get("approved", False))
    _approval_event.set()
    return jsonify({"status": "ok"})


@app.route("/files")
def list_files():
    session_id = request.args.get("session_id", "").strip()[:64]
    workspace = Path("workspace") / session_id if session_id else Path("workspace")
    if not workspace.exists():
        return jsonify({"files": []})
    files = [
        str(p.relative_to(workspace)).replace("\\", "/")
        for p in workspace.rglob("*")
        if p.is_file() and not p.name.startswith(".")
    ]
    return jsonify({"files": sorted(files)})


@app.route("/files/<path:filename>")
def read_file(filename):
    session_id = request.args.get("session_id", "").strip()[:64]
    base = Path("workspace") / session_id if session_id else Path("workspace")
    filepath = base / filename
    try:
        content = filepath.read_text(encoding="utf-8")
        return jsonify({"content": content})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 404


@app.route("/preview/<session_id>/<path:filename>")
def preview_file(session_id, filename):
    """Serve a session-scoped workspace file so HTML/CSS/JS renders in the browser."""
    workspace = (Path("workspace") / session_id).resolve()
    # Security: ensure resolved path stays inside workspace/
    if not str(workspace).startswith(str(Path("workspace").resolve())):
        return "Forbidden", 403
    return send_from_directory(workspace, filename)


@app.route("/status")
def status():
    return jsonify({"running": _agent_running})


@app.route("/chat-history", methods=["GET"])
def get_chat_history():
    return jsonify({"sessions": _load_chat_history()})


@app.route("/chat-history", methods=["POST"])
def post_chat_history():
    data = request.get_json(force=True)
    sessions = _load_chat_history()
    session_id = data.get("sessionId", "")
    # Update existing  entry for the same session instead of duplicating
    for entry in sessions:
        if entry.get("sessionId") == session_id:
            # Only update task if explicitly provided (first save sets it, follow-ups skip it)
            if data.get("task"):
                entry["task"] = data["task"]
            entry["cost"] = data.get("cost", entry["cost"])
            entry["completedAt"] = data.get("completedAt", entry["completedAt"])
            _save_chat_history(sessions)
            return jsonify({"status": "ok"})
    # No existing entry — append new one
    sessions.append({
        "id": str(uuid.uuid4()),
        "sessionId": session_id,
        "task": data.get("task", ""),
        "cost": data.get("cost", "$0"),
        "completedAt": data.get("completedAt", datetime.now().isoformat()),
    })
    _save_chat_history(sessions)
    return jsonify({"status": "ok"})


@app.route("/chat-history", methods=["DELETE"])
def clear_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)
    return jsonify({"status": "ok"})


@app.route("/generate-image", methods=["POST"])
def generate_image_route():
    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()[:1000]
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        url = response.data[0].url
        revised = getattr(response.data[0], "revised_prompt", prompt)
        return jsonify({"url": url, "revised_prompt": revised})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)

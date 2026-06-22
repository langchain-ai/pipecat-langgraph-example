"""FastAPI server: a thin chat frontend over the gym support graph.

The frontend is deliberately tiny — one static HTML page with a chat box. All
the interesting behavior is in the graph. Each browser session gets a
``thread_id``; the graph's checkpointer keys conversation state on it, so the
``active_agent`` survives across turns.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from langchain.messages import AIMessage
from pydantic import BaseModel

load_dotenv()  # pull .env (model + LangSmith keys) into the environment

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from .graph import build_graph  # noqa: E402  (must follow load_dotenv)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="LangGym Support")
# The text server persists per-thread state, so each request sends only the new
# message. (The voice path calls build_graph() with no checkpointer — stateless.)
graph = build_graph(checkpointer=InMemorySaver())


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    active_agent: str


@app.get("/")
def index() -> FileResponse:
    # Single known file — no user-controlled path, so no traversal risk.
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = graph.invoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config={"configurable": {"thread_id": req.thread_id}},
    )
    reply = next(
        (
            m.text() if hasattr(m, "text") else str(m.content)
            for m in reversed(result["messages"])
            if isinstance(m, AIMessage) and m.content
        ),
        "(no response)",
    )
    return ChatResponse(reply=reply, active_agent=result.get("active_agent") or "triage")


def main() -> None:
    """Console entry point: `pipecat-langgraph-example`."""
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"\n  LangGym Support running at http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

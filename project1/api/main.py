"""
Enterprise-RAG: FastAPI production backend with streaming support.
"""
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from src.config import config
from src.pipeline import RAGPipeline, get_pipeline

# Configure logging
logger.add(
    config.get("logging", {}).get("file", "./data/logs/rag.log"),
    rotation="100 MB",
    level=config.get("logging", {}).get("level", "INFO"),
)

app = FastAPI(
    title="Enterprise-RAG API",
    description="企业知识库 RAG 问答系统 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directories exist
Path("./data/uploads").mkdir(parents=True, exist_ok=True)
Path("./data/logs").mkdir(parents=True, exist_ok=True)


# ── Request/Response Models ──

class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20)
    use_cot: bool = Field(default=True)
    conversation_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    reasoning: str | None = None
    contexts: list[dict[str, Any]]
    question: str
    conversation_id: str | None = None


class IngestResponse(BaseModel):
    status: str
    chunks_count: int
    message: str


class DocumentInfo(BaseModel):
    name: str
    chunks: int
    size_bytes: int
    ingested_at: str


# ── In-memory state ──
_ingested_docs: dict[str, dict] = {}


# ── Startup ──

@app.on_event("startup")
async def startup():
    """Initialize pipeline on startup (non-blocking for demo ingestion)."""
    logger.info("Enterprise-RAG API starting...")
    # Pre-load the pipeline singleton (triggers model loading)
    try:
        get_pipeline()
        logger.info("Pipeline initialized")
    except Exception as e:
        logger.warning(f"Pipeline init warning (may need models): {e}")

    # Ingest demo data in background to not block startup
    import asyncio
    async def _ingest_demo():
        demo_dir = Path(config["project"].get("demo_data_dir", "./data/demo"))
        if demo_dir.exists() and any(demo_dir.iterdir()):
            try:
                pipeline = get_pipeline()
                count = pipeline.ingest_directory(str(demo_dir))
                logger.info(f"Demo data ingested: {count} chunks")
            except Exception as e:
                logger.warning(f"Demo ingestion skipped: {e}")

    asyncio.create_task(_ingest_demo())
    logger.info("Enterprise-RAG API ready")


# ── Health Check ──

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


# ── RAG Query ──

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute a RAG query."""
    pipeline = get_pipeline()
    try:
        result = pipeline.query(
            question=request.question,
            top_k=request.top_k,
            use_cot=request.use_cot,
        )
        return QueryResponse(
            answer=result.get("answer", ""),
            reasoning=result.get("reasoning"),
            contexts=result.get("contexts", []),
            question=request.question,
            conversation_id=request.conversation_id,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Streaming Query ──

@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """Execute a RAG query with SSE streaming."""
    pipeline = get_pipeline()

    async def generate():
        try:
            stream = pipeline.query(
                question=request.question,
                top_k=request.top_k,
                use_cot=False,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield f"data: {content}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Document Upload ──

@app.post("/api/documents/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a single document."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate extension
    ext = Path(file.filename).suffix.lower()
    allowed = config.get("security", {}).get("allowed_extensions", [])
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Allowed: {allowed}",
        )

    # Save file
    upload_dir = Path(config["project"]["data_dir"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = upload_dir / safe_name

    content = await file.read()
    file_path.write_bytes(content)

    # Validate size
    max_size = config.get("security", {}).get("max_file_size_mb", 100) * 1024 * 1024
    if len(content) > max_size:
        file_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {len(content) / 1024 / 1024:.1f}MB",
        )

    # Ingest
    pipeline = get_pipeline()
    try:
        chunks_count = pipeline.ingest_file(str(file_path))
        _ingested_docs[file.filename] = {
            "name": file.filename,
            "chunks": chunks_count,
            "size_bytes": len(content),
            "ingested_at": datetime.now().isoformat(),
            "path": str(file_path),
        }
        return IngestResponse(
            status="success",
            chunks_count=chunks_count,
            message=f"Ingested {file.filename} ({chunks_count} chunks)",
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Document Management ──

@app.get("/api/documents")
async def list_documents():
    """List all ingested documents."""
    pipeline = get_pipeline()
    chunks = pipeline.get_chunks()

    # Group by source
    docs: dict[str, dict] = {}
    for c in chunks:
        source = c["metadata"].get("source", "unknown")
        if source not in docs:
            docs[source] = {"name": source, "chunks": 0, "pages": set()}
        docs[source]["chunks"] += 1
        page = c["metadata"].get("page")
        if page:
            docs[source]["pages"].add(page)

    result = []
    for name, info in docs.items():
        entry = {
            "name": name,
            "chunks": info["chunks"],
            "pages": sorted(info["pages"]) if info["pages"] else [],
            **(_ingested_docs.get(name, {})),
        }
        result.append(entry)

    return {"documents": result, "total": len(result)}


@app.delete("/api/documents/{name}")
async def delete_document(name: str):
    """Delete an ingested document by name."""
    pipeline = get_pipeline()
    deleted = pipeline.delete_document(name)
    _ingested_docs.pop(name, None)
    return {"status": "deleted", "name": name, "chunks_removed": deleted}


@app.post("/api/documents/reindex")
async def reindex_all():
    """Re-index all documents from the uploads directory."""
    pipeline = get_pipeline()
    count = pipeline.ingest_directory()
    return {"status": "reindexed", "chunks_count": count}


# ── Conversation History ──

@app.get("/api/conversations")
async def list_conversations():
    """List saved conversations from SQLite."""
    db_path = config.get("conversation", {}).get("db_path", "./data/conversations.db")
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT,
            updated_at TEXT
        )"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )"""
    )
    conn.commit()

    cursor.execute(
        "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
    )
    rows = cursor.fetchall()
    conn.close()

    return {
        "conversations": [
            {
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]
    }


@app.post("/api/conversations")
async def create_conversation(title: str = Query(default="新对话")):
    """Create a new conversation."""
    db_path = config.get("conversation", {}).get("db_path", "./data/conversations.db")
    import sqlite3

    conv_id = uuid.uuid4().hex
    now = datetime.now().isoformat()

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (conv_id, title, now, now),
    )
    conn.commit()
    conn.close()

    return {"id": conv_id, "title": title, "created_at": now}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation and its messages."""
    db_path = config.get("conversation", {}).get("db_path", "./data/conversations.db")
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()

    return {"status": "deleted", "id": conv_id}


# ── Static Files & Root Redirect (must be after all API routes) ──

static_dir = Path(__file__).resolve().parent.parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def root():
    """Serve the competition-grade web UI directly."""
    index_path = static_dir / "index.html"
    return FileResponse(str(index_path))


app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Entry Point ──

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")

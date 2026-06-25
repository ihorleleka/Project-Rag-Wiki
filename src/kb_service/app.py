import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP

from .indexer import KnowledgeIndex
from .settings import Settings


def create_app():
    settings = Settings.load()
    index = KnowledgeIndex(settings)
    watcher_task = None
    startup_task = None
    readiness: dict[str, str] = {"state": "starting", "last_error": ""}
    mcp_runtime: dict[str, bool] = {"running": False}

    mcp = FastMCP("repo-knowledge")
    mcp.settings.streamable_http_path = "/"
    mcp_app = mcp.streamable_http_app()

    async def watcher_loop():
        while True:
            try:
                index.reindex()
            except Exception:
                pass
            await asyncio.sleep(settings.watch_interval_seconds)

    async def startup_reindex_loop():
        try:
            await asyncio.to_thread(index.reindex)
            readiness["state"] = "ready"
            readiness["last_error"] = ""
        except Exception as ex:
            readiness["state"] = "error"
            readiness["last_error"] = str(ex)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal watcher_task, startup_task
        async with mcp.session_manager.run():
            mcp_runtime["running"] = True
            startup_task = asyncio.create_task(startup_reindex_loop())
            try:
                await asyncio.wait_for(asyncio.shield(startup_task), timeout=settings.startup_reindex_timeout_seconds)
            except asyncio.TimeoutError:
                readiness["state"] = "warming"
            if settings.watch_interval_seconds > 0:
                watcher_task = asyncio.create_task(watcher_loop())
            try:
                yield
            finally:
                if startup_task:
                    startup_task.cancel()
                if watcher_task:
                    watcher_task.cancel()
                mcp_runtime["running"] = False

    app = FastAPI(title="Repository Knowledge Service", lifespan=lifespan, redirect_slashes=False)

    @app.get(settings.health_path)
    async def health():
        healthy = readiness["state"] == "ready" and mcp_runtime["running"]
        payload = {
            "status": "ok" if healthy else "degraded",
            "service": readiness["state"],
            "mcp": "running" if mcp_runtime["running"] else "stopped",
            "wiki_root": str(settings.wiki_root),
            "kb_root": str(settings.kb_root),
            "embedding_model": settings.embedding_model,
            "watch_interval_seconds": str(settings.watch_interval_seconds),
        }
        if healthy:
            return payload
        return JSONResponse(
            status_code=503,
            content={
                **payload,
                "last_error": readiness["last_error"],
            },
        )

    @mcp.tool()
    def wiki_search(query: str, top_k: int | None = None):
        """Search the repository wiki for typed context packets and relevant chunks matching a query."""
        results = []
        for r in index.search(query, top_k):
            item = {
                "source_file": r.source_file,
                "chunk_id": r.chunk_id,
                "relevance_score": r.score,
                "context": r.context,
            }
            record_type = getattr(r, "record_type", "chunk")
            item["record_type"] = record_type
            packet = getattr(r, "context_packet", None)
            if packet:
                item["context_packet"] = packet
                item.update(packet)
            metadata = getattr(r, "metadata", None)
            if metadata:
                item["semantic_metadata"] = {
                    key: value
                    for key, value in metadata.items()
                    if key
                    in {
                        "note_id",
                        "kind",
                        "scope",
                        "status",
                        "use_this_when",
                        "rule",
                        "decision",
                        "rationale",
                        "consequences",
                        "summary",
                        "constraints",
                        "anti_patterns",
                        "key_facts",
                        "steps",
                        "terms",
                        "aliases",
                        "evidence",
                        "examples",
                        "retrieval_hints",
                        "raw_prose",
                    }
                }
            results.append(item)
        return results

    @mcp.tool()
    def wiki_read(path: str):
        """Read a wiki document by path and return its full content."""
        return index.read_doc(path)

    @mcp.tool()
    def wiki_list():
        """List all wiki documents currently available in the knowledge base."""
        return index.list_docs()

    @mcp.tool()
    def wiki_schema_report():
        """Report typed note schema health, packet gaps, stale verification, duplicate ids, and broken wiki links."""
        return index.schema_report()

    @mcp.tool()
    def wiki_write(path: str, content: str):
        """Create or replace a wiki document, then refresh the search index."""
        index.write_doc(path, content)
        index.reindex()
        return {"status": "ok", "path": path}

    canonical_mcp_path = settings.mcp_path
    legacy_mcp_path = canonical_mcp_path[:-1] if canonical_mcp_path.endswith("/") else canonical_mcp_path
    app.mount(canonical_mcp_path, mcp_app)
    if legacy_mcp_path and legacy_mcp_path != canonical_mcp_path:
        app.mount(legacy_mcp_path, mcp_app)

    return app

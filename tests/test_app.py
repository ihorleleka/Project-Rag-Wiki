from __future__ import annotations

import asyncio
import importlib
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass
class DummySearchResult:
    source_file: str
    chunk_id: str
    score: float
    context: str


class DummyKnowledgeIndex:
    def __init__(self, settings):
        self.settings = settings
        self.reindex_calls = 0
        self.search_calls: list[tuple[str, int | None]] = []
        self.read_calls: list[str] = []
        self.write_calls: list[tuple[str, str]] = []

    def reindex(self):
        self.reindex_calls += 1
        return {"changed": 1, "removed": 0, "total_files": 1}

    def search(self, query, top_k=None):
        self.search_calls.append((query, top_k))
        return [DummySearchResult("wiki/page.md", "0", 0.91, "result context")]

    def read_doc(self, path):
        self.read_calls.append(path)
        return "document body"

    def list_docs(self):
        return ["wiki/page.md"]

    def write_doc(self, path, content):
        self.write_calls.append((path, content))


class DummyJSONResponse:
    def __init__(self, content=None, status_code=200):
        self.content = content
        self.status_code = status_code


class DummyFastAPI:
    def __init__(self, title, lifespan=None, redirect_slashes=True):
        self.title = title
        self.lifespan = lifespan
        self.redirect_slashes = redirect_slashes
        self.routes: dict[tuple[str, str], object] = {}
        self.mounts: list[tuple[str, object]] = []

    def get(self, path):
        def decorator(func):
            self.routes[("GET", path)] = func
            return func

        return decorator

    def post(self, path):
        def decorator(func):
            self.routes[("POST", path)] = func
            return func

        return decorator

    def mount(self, path, app):
        self.mounts.append((path, app))


class DummySessionManager:
    def run(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyMCP:
    last_instance: "DummyMCP | None" = None

    def __init__(self, name):
        self.name = name
        self.settings = types.SimpleNamespace(streamable_http_path=None)
        self.session_manager = DummySessionManager()
        self.tools: list[tuple[str, object]] = []
        DummyMCP.last_instance = self

    def streamable_http_app(self):
        return "dummy-mcp-app"

    def tool(self):
        def decorator(func):
            self.tools.append((func.__name__, func))
            return func

        return decorator


def install_fakes():
    fastapi_module = types.ModuleType("fastapi")
    fastapi_module.FastAPI = DummyFastAPI

    responses_module = types.ModuleType("fastapi.responses")
    responses_module.JSONResponse = DummyJSONResponse

    mcp_server_module = types.ModuleType("mcp.server")
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = DummyMCP

    indexer_module = types.ModuleType("kb_service.indexer")
    indexer_module.KnowledgeIndex = DummyKnowledgeIndex

    sys.modules["fastapi"] = fastapi_module
    sys.modules["fastapi.responses"] = responses_module
    sys.modules["mcp"] = types.ModuleType("mcp")
    sys.modules["mcp.server"] = mcp_server_module
    sys.modules["mcp.server.fastmcp"] = fastmcp_module
    sys.modules["kb_service.indexer"] = indexer_module


class AppBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        install_fakes()
        sys.modules.pop("kb_service.app", None)
        self.app_module = importlib.import_module("kb_service.app")
        self.app_module = importlib.reload(self.app_module)
        self.original_settings_load = self.app_module.Settings.load

    def tearDown(self) -> None:
        self.app_module.Settings.load = self.original_settings_load

    def test_create_app_exposes_expected_tools(self) -> None:
        app = self.app_module.create_app()
        tool_names = [name for name, _ in DummyMCP.last_instance.tools]

        self.assertEqual(
            tool_names,
            ["wiki_search", "wiki_read", "wiki_list", "wiki_write"],
        )
        self.assertEqual(app.mounts, [("/mcp/", "dummy-mcp-app"), ("/mcp", "dummy-mcp-app")])

    def test_health_reports_ready_when_startup_and_mcp_are_running(self) -> None:
        settings = types.SimpleNamespace(
            wiki_root=Path("wiki"),
            kb_root=Path("kb"),
            host="0.0.0.0",
            port=7331,
            mcp_path="/mcp/",
            health_path="/health",
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=500,
            chunk_overlap=150,
            top_k=8,
            merge_adjacent_window=1,
            staleness_days=90,
            watch_interval_seconds=15,
            startup_reindex_timeout_seconds=3,
        )
        self.app_module.Settings.load = staticmethod(lambda: settings)

        app = self.app_module.create_app()

        async def run_health():
            async with app.lifespan(app):
                health = await app.routes[("GET", "/health")]()
                self.assertEqual(health["status"], "ok")
                self.assertEqual(health["service"], "ready")
                self.assertEqual(health["mcp"], "running")

        asyncio.run(run_health())

    def test_search_tool_returns_serializable_results(self) -> None:
        app = self.app_module.create_app()
        search = DummyMCP.last_instance.tools[0][1]
        result = search("query", 2)

        self.assertEqual(
            result,
            [
                {
                    "source_file": "wiki/page.md",
                    "chunk_id": "0",
                    "relevance_score": 0.91,
                    "context": "result context",
                    "record_type": "chunk",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()

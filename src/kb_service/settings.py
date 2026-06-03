import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    wiki_root: Path
    kb_root: Path
    host: str
    port: int
    mcp_path: str
    health_path: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    merge_adjacent_window: int
    watch_interval_seconds: int
    startup_reindex_timeout_seconds: int

    @staticmethod
    def load() -> "Settings":
        wiki_root = Path(os.getenv("KB_WIKI_ROOT", "./wiki")).resolve()
        kb_root = Path(os.getenv("KB_ROOT", "./wiki/.kb")).resolve()
        mcp_path = os.getenv("KB_MCP_PATH", "/mcp/")
        if not mcp_path.startswith("/"):
            mcp_path = f"/{mcp_path}"
        if not mcp_path.endswith("/"):
            mcp_path = f"{mcp_path}/"

        return Settings(
            wiki_root=wiki_root,
            kb_root=kb_root,
            host=os.getenv("KB_HOST", "0.0.0.0"),
            port=int(os.getenv("KB_PORT", "7331")),
            mcp_path=mcp_path,
            health_path=os.getenv("KB_HEALTH_PATH", "/health"),
            embedding_model=os.getenv("KB_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            chunk_size=int(os.getenv("KB_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("KB_CHUNK_OVERLAP", "150")),
            top_k=int(os.getenv("KB_TOP_K", "8")),
            merge_adjacent_window=max(0, int(os.getenv("KB_MERGE_ADJACENT_WINDOW", "1"))),
            watch_interval_seconds=int(os.getenv("KB_WATCH_INTERVAL_SECONDS", "15")),
            startup_reindex_timeout_seconds=max(1, int(os.getenv("KB_STARTUP_REINDEX_TIMEOUT_SECONDS", "3"))),
        )

from __future__ import annotations

import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kb_service.settings import Settings


@contextmanager
def temporary_env(**updates: str) -> None:
    original: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class SettingsTests(unittest.TestCase):
    def test_load_normalizes_paths_and_applies_defaults(self) -> None:
        with TemporaryDirectory() as tmpdir, temporary_env(
            KB_WIKI_ROOT=str(Path(tmpdir) / "wiki"),
            KB_ROOT=str(Path(tmpdir) / "kb"),
            KB_MCP_PATH="mcp",
            KB_HEALTH_PATH="health",
            KB_PORT="7331",
        ):
            settings = Settings.load()

        self.assertTrue(str(settings.wiki_root).endswith("wiki"))
        self.assertTrue(str(settings.kb_root).endswith("kb"))
        self.assertEqual(settings.mcp_path, "/mcp/")
        self.assertEqual(settings.health_path, "health")
        self.assertEqual(settings.host, "0.0.0.0")
        self.assertEqual(settings.port, 7331)
        self.assertEqual(settings.staleness_days, 90)
        self.assertEqual(settings.startup_reindex_timeout_seconds, 3)


if __name__ == "__main__":
    unittest.main()

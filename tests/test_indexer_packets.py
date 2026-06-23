from __future__ import annotations

import importlib
import sys
import types
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class DummyCollection:
    def add(self, **kwargs):
        return None

    def delete(self, **kwargs):
        return None

    def query(self, **kwargs):
        return {}

    def get(self, **kwargs):
        return {}


class DummyClient:
    def __init__(self, path):
        self.path = path

    def get_or_create_collection(self, *args, **kwargs):
        return DummyCollection()


class DummyProvider:
    def __init__(self, model):
        self.model = model

    def embed(self, texts):
        return [[0.0] for _ in texts]


def install_fakes():
    chromadb_module = types.ModuleType("chromadb")
    chromadb_module.PersistentClient = DummyClient
    sys.modules["chromadb"] = chromadb_module

    frontmatter_module = types.ModuleType("frontmatter")
    frontmatter_module.loads = lambda raw: types.SimpleNamespace(content=raw, metadata={})
    sys.modules["frontmatter"] = frontmatter_module

    embeddings_module = types.ModuleType("kb_service.embeddings")
    embeddings_module.LocalSentenceTransformerProvider = DummyProvider
    sys.modules["kb_service.embeddings"] = embeddings_module


class ContextPacketTests(unittest.TestCase):
    def setUp(self) -> None:
        install_fakes()
        sys.modules.pop("kb_service.indexer", None)
        self.indexer_module = importlib.import_module("kb_service.indexer")

    def test_compile_context_packet_extracts_decision_ready_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            settings = types.SimpleNamespace(
                wiki_root=Path(tmpdir) / "wiki",
                kb_root=Path(tmpdir) / "kb",
                embedding_model="dummy",
                staleness_days=90,
            )
            index = self.indexer_module.KnowledgeIndex(settings)
            body = """# Image Guidance

## Use this when
Image retrieval needs implementation rules.

## Decision
Return compiled context packets before raw chunks.

## Do
- Parse semantic sections.
- Preserve Markdown as the editable format.

## Do not
- Require agents to maintain generated packet files.

## Evidence
- src/kb_service/indexer.py

## Retrieval hints
- MCP image support contract
"""
            packet = index.compile_context_packet(
                "Image.md",
                {
                    "id": "image-guidance",
                    "scope": "project-specific",
                    "last_verified": date.today().isoformat(),
                    "status": "active",
                    "applies_to": ["wiki_search", "indexer"],
                },
                body,
            )

        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.rule, "Return compiled context packets before raw chunks.")
        self.assertEqual(packet.confidence, "high")
        self.assertFalse(packet.needs_verification)
        self.assertEqual(packet.applies_to, ["wiki_search", "indexer"])
        self.assertIn("Parse semantic sections.", packet.do)
        self.assertIn("Require agents to maintain generated packet files.", packet.do_not)
        self.assertEqual(packet.metadata["decision"], "Return compiled context packets before raw chunks.")
        self.assertIn("MCP image support contract", packet.metadata["retrieval_hints"])
        self.assertIn("raw_prose", packet.metadata)
        self.assertNotIn("Raw prose:", packet.index_text)

    def test_compile_context_packet_flags_missing_or_stale_verification(self) -> None:
        with TemporaryDirectory() as tmpdir:
            settings = types.SimpleNamespace(
                wiki_root=Path(tmpdir) / "wiki",
                kb_root=Path(tmpdir) / "kb",
                embedding_model="dummy",
                staleness_days=90,
            )
            index = self.indexer_module.KnowledgeIndex(settings)
            body = """## Decision
Verify stale notes before applying them.

## Do
- Check current code.

## Evidence
- README.md
"""
            packet = index.compile_context_packet(
                "Stale.md",
                {"last_verified": (date.today() - timedelta(days=91)).isoformat()},
                body,
            )

        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.confidence, "medium")
        self.assertTrue(packet.needs_verification)
        self.assertIn("last_verified exceeds staleness threshold", packet.gaps)


if __name__ == "__main__":
    unittest.main()

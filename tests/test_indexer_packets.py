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

## Rationale
Packet results give agents decision-ready context without loading full notes.

## Consequences
Raw chunks remain available as fallback when no packet matches.

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
                    "kind": "decision",
                    "scope": "project-specific",
                    "last_verified": date.today().isoformat(),
                    "status": "active",
                    "applies_to": ["wiki_search", "indexer"],
                },
                body,
            )

        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.kind, "decision")
        self.assertEqual(packet.rule, "Return compiled context packets before raw chunks.")
        self.assertEqual(packet.confidence, "high")
        self.assertFalse(packet.needs_verification)
        self.assertEqual(packet.applies_to, ["wiki_search", "indexer"])
        self.assertEqual(packet.metadata["context_packet"]["decision"], "Return compiled context packets before raw chunks.")
        self.assertEqual(packet.metadata["context_packet"]["rationale"], "Packet results give agents decision-ready context without loading full notes.")
        self.assertIn("Parse semantic sections.", packet.do)
        self.assertIn("Require agents to maintain generated packet files.", packet.do_not)
        self.assertEqual(packet.metadata["decision"], "Return compiled context packets before raw chunks.")
        self.assertIn("MCP image support contract", packet.metadata["retrieval_hints"])
        self.assertIn("raw_prose", packet.metadata)
        self.assertNotIn("Raw prose:", packet.index_text)

    def test_compile_context_packet_supports_reference_notes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            settings = types.SimpleNamespace(
                wiki_root=Path(tmpdir) / "wiki",
                kb_root=Path(tmpdir) / "kb",
                embedding_model="dummy",
                staleness_days=90,
            )
            index = self.indexer_module.KnowledgeIndex(settings)
            body = """# Wiki Vocabulary

## Use this when
Agents need to understand wiki note taxonomy.

## Summary
Reference notes store durable facts that are useful for retrieval but are not rules.

## Key facts
- A reference note can describe concepts, fields, or API shapes.
- It should not force Do or Do not sections.

## Evidence
- README.md

## Retrieval hints
- wiki note kind reference packet
"""
            packet = index.compile_context_packet(
                "Vocabulary.md",
                {
                    "id": "wiki-vocabulary",
                    "kind": "reference",
                    "scope": "general",
                    "last_verified": date.today().isoformat(),
                    "status": "active",
                    "applies_to": ["wiki"],
                },
                body,
            )

        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.kind, "reference")
        self.assertEqual(packet.confidence, "high")
        self.assertEqual(packet.rule, "Reference notes store durable facts that are useful for retrieval but are not rules.")
        self.assertIn("A reference note can describe concepts, fields, or API shapes.", packet.metadata["key_facts"])
        self.assertEqual(packet.gaps, [])

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

    def test_schema_report_flags_legacy_schema_and_link_gaps(self) -> None:
        with TemporaryDirectory() as tmpdir:
            wiki_root = Path(tmpdir) / "wiki"
            wiki_root.mkdir()
            (wiki_root / "Legacy.md").write_text(
                """# Legacy

## Decision
Use typed packets.

## Evidence
- [[Missing]]
""",
                encoding="utf-8",
            )
            settings = types.SimpleNamespace(
                wiki_root=wiki_root,
                kb_root=Path(tmpdir) / "kb",
                embedding_model="dummy",
                staleness_days=90,
            )
            index = self.indexer_module.KnowledgeIndex(settings)
            report = index.schema_report()

        self.assertEqual(report["schema_version"], self.indexer_module.INDEX_SCHEMA_VERSION)
        self.assertEqual(report["total_files"], 1)
        self.assertEqual(report["summary"]["packet_files"], 1)
        self.assertEqual(report["summary"]["files_with_issues"], 1)
        entry = report["files"][0]
        self.assertEqual(entry["source_file"], "Legacy.md")
        self.assertEqual(entry["kind"], "decision")
        self.assertFalse(entry["explicit_kind"])
        self.assertTrue(entry["packet_compiled"])
        self.assertIn("Missing", entry["broken_links"])
        issue_codes = {issue["code"] for issue in entry["issues"]}
        self.assertIn("missing_or_invalid_kind", issue_codes)
        self.assertIn("missing_last_verified", issue_codes)
        self.assertIn("broken_wiki_link", issue_codes)


if __name__ == "__main__":
    unittest.main()

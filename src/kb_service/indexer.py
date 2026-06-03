from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import frontmatter

from .embeddings import LocalSentenceTransformerProvider
from .settings import Settings
from .text_utils import chunks, extract_links, relative_md_paths, sha256_text


@dataclass
class SearchResult:
    source_file: str
    chunk_id: str
    score: float
    context: str


class KnowledgeIndex:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.kb_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.settings.kb_root / "manifest.json"
        self.client = chromadb.PersistentClient(path=str(self.settings.kb_root / "chroma"))
        self.collection = self.client.get_or_create_collection("wiki_chunks", metadata={"hnsw:space": "cosine"})
        self.provider = LocalSentenceTransformerProvider(settings.embedding_model)

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"files": {}, "updated_utc": None}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, default=self._json_default),
            encoding="utf-8",
        )

    @staticmethod
    def _json_default(value: Any) -> str:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        return str(value)

    def reindex(self) -> dict[str, int]:
        manifest = self._read_manifest()
        prev = manifest.get("files", {})
        current: dict[str, Any] = {}
        changed = 0
        removed = 0

        indexed_paths = {str(p).replace('\\', '/') for p in relative_md_paths(self.settings.wiki_root)}

        for prev_path in list(prev.keys()):
            if prev_path not in indexed_paths:
                ids = prev[prev_path].get("chunk_ids", [])
                if ids:
                    self.collection.delete(ids=ids)
                removed += 1

        for rel_path in relative_md_paths(self.settings.wiki_root):
            rel = str(rel_path).replace('\\', '/')
            full = self.settings.wiki_root / rel_path
            raw = full.read_text(encoding="utf-8")
            parsed = frontmatter.loads(raw)
            body = parsed.content
            digest = sha256_text(raw)
            old = prev.get(rel)

            doc_record = {
                "hash": digest,
                "links": extract_links(body),
                "frontmatter": parsed.metadata,
                "chunk_ids": [],
            }

            if old and old.get("hash") == digest:
                current[rel] = old
                continue

            if old and old.get("chunk_ids"):
                self.collection.delete(ids=old["chunk_ids"])

            chunk_texts = chunks(body, self.settings.chunk_size, self.settings.chunk_overlap)
            chunk_ids = [f"{rel}::chunk::{idx}" for idx in range(len(chunk_texts))]
            vectors = self.provider.embed(chunk_texts) if chunk_texts else []
            metadatas = [
                {
                    "source_file": rel,
                    "chunk_id": idx,
                    "content_hash": digest,
                }
                for idx in range(len(chunk_texts))
            ]
            if chunk_ids:
                self.collection.add(ids=chunk_ids, embeddings=vectors, documents=chunk_texts, metadatas=metadatas)

            doc_record["chunk_ids"] = chunk_ids
            current[rel] = doc_record
            changed += 1

        manifest["files"] = current
        self._write_manifest(manifest)
        return {"changed": changed, "removed": removed, "total_files": len(current)}

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        k = top_k if top_k is not None and top_k > 0 else self.settings.top_k
        vector = self.provider.embed([query])[0]
        res = self.collection.query(query_embeddings=[vector], n_results=k, include=["documents", "metadatas", "distances"])

        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]

        raw: list[tuple[str, int, float, str]] = []
        for i, chunk_id in enumerate(ids):
            score = 1.0 - float(distances[i]) if i < len(distances) else 0.0
            meta = metas[i] if i < len(metas) else {}
            # Chroma can return null metadata entries for some rows.
            if not isinstance(meta, dict):
                meta = {}
            source_file = str(meta.get("source_file", ""))
            try:
                chunk_idx = int(meta.get("chunk_id", 0))
            except (TypeError, ValueError):
                chunk_idx = 0
            doc_text = docs[i] if i < len(docs) else ""
            if doc_text is None:
                doc_text = ""
            raw.append(
                (
                    source_file,
                    chunk_idx,
                    score,
                    str(doc_text),
                )
            )

        if self.settings.merge_adjacent_window <= 0:
            return [
                SearchResult(
                    source_file=source_file,
                    chunk_id=str(chunk_idx),
                    score=score,
                    context=context,
                )
                for source_file, chunk_idx, score, context in raw
            ]

        needed_ids: set[str] = set()
        for source_file, chunk_idx, _, _ in raw:
            for idx in range(max(0, chunk_idx - self.settings.merge_adjacent_window), chunk_idx + self.settings.merge_adjacent_window + 1):
                needed_ids.add(f"{source_file}::chunk::{idx}")

        neighbor_docs: dict[str, str] = {}
        if needed_ids:
            get_res = self.collection.get(ids=list(needed_ids), include=["documents"])
            fetched_ids = get_res.get("ids", [])
            fetched_docs = get_res.get("documents", [])
            for i, doc_id in enumerate(fetched_ids):
                if i < len(fetched_docs):
                    neighbor_docs[str(doc_id)] = str(fetched_docs[i])

        out: list[SearchResult] = []
        for source_file, chunk_idx, score, context in raw:
            merged_parts: list[str] = []
            for idx in range(max(0, chunk_idx - self.settings.merge_adjacent_window), chunk_idx + self.settings.merge_adjacent_window + 1):
                doc_id = f"{source_file}::chunk::{idx}"
                text = neighbor_docs.get(doc_id)
                if text:
                    merged_parts.append(text)

            merged_context = "\n...\n".join(merged_parts) if merged_parts else context
            out.append(
                SearchResult(
                    source_file=source_file,
                    chunk_id=str(chunk_idx),
                    score=score,
                    context=merged_context,
                )
            )
        return out

    def list_docs(self) -> list[str]:
        manifest = self._read_manifest()
        return sorted(manifest.get("files", {}).keys())

    def read_doc(self, rel_path: str) -> str:
        target = (self.settings.wiki_root / rel_path).resolve()
        if not str(target).startswith(str(self.settings.wiki_root.resolve())):
            raise ValueError("path must stay within wiki root")
        return target.read_text(encoding="utf-8")

    def write_doc(self, rel_path: str, content: str) -> None:
        target = (self.settings.wiki_root / rel_path).resolve()
        if not str(target).startswith(str(self.settings.wiki_root.resolve())):
            raise ValueError("path must stay within wiki root")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def append_doc(self, rel_path: str, content: str) -> None:
        current = ""
        try:
            current = self.read_doc(rel_path)
        except FileNotFoundError:
            pass
        merged = current + ("\n" if current and not current.endswith("\n") else "") + content
        self.write_doc(rel_path, merged)

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import chromadb
import frontmatter

from .embeddings import LocalSentenceTransformerProvider
from .settings import Settings
from .text_utils import chunks, extract_links, relative_md_paths, sha256_text

SEMANTIC_SECTIONS = {
    "use this when": "use_this_when",
    "rule": "rule",
    "decision": "decision",
    "rationale": "rationale",
    "consequences": "consequences",
    "do": "do",
    "do not": "do_not",
    "summary": "summary",
    "key facts": "key_facts",
    "steps": "steps",
    "terms": "terms",
    "aliases": "aliases",
    "evidence": "evidence",
    "retrieval hints": "retrieval_hints",
}

SECTION_PRIORITY = {
    "packet": 0,
    "decision": 1,
    "do": 2,
    "do_not": 2,
    "evidence": 3,
    "raw": 4,
}

NOTE_KINDS = {"rule", "decision", "reference", "runbook", "glossary"}
NOTE_STATUSES = {"active", "superseded", "deprecated", "pending"}

REQUIRED_SECTIONS = {
    "rule": {
        "use_this_when": "Use this when",
        "rule": "Rule",
        "do": "Do",
        "do_not": "Do not",
        "evidence": "Evidence",
        "retrieval_hints": "Retrieval hints",
    },
    "decision": {
        "use_this_when": "Use this when",
        "decision": "Decision",
        "rationale": "Rationale",
        "consequences": "Consequences",
        "evidence": "Evidence",
        "retrieval_hints": "Retrieval hints",
    },
    "reference": {
        "use_this_when": "Use this when",
        "summary": "Summary",
        "key_facts": "Key facts",
        "evidence": "Evidence",
        "retrieval_hints": "Retrieval hints",
    },
    "runbook": {
        "use_this_when": "Use this when",
        "steps": "Steps",
        "do_not": "Do not",
        "evidence": "Evidence",
        "retrieval_hints": "Retrieval hints",
    },
    "glossary": {
        "terms": "Terms",
        "aliases": "Aliases",
        "retrieval_hints": "Retrieval hints",
    },
}

INDEX_SCHEMA_VERSION = 4
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
PATH_RE = re.compile(r"(?P<path>(?:[A-Za-z]:[\\/])?[\w .@()+={}\[\],'-]+[\\/][\w .@()+={}\[\],'-]+)")


@dataclass
class SearchResult:
    source_file: str
    chunk_id: str
    score: float
    context: str
    record_type: str = "chunk"
    context_packet: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ContextPacket:
    kind: str
    rule: str
    confidence: str
    source: str
    last_verified: str | None
    needs_verification: bool
    applies_to: list[str]
    do: list[str]
    do_not: list[str]
    evidence: list[str]
    gaps: list[str]
    metadata: dict[str, Any]
    index_text: str


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalise_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return text


def _parse_last_verified(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _normalise_kind(value: Any, sections: dict[str, str]) -> tuple[str, bool]:
    raw = str(value or "").strip().lower()
    if raw in NOTE_KINDS:
        return raw, True
    if "decision" in sections:
        return "decision", False
    if "rule" in sections or "do" in sections or "do_not" in sections:
        return "rule", False
    if "steps" in sections:
        return "runbook", False
    if "summary" in sections or "key_facts" in sections:
        return "reference", False
    if "terms" in sections or "aliases" in sections:
        return "glossary", False
    return "reference", False


def _clean_heading(heading: str) -> str:
    return heading.strip().strip("#").strip().rstrip(":").lower()


def parse_semantic_sections(markdown: str) -> dict[str, str]:
    matches = list(HEADING_RE.finditer(markdown))
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        title = _clean_heading(match.group(2))
        key = SEMANTIC_SECTIONS.get(title)
        if not key:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        sections[key] = markdown[start:end].strip()
    return sections


def _list_items(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", stripped).strip()
        if stripped:
            items.append(stripped)
    return items


def _first_sentence(text: str) -> str:
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not compact:
        return ""
    match = re.search(r"(.+?[.!?])(?:\s|$)", compact)
    return match.group(1).strip() if match else compact


def _first_content(*values: str, items: list[str] | None = None) -> str:
    for value in values:
        sentence = _first_sentence(value)
        if sentence:
            return sentence
    if items:
        return items[0]
    return ""


def _as_metadata_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, default=KnowledgeIndex._json_default)


def _missing_wiki_links(links: list[str], indexed_paths: set[str]) -> list[str]:
    indexed_stems = {str(Path(path).with_suffix("")).replace("\\", "/") for path in indexed_paths}
    missing: list[str] = []
    for link in links:
        normalized = link.strip().replace("\\", "/").strip("/")
        if not normalized:
            continue
        candidates = {normalized}
        if not normalized.endswith(".md"):
            candidates.add(f"{normalized}.md")
        candidates.add(str(Path(normalized).with_suffix("")).replace("\\", "/"))
        if not any(candidate in indexed_paths or candidate in indexed_stems for candidate in candidates):
            missing.append(link)
    return missing


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

    def _evidence_changed_after_verification(self, evidence: list[str], last_verified: date | None) -> bool:
        if last_verified is None:
            return False

        roots = [self.settings.wiki_root, self.settings.wiki_root.parent]
        for item in evidence:
            for match in PATH_RE.finditer(item):
                raw_path = match.group("path").strip().strip("`.,;:)")
                candidates: list[Path]
                path = Path(raw_path)
                if path.is_absolute():
                    candidates = [path]
                else:
                    candidates = [root / path for root in roots]
                for candidate in candidates:
                    try:
                        if candidate.is_file() and datetime.fromtimestamp(candidate.stat().st_mtime).date() > last_verified:
                            return True
                    except OSError:
                        continue
        return False

    def compile_context_packet(self, rel: str, metadata: dict[str, Any], body: str) -> ContextPacket | None:
        sections = parse_semantic_sections(body)
        kind, explicit_kind = _normalise_kind(metadata.get("kind"), sections)
        note_rule = sections.get("rule", "").strip()
        decision = sections.get("decision", "").strip()
        rationale = sections.get("rationale", "").strip()
        consequences = sections.get("consequences", "").strip()
        summary = sections.get("summary", "").strip()
        do_items = _list_items(sections.get("do", ""))
        do_not_items = _list_items(sections.get("do_not", ""))
        key_facts = _list_items(sections.get("key_facts", ""))
        steps = _list_items(sections.get("steps", ""))
        terms = _list_items(sections.get("terms", ""))
        aliases = _list_items(sections.get("aliases", ""))
        evidence_items = _list_items(sections.get("evidence", ""))
        retrieval_hints = _list_items(sections.get("retrieval_hints", ""))
        use_this_when = sections.get("use_this_when", "").strip()

        if not any([note_rule, decision, summary, do_items, do_not_items, key_facts, steps, terms, aliases, evidence_items]):
            return None

        last_verified = _normalise_date(metadata.get("last_verified"))
        verified_date = _parse_last_verified(last_verified)
        stale = verified_date is None or verified_date < date.today() - timedelta(days=self.settings.staleness_days)
        evidence_changed = self._evidence_changed_after_verification(evidence_items, verified_date)
        needs_verification = stale or evidence_changed

        gaps: list[str] = []
        if not explicit_kind:
            gaps.append("missing or invalid kind frontmatter")
        for section_key, section_label in REQUIRED_SECTIONS[kind].items():
            if section_key not in sections:
                gaps.append(f"missing {section_label} section")
        if not last_verified:
            gaps.append("missing last_verified")
        elif stale:
            gaps.append("last_verified exceeds staleness threshold")
        if evidence_changed:
            gaps.append("evidence source changed after verification")

        rule = _first_content(
            note_rule,
            decision,
            summary,
            use_this_when,
            rationale,
            consequences,
            items=do_items or key_facts or steps or terms,
        )
        confidence = "medium" if needs_verification or gaps else "high"
        applies_to = _string_list(metadata.get("applies_to"))

        semantic_metadata = {
            "note_id": str(metadata.get("id", "") or ""),
            "kind": kind,
            "scope": str(metadata.get("scope", "") or ""),
            "status": str(metadata.get("status", "") or ""),
            "use_this_when": use_this_when,
            "rule": note_rule,
            "decision": decision,
            "rationale": rationale,
            "consequences": consequences,
            "summary": summary,
            "constraints": do_items,
            "anti_patterns": do_not_items,
            "key_facts": key_facts,
            "steps": steps,
            "terms": terms,
            "aliases": aliases,
            "evidence": evidence_items,
            "examples": [],
            "retrieval_hints": retrieval_hints,
            "raw_prose": body,
        }

        packet = {
            "kind": kind,
            "rule": rule,
            "decision": decision,
            "rationale": rationale,
            "consequences": consequences,
            "confidence": confidence,
            "source": rel,
            "last_verified": last_verified,
            "needs_verification": needs_verification,
            "applies_to": applies_to,
            "do": do_items,
            "do_not": do_not_items,
            "summary": summary,
            "key_facts": key_facts,
            "steps": steps,
            "terms": terms,
            "aliases": aliases,
            "evidence": evidence_items,
            "gaps": gaps,
        }

        index_parts = [
            f"Kind: {kind}",
            f"Use this when: {use_this_when}",
            f"Rule: {note_rule}",
            f"Decision: {decision}",
            f"Rationale: {rationale}",
            f"Consequences: {consequences}",
            f"Summary: {summary}",
            "Do: " + "; ".join(do_items),
            "Do not: " + "; ".join(do_not_items),
            "Key facts: " + "; ".join(key_facts),
            "Steps: " + "; ".join(steps),
            "Terms: " + "; ".join(terms),
            "Aliases: " + "; ".join(aliases),
            "Evidence: " + "; ".join(evidence_items),
            "Retrieval hints: " + "; ".join(retrieval_hints),
            "Applies to: " + "; ".join(applies_to),
        ]

        return ContextPacket(
            kind=kind,
            rule=rule,
            confidence=confidence,
            source=rel,
            last_verified=last_verified,
            needs_verification=needs_verification,
            applies_to=applies_to,
            do=do_items,
            do_not=do_not_items,
            evidence=evidence_items,
            gaps=gaps,
            metadata={**semantic_metadata, "context_packet": packet},
            index_text="\n".join(part for part in index_parts if part.strip()),
        )

    def schema_report(self) -> dict[str, Any]:
        indexed_paths = {str(p).replace("\\", "/") for p in relative_md_paths(self.settings.wiki_root)}
        records: list[dict[str, Any]] = []
        id_counts: dict[str, int] = {}

        for rel_path in relative_md_paths(self.settings.wiki_root):
            rel = str(rel_path).replace("\\", "/")
            raw = (self.settings.wiki_root / rel_path).read_text(encoding="utf-8")
            parsed = frontmatter.loads(raw)
            body = parsed.content
            sections = parse_semantic_sections(body)
            kind, explicit_kind = _normalise_kind(parsed.metadata.get("kind"), sections)
            links = extract_links(body)
            note_id = str(parsed.metadata.get("id", "") or "").strip()
            if note_id:
                id_counts[note_id] = id_counts.get(note_id, 0) + 1
            records.append(
                {
                    "source_file": rel,
                    "metadata": parsed.metadata,
                    "body": body,
                    "sections": sections,
                    "kind": kind,
                    "explicit_kind": explicit_kind,
                    "note_id": note_id,
                    "links": links,
                    "broken_links": _missing_wiki_links(links, indexed_paths),
                }
            )

        files: list[dict[str, Any]] = []
        by_kind: dict[str, int] = {}
        by_status: dict[str, int] = {}
        packet_files = 0
        files_with_issues = 0
        issue_count = 0

        for record in records:
            metadata = record["metadata"]
            sections = record["sections"]
            kind = str(record["kind"])
            packet = self.compile_context_packet(record["source_file"], metadata, record["body"])
            missing_sections = [
                section_label
                for section_key, section_label in REQUIRED_SECTIONS[kind].items()
                if section_key not in sections
            ]
            last_verified = _normalise_date(metadata.get("last_verified"))
            verified_date = _parse_last_verified(last_verified)
            status = str(metadata.get("status", "") or "").strip()
            issues: list[dict[str, str]] = []

            def add_issue(severity: str, code: str, message: str) -> None:
                issues.append({"severity": severity, "code": code, "message": message})

            if not record["note_id"]:
                add_issue("warning", "missing_id", "missing id frontmatter")
            elif id_counts.get(record["note_id"], 0) > 1:
                add_issue("warning", "duplicate_id", f"duplicate id frontmatter: {record['note_id']}")
            if not record["explicit_kind"]:
                add_issue("error", "missing_or_invalid_kind", "missing or invalid kind frontmatter")
            if not status:
                add_issue("warning", "missing_status", "missing status frontmatter")
            elif status not in NOTE_STATUSES:
                add_issue("warning", "invalid_status", f"status should be one of {sorted(NOTE_STATUSES)}")
            if not last_verified:
                add_issue("error", "missing_last_verified", "missing last_verified frontmatter")
            elif verified_date is None:
                add_issue("error", "invalid_last_verified", "last_verified is not an ISO date")
            for section in missing_sections:
                add_issue("error", "missing_required_section", f"missing {section} section")
            if packet is None:
                add_issue("error", "packet_not_compiled", "note does not compile into a context packet")
            elif packet.needs_verification:
                add_issue("warning", "needs_verification", "packet needs verification")
            for link in record["broken_links"]:
                add_issue("warning", "broken_wiki_link", f"wiki link target not found: {link}")

            if packet:
                packet_files += 1
            if issues:
                files_with_issues += 1
                issue_count += len(issues)
            by_kind[kind] = by_kind.get(kind, 0) + 1
            if status:
                by_status[status] = by_status.get(status, 0) + 1

            files.append(
                {
                    "source_file": record["source_file"],
                    "note_id": record["note_id"],
                    "kind": kind,
                    "explicit_kind": record["explicit_kind"],
                    "status": status,
                    "last_verified": last_verified,
                    "packet_compiled": packet is not None,
                    "confidence": packet.confidence if packet else "none",
                    "needs_verification": packet.needs_verification if packet else True,
                    "gaps": packet.gaps if packet else ["packet not compiled"],
                    "missing_sections": missing_sections,
                    "links": record["links"],
                    "broken_links": record["broken_links"],
                    "issues": issues,
                }
            )

        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "wiki_root": str(self.settings.wiki_root),
            "total_files": len(files),
            "summary": {
                "packet_files": packet_files,
                "files_with_issues": files_with_issues,
                "issue_count": issue_count,
                "by_kind": dict(sorted(by_kind.items())),
                "by_status": dict(sorted(by_status.items())),
            },
            "files": files,
        }

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
                "schema_version": INDEX_SCHEMA_VERSION,
                "links": extract_links(body),
                "frontmatter": parsed.metadata,
                "chunk_ids": [],
            }

            if old and old.get("hash") == digest and old.get("schema_version") == INDEX_SCHEMA_VERSION:
                current[rel] = old
                continue

            if old and old.get("chunk_ids"):
                self.collection.delete(ids=old["chunk_ids"])

            packet = self.compile_context_packet(rel, parsed.metadata, body)
            packet_texts = [packet.index_text] if packet else []
            packet_ids = [f"{rel}::packet::0"] if packet else []
            packet_metadatas = []
            if packet:
                packet_metadatas.append(
                    {
                        "source_file": rel,
                        "chunk_id": "packet",
                        "record_type": "packet",
                        "section_rank": SECTION_PRIORITY["packet"],
                        "context_packet": json.dumps(
                            {
                                "kind": packet.kind,
                                "rule": packet.rule,
                                "decision": packet.metadata.get("decision", ""),
                                "rationale": packet.metadata.get("rationale", ""),
                                "consequences": packet.metadata.get("consequences", ""),
                                "confidence": packet.confidence,
                                "source": packet.source,
                                "last_verified": packet.last_verified,
                                "needs_verification": packet.needs_verification,
                                "applies_to": packet.applies_to,
                                "do": packet.do,
                                "do_not": packet.do_not,
                                "summary": packet.metadata.get("summary", ""),
                                "key_facts": packet.metadata.get("key_facts", []),
                                "steps": packet.metadata.get("steps", []),
                                "terms": packet.metadata.get("terms", []),
                                "aliases": packet.metadata.get("aliases", []),
                                "evidence": packet.evidence,
                                "gaps": packet.gaps,
                            },
                            default=self._json_default,
                        ),
                        **{key: _as_metadata_value(value) for key, value in packet.metadata.items() if key != "context_packet"},
                    }
                )

            chunk_texts = chunks(body, self.settings.chunk_size, self.settings.chunk_overlap)
            chunk_ids = [f"{rel}::chunk::{idx}" for idx in range(len(chunk_texts))]
            chunk_metadatas = [
                {
                    "source_file": rel,
                    "chunk_id": idx,
                    "content_hash": digest,
                    "record_type": "chunk",
                    "section_rank": SECTION_PRIORITY["raw"],
                }
                for idx in range(len(chunk_texts))
            ]
            index_ids = packet_ids + chunk_ids
            index_texts = packet_texts + chunk_texts
            index_metadatas = packet_metadatas + chunk_metadatas
            vectors = self.provider.embed(index_texts) if index_texts else []
            if index_ids:
                self.collection.add(ids=index_ids, embeddings=vectors, documents=index_texts, metadatas=index_metadatas)

            doc_record["chunk_ids"] = index_ids
            current[rel] = doc_record
            changed += 1

        manifest["files"] = current
        self._write_manifest(manifest)
        return {"changed": changed, "removed": removed, "total_files": len(current)}

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        k = top_k if top_k is not None and top_k > 0 else self.settings.top_k
        vector = self.provider.embed([query])[0]
        res = self.collection.query(
            query_embeddings=[vector],
            n_results=max(k * 3, k),
            include=["documents", "metadatas", "distances"],
        )

        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]

        raw: list[tuple[str, int, float, str, str, dict[str, Any] | None, dict[str, Any]]] = []
        for i, chunk_id in enumerate(ids):
            score = 1.0 - float(distances[i]) if i < len(distances) else 0.0
            meta = metas[i] if i < len(metas) else {}
            # Chroma can return null metadata entries for some rows.
            if not isinstance(meta, dict):
                meta = {}
            source_file = str(meta.get("source_file", ""))
            record_type = str(meta.get("record_type", "chunk"))
            try:
                chunk_idx = int(meta.get("chunk_id", 0))
            except (TypeError, ValueError):
                chunk_idx = 0
            doc_text = docs[i] if i < len(docs) else ""
            if doc_text is None:
                doc_text = ""
            packet = None
            if record_type == "packet":
                try:
                    packet_raw = meta.get("context_packet")
                    packet = json.loads(str(packet_raw)) if packet_raw else None
                except (TypeError, ValueError, json.JSONDecodeError):
                    packet = None
            raw.append(
                (
                    source_file,
                    chunk_idx,
                    score,
                    str(doc_text),
                    record_type,
                    packet,
                    meta,
                )
            )

        raw.sort(key=lambda item: (SECTION_PRIORITY.get(item[4], SECTION_PRIORITY["raw"]), -item[2]))
        raw = raw[:k]

        if self.settings.merge_adjacent_window <= 0:
            return [
                SearchResult(
                    source_file=source_file,
                    chunk_id=str(chunk_idx),
                    score=score,
                    context=context,
                    record_type=record_type,
                    context_packet=packet,
                    metadata=meta,
                )
                for source_file, chunk_idx, score, context, record_type, packet, meta in raw
            ]

        needed_ids: set[str] = set()
        for source_file, chunk_idx, _, _, record_type, _, _ in raw:
            if record_type != "chunk":
                continue
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
        for source_file, chunk_idx, score, context, record_type, packet, meta in raw:
            if record_type != "chunk":
                out.append(
                    SearchResult(
                        source_file=source_file,
                        chunk_id="packet",
                        score=score,
                        context=packet["rule"] if packet and packet.get("rule") else context,
                        record_type=record_type,
                        context_packet=packet,
                        metadata=meta,
                    )
                )
                continue

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
                    record_type=record_type,
                    context_packet=packet,
                    metadata=meta,
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


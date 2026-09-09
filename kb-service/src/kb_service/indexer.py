from __future__ import annotations

import json
import os
import re
import threading
import logging
import fnmatch
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

import chromadb
import frontmatter

from .embeddings import OnnxMiniLmProvider
from .atomic_io import atomic_write_text
from .contract import INDEX_SCHEMA_VERSION
from .evidence import EvidenceInspector, EvidenceReport, declared_evidence_report
from .lexical import BM25Index, is_identifier_like
from .reranker import load_reranker
from .settings import Settings
from .text_utils import (
    extract_links,
    markdown_chunks,
    merge_contexts_without_overlap,
    relative_md_paths,
    sha256_text,
)

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
    "context": "context",
    "findings": "findings",
    "eliminated approaches": "eliminated_approaches",
    "scope and completeness": "scope_and_completeness",
    "evidence": "evidence",
    "retrieval hints": "retrieval_hints",
    "capability contract": "capability_contract",
    "behavior model": "behavior_model",
    "interaction model": "interaction_model",
    "architecture boundaries": "architecture_boundaries",
    "data and integration contracts": "data_integration_contracts",
    "quality attributes": "quality_attributes",
    "acceptance and verification": "acceptance_verification",
    "reconstruction guidance": "reconstruction_guidance",
    "open questions": "open_questions",
}

CAPABILITY_SECTION_LABELS = {
    "capability_contract": "Capability contract",
    "behavior_model": "Behavior model",
    "interaction_model": "Interaction model",
    "architecture_boundaries": "Architecture boundaries",
    "data_integration_contracts": "Data and integration contracts",
    "quality_attributes": "Quality attributes",
    "acceptance_verification": "Acceptance and verification",
    "reconstruction_guidance": "Reconstruction guidance",
    "open_questions": "Open questions",
}
CAPABILITY_REQUIRED_SECTIONS = {
    "capability_contract": "Capability contract",
    "architecture_boundaries": "Architecture boundaries",
    "acceptance_verification": "Acceptance and verification",
}

SECTION_PRIORITY = {
    "packet": 0,
    "decision": 1,
    "do": 2,
    "do_not": 2,
    "evidence": 3,
    "raw": 4,
}

NOTE_KINDS = {"rule", "decision", "reference", "runbook", "glossary", "investigation"}
NOTE_STATUSES = {"active", "superseded", "deprecated", "pending"}
DEFAULT_NOTE_MAX_LINES = 200
PACKET_EMBEDDING_TOKEN_BUDGET = 240
PACKET_SCORE_BOOST = 0.04
UNVERIFIED_PACKET_PENALTY = 0.03
# Evidence whose anchors disappeared or changed since the note was last verified
# is auto-demoted in ranking so drift self-corrects without a manual audit pass.
EVIDENCE_CHANGED_PENALTY = 0.05
CHANGED_EVIDENCE_STATES = {"missing", "changed_since_verification"}
MAX_RESULTS_PER_SOURCE = 2
INACTIVE_NOTE_STATUSES = {"superseded", "deprecated"}

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
    "investigation": {
        "use_this_when": "Use this when",
        "context": "Context",
        "findings": "Findings",
        "eliminated_approaches": "Eliminated approaches",
        "scope_and_completeness": "Scope and completeness",
        "evidence": "Evidence",
        "retrieval_hints": "Retrieval hints",
    },
}

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

LOGGER = logging.getLogger(__name__)

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
class SearchCandidate:
    source_file: str
    chunk_idx: int
    semantic_score: float
    context: str
    record_type: str
    packet: dict[str, Any] | None
    metadata: dict[str, Any]
    lexical_score: float = 0.0
    dense_rank: int | None = None
    lexical_rank: int | None = None
    fused_score: float = 0.0

    @property
    def trust_delta(self) -> float:
        """Score adjustment that prefers verified owner packets and demotes drift.

        Kept on the 0..1 relevance scale so it applies identically to the dense
        cosine score and to the normalized hybrid fusion score.
        """
        if self.record_type != "packet":
            return 0.0
        delta = PACKET_SCORE_BOOST
        packet = self.packet or {}
        if packet.get("verification_required", packet.get("needs_verification", False)):
            delta -= UNVERIFIED_PACKET_PENALTY
        if str(packet.get("evidence_state", "")) in CHANGED_EVIDENCE_STATES:
            delta -= EVIDENCE_CHANGED_PENALTY
        return delta

    @property
    def gate_score(self) -> float:
        """Relevance used for the ``min_relevance`` admission gate."""
        return max(self.semantic_score, self.lexical_score)

    @property
    def ranking_score(self) -> float:
        return max(0.0, min(1.0, self.gate_score + self.trust_delta))

    @property
    def hybrid_score(self) -> float:
        return max(0.0, min(1.0, self.fused_score + self.trust_delta))

    @property
    def note_status(self) -> str:
        value = self.metadata.get("note_status")
        if not value and self.packet:
            value = self.packet.get("status")
        return str(value or "active").strip().lower()


@dataclass
class ContextPacket:
    kind: str
    rule: str
    schema_health: str
    freshness_state: str
    evidence_state: str
    source: str
    last_verified: str | None
    verification_required: bool
    applies_to: list[str]
    do: list[str]
    do_not: list[str]
    evidence: list[str]
    gaps: list[str]
    metadata: dict[str, Any]
    index_text: str

    @property
    def needs_verification(self) -> bool:
        """Compatibility alias scheduled for removal in a future tool contract."""
        return self.verification_required


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
    if "findings" in sections or "eliminated_approaches" in sections:
        return "investigation", False
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


def _document_title(markdown: str) -> str:
    match = re.search(r"^#\s+(.+?)\s*$", markdown, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _normalise_path_prefix(path_prefix: str | None) -> str:
    if not path_prefix:
        return ""
    normalized = str(path_prefix).replace("\\", "/").strip().strip("/")
    if normalized.endswith(".md"):
        return normalized
    return normalized


def _path_within_scope(source_file: str, scope: str) -> bool:
    if not scope:
        return True
    normalized = str(source_file).replace("\\", "/").strip("/")
    if scope.endswith(".md"):
        return normalized == scope
    return normalized == scope or normalized.startswith(f"{scope}/")


def _slugify_note_title(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(title or "").strip().lower()).strip("-")
    return slug or "capture"


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
        self.provider = OnnxMiniLmProvider(settings.embedding_model)
        # Index mutations are serialized independently from document writes.
        # Embedding a reindex can take seconds; holding this lock while doing so
        # used to make wiki_write/delete/rename wait behind the whole reindex.
        self._write_lock = threading.RLock()
        self._document_lock = threading.RLock()
        self._lexical_lock = threading.RLock()
        self._lexical_cache: dict[str, Any] | None = None
        self.reranker = load_reranker(settings)

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"files": {}, "updated_utc": None}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_text(
            self.manifest_path,
            json.dumps(manifest, indent=2, default=self._json_default) + "\n",
        )

    def index_revision(self) -> str:
        manifest = self._read_manifest()
        canonical_files = json.dumps(
            manifest.get("files", {}),
            sort_keys=True,
            separators=(",", ":"),
            default=self._json_default,
        )
        return sha256_text(canonical_files)

    @staticmethod
    def _json_default(value: Any) -> str:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        return str(value)

    def _build_packet_embedding_document(
        self,
        *,
        rel: str,
        title: str,
        note_id: str,
        use_this_when: str,
        retrieval_hints: list[str],
        aliases: list[str],
        applies_to: list[str],
        primary: str,
        constraints: list[str],
        anti_patterns: list[str],
        key_facts: list[str],
        steps: list[str],
        capability_contract: list[str],
        architecture_boundaries: list[str],
        quality_attributes: list[str],
        acceptance_verification: list[str],
        findings: list[str] | None = None,
        context: str = "",
    ) -> str:
        token_budget = min(PACKET_EMBEDDING_TOKEN_BUDGET, self.provider.max_input_tokens)
        lines: list[str] = []

        def append_line(label: str, value: str, max_value_tokens: int) -> None:
            value = value.strip()
            if not value:
                return
            bounded_value = self.provider.truncate_to_tokens(value, max_value_tokens)
            if not bounded_value:
                return
            candidate = "\n".join([*lines, f"{label}: {bounded_value}"])
            if self.provider.token_count(candidate) <= token_budget:
                lines.append(f"{label}: {bounded_value}")
                return

            low, high = 0, max_value_tokens
            accepted = ""
            while low <= high:
                midpoint = (low + high) // 2
                truncated = self.provider.truncate_to_tokens(value, midpoint)
                candidate = "\n".join([*lines, f"{label}: {truncated}"])
                if truncated and self.provider.token_count(candidate) <= token_budget:
                    accepted = truncated
                    low = midpoint + 1
                else:
                    high = midpoint - 1
            if accepted:
                lines.append(f"{label}: {accepted}")

        # Retrieval identity and routing fields must precede descriptive detail.
        append_line("Title", title, 24)
        append_line("Source", rel, 24)
        append_line("Name", note_id, 16)
        append_line("Use this when", use_this_when, 48)
        append_line("Retrieval hints", "; ".join(retrieval_hints), 48)
        append_line("Aliases", "; ".join(aliases), 32)
        append_line("Applies to", "; ".join(applies_to), 32)
        append_line("Primary contract", primary, 64)
        append_line("Capability contract", "; ".join(capability_contract), 48)
        append_line("Architecture boundaries", "; ".join(architecture_boundaries), 40)
        append_line("Quality attributes", "; ".join(quality_attributes), 32)
        append_line("Verification", "; ".join(acceptance_verification), 32)

        # Only decision-bearing detail competes for the remaining token budget.
        append_line("Constraints", "; ".join(constraints), 48)
        append_line("Anti-patterns", "; ".join(anti_patterns), 40)
        append_line("Key facts", "; ".join(key_facts), 48)
        append_line("Steps", "; ".join(steps), 48)
        append_line("Findings", "; ".join(findings or []), 48)
        append_line("Context", context, 40)

        document = "\n".join(lines)
        if self.provider.token_count(document) > token_budget:
            raise ValueError(
                f"packet embedding document exceeds token budget: {self.provider.token_count(document)} > {token_budget}"
            )
        return document

    def compile_context_packet(
        self,
        rel: str,
        metadata: dict[str, Any],
        body: str,
        evidence_report: EvidenceReport | None = None,
    ) -> ContextPacket | None:
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
        evidence_report = evidence_report or declared_evidence_report(
            evidence_items,
            int(getattr(self.settings, "evidence_max_anchors", 12)),
        )
        retrieval_hints = _list_items(sections.get("retrieval_hints", ""))
        use_this_when = sections.get("use_this_when", "").strip()
        capability_contract = _list_items(sections.get("capability_contract", ""))
        behavior_model = _list_items(sections.get("behavior_model", ""))
        interaction_model = _list_items(sections.get("interaction_model", ""))
        architecture_boundaries = _list_items(sections.get("architecture_boundaries", ""))
        data_integration_contracts = _list_items(sections.get("data_integration_contracts", ""))
        quality_attributes = _list_items(sections.get("quality_attributes", ""))
        acceptance_verification = _list_items(sections.get("acceptance_verification", ""))
        reconstruction_guidance = _list_items(sections.get("reconstruction_guidance", ""))
        open_questions = _list_items(sections.get("open_questions", ""))
        context = sections.get("context", "").strip()
        findings = _list_items(sections.get("findings", ""))
        eliminated_approaches = _list_items(sections.get("eliminated_approaches", ""))
        scope_and_completeness = sections.get("scope_and_completeness", "").strip()

        if not any([note_rule, decision, summary, do_items, do_not_items, key_facts, steps, terms, aliases, evidence_items, findings, context]):
            return None

        last_verified = _normalise_date(metadata.get("last_verified"))
        verified_date = _parse_last_verified(last_verified)
        stale = verified_date is not None and verified_date < date.today() - timedelta(days=self.settings.staleness_days)

        gaps: list[str] = []
        schema_issues: list[str] = []
        if not explicit_kind:
            schema_issues.append("missing or invalid kind frontmatter")
        for section_key, section_label in REQUIRED_SECTIONS[kind].items():
            if section_key not in sections:
                schema_issues.append(f"missing {section_label} section")
        note_id = str(metadata.get("id", "") or "").strip()
        if not note_id:
            schema_issues.append("missing id frontmatter")
        note_status = str(metadata.get("status", "") or "").strip()
        if not note_status:
            schema_issues.append("missing status frontmatter")
        elif note_status not in NOTE_STATUSES:
            schema_issues.append(f"invalid status frontmatter: {note_status}")
        gaps.extend(schema_issues)
        if not last_verified:
            gaps.append("missing last_verified")
        elif stale:
            gaps.append("last_verified exceeds staleness threshold")
        if evidence_report.state == "missing":
            gaps.append("evidence not provided")
            if evidence_report.missing_targets:
                gaps[-1] = "missing evidence anchors: " + ", ".join(evidence_report.missing_targets[:4])
        if evidence_report.state == "changed_since_verification":
            targets = evidence_report.changed_targets or tuple(
                anchor.target
                for anchor in evidence_report.anchors
                if anchor.working_tree_state != "clean"
            )
            gaps.append("evidence changed since verification: " + ", ".join(targets[:4]))
        if evidence_report.excessive_inventory:
            gaps.append(
                f"evidence inventory has {evidence_report.verifiable_count} anchors; "
                f"prefer at most {evidence_report.max_anchors} owner-level anchors"
            )
        evidence_issues: list[str] = []
        if evidence_report.missing_targets:
            sample = ", ".join(evidence_report.missing_targets[:4])
            remainder = len(evidence_report.missing_targets) - 4
            evidence_issues.append(f"missing: {sample}" + (f" (+{remainder} more)" if remainder > 0 else ""))
        if evidence_report.changed_targets:
            sample = ", ".join(evidence_report.changed_targets[:4])
            remainder = len(evidence_report.changed_targets) - 4
            evidence_issues.append(f"changed: {sample}" + (f" (+{remainder} more)" if remainder > 0 else ""))

        schema_health = "complete" if not schema_issues else "incomplete"
        freshness_state = "unknown" if verified_date is None else ("stale" if stale else "current")
        evidence_state = evidence_report.state
        verification_required = (
            schema_health != "complete"
            or freshness_state != "current"
            or evidence_state != "present"
        )

        # Resolved anchor provenance lets agents cite the exact repository file,
        # symbol/locator, and its current trust state instead of re-deriving it.
        evidence_provenance: list[dict[str, str]] = []
        for anchor in evidence_report.anchors[: evidence_report.max_anchors]:
            entry: dict[str, str] = {"target": anchor.target}
            if anchor.locator:
                entry["locator"] = anchor.locator
            if anchor.kind and anchor.kind != "path":
                entry["kind"] = anchor.kind
            entry["state"] = (
                "missing"
                if not anchor.exists
                else ("modified" if anchor.working_tree_state != "clean" else "present")
            )
            evidence_provenance.append(entry)

        rule = _first_content(
            sections.get("capability_contract", ""),
            note_rule,
            decision,
            summary,
            context,
            use_this_when,
            rationale,
            consequences,
            items=do_items or key_facts or steps or findings or terms,
        )
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
            "evidence_summary": evidence_report.summary(),
            "evidence_issues": evidence_issues,
            "examples": [],
            "retrieval_hints": retrieval_hints,
            "capability_contract": capability_contract,
            "behavior_model": behavior_model,
            "interaction_model": interaction_model,
            "architecture_boundaries": architecture_boundaries,
            "data_integration_contracts": data_integration_contracts,
            "quality_attributes": quality_attributes,
            "acceptance_verification": acceptance_verification,
            "reconstruction_guidance": reconstruction_guidance,
            "has_open_questions": bool(open_questions),
            "context": context,
            "findings": findings,
            "eliminated_approaches": eliminated_approaches,
            "scope_and_completeness": scope_and_completeness,
        }

        packet = {
            "kind": kind,
            "rule": rule,
            "decision": decision,
            "rationale": rationale,
            "consequences": consequences,
            "schema_health": schema_health,
            "freshness_state": freshness_state,
            "evidence_state": evidence_state,
            "source": rel,
            "last_verified": last_verified,
            "verification_required": verification_required,
            "needs_verification": verification_required,
            "applies_to": applies_to,
            "do": do_items,
            "do_not": do_not_items,
            "summary": summary,
            "key_facts": key_facts,
            "steps": steps,
            "terms": terms,
            "aliases": aliases,
            "evidence": evidence_items,
            "evidence_summary": evidence_report.summary(),
            "evidence_issues": evidence_issues,
            "evidence_provenance": evidence_provenance,
            "gaps": gaps,
            "status": note_status,
            "capability_contract": capability_contract,
            "behavior_model": behavior_model,
            "interaction_model": interaction_model,
            "architecture_boundaries": architecture_boundaries,
            "data_integration_contracts": data_integration_contracts,
            "quality_attributes": quality_attributes,
            "acceptance_verification": acceptance_verification,
            "reconstruction_guidance": reconstruction_guidance,
            "has_open_questions": bool(open_questions),
            "context": context,
            "findings": findings,
            "eliminated_approaches": eliminated_approaches,
            "scope_and_completeness": scope_and_completeness,
        }

        index_text = self._build_packet_embedding_document(
            rel=rel,
            title=_document_title(body),
            note_id=str(metadata.get("id", "") or ""),
            use_this_when=use_this_when,
            retrieval_hints=retrieval_hints,
            aliases=aliases,
            applies_to=applies_to,
            primary=rule,
            constraints=do_items,
            anti_patterns=do_not_items,
            key_facts=key_facts,
            steps=steps,
            capability_contract=capability_contract,
            architecture_boundaries=architecture_boundaries,
            quality_attributes=quality_attributes,
            acceptance_verification=acceptance_verification,
            findings=findings,
            context=context,
        )

        return ContextPacket(
            kind=kind,
            rule=rule,
            schema_health=schema_health,
            freshness_state=freshness_state,
            evidence_state=evidence_state,
            source=rel,
            last_verified=last_verified,
            verification_required=verification_required,
            applies_to=applies_to,
            do=do_items,
            do_not=do_not_items,
            evidence=evidence_items,
            gaps=gaps,
            metadata={**semantic_metadata, "context_packet": packet},
            index_text=index_text,
        )

    def schema_report(self) -> dict[str, Any]:
        indexed_paths = {str(p).replace("\\", "/") for p in relative_md_paths(self.settings.wiki_root)}
        manifest_files = self._read_manifest().get("files", {})
        records: list[dict[str, Any]] = []
        id_counts: dict[str, int] = {}
        max_note_lines = max(1, int(getattr(self.settings, "note_max_lines", DEFAULT_NOTE_MAX_LINES)))

        evidence_inspector = EvidenceInspector(
            getattr(self.settings, "repository_root", self.settings.wiki_root.parent),
            int(getattr(self.settings, "evidence_max_anchors", 12)),
            getattr(self.settings, "repository_root", self.settings.wiki_root.parent)
            / self.settings.wiki_root.name,
        )
        for rel_path in relative_md_paths(self.settings.wiki_root):
            rel = str(rel_path).replace("\\", "/")
            raw = (self.settings.wiki_root / rel_path).read_text(encoding="utf-8")
            line_count = len(raw.splitlines())
            parsed = frontmatter.loads(raw)
            body = parsed.content
            sections = parse_semantic_sections(body)
            previous_snapshot = manifest_files.get(rel, {}).get("evidence_snapshot", {})
            evidence_report = evidence_inspector.inspect(
                _list_items(sections.get("evidence", "")),
                previous_snapshot,
            )
            kind, explicit_kind = _normalise_kind(parsed.metadata.get("kind"), sections)
            links = extract_links(body)
            note_id = str(parsed.metadata.get("id", "") or "").strip()
            if note_id:
                id_counts[note_id] = id_counts.get(note_id, 0) + 1
            records.append(
                {
                    "source_file": rel,
                    "line_count": line_count,
                    "metadata": parsed.metadata,
                    "body": body,
                    "sections": sections,
                    "kind": kind,
                    "explicit_kind": explicit_kind,
                    "note_id": note_id,
                    "links": links,
                    "broken_links": _missing_wiki_links(links, indexed_paths),
                    "evidence_report": evidence_report,
                }
            )

        files: list[dict[str, Any]] = []
        by_kind: dict[str, int] = {}
        by_status: dict[str, int] = {}
        packet_files = 0
        oversized_files = 0
        files_with_issues = 0
        issue_count = 0

        for record in records:
            metadata = record["metadata"]
            sections = record["sections"]
            kind = str(record["kind"])
            evidence_report = record["evidence_report"]
            packet = self.compile_context_packet(
                record["source_file"], metadata, record["body"], evidence_report
            )
            missing_sections = [
                section_label
                for section_key, section_label in REQUIRED_SECTIONS[kind].items()
                if section_key not in sections
            ]
            capability_specification = any(key in sections for key in CAPABILITY_SECTION_LABELS)
            missing_capability_sections = [
                section_label
                for section_key, section_label in CAPABILITY_REQUIRED_SECTIONS.items()
                if capability_specification and section_key not in sections
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
            if record["line_count"] > max_note_lines:
                add_issue(
                    "warning",
                    "note_too_large",
                    f"note has {record['line_count']} lines; split or condense notes above {max_note_lines} lines",
                )
            for section in missing_sections:
                add_issue("error", "missing_required_section", f"missing {section} section")
            for section in missing_capability_sections:
                add_issue(
                    "warning",
                    "missing_capability_section",
                    f"capability specification is missing {section} section",
                )
            if packet is None:
                add_issue("error", "packet_not_compiled", "note does not compile into a context packet")
            elif packet.verification_required:
                add_issue("warning", "verification_required", "packet requires verification")
            for link in record["broken_links"]:
                add_issue("warning", "broken_wiki_link", f"wiki link target not found: {link}")
            if evidence_report.missing_targets:
                sample = ", ".join(evidence_report.missing_targets[:4])
                remainder = len(evidence_report.missing_targets) - 4
                add_issue(
                    "warning",
                    "missing_evidence_anchors",
                    f"{len(evidence_report.missing_targets)} evidence anchor(s) not found: {sample}"
                    + (f" (+{remainder} more)" if remainder > 0 else ""),
                )
            if evidence_report.changed_targets:
                sample = ", ".join(evidence_report.changed_targets[:4])
                remainder = len(evidence_report.changed_targets) - 4
                add_issue(
                    "warning",
                    "changed_evidence_anchors",
                    f"{len(evidence_report.changed_targets)} evidence anchor(s) changed: {sample}"
                    + (f" (+{remainder} more)" if remainder > 0 else ""),
                )
            if evidence_report.excessive_inventory:
                add_issue(
                    "warning",
                    "evidence_inventory_too_large",
                    f"{evidence_report.verifiable_count} evidence anchors exceed the owner-level budget of {evidence_report.max_anchors}",
                )

            if packet:
                packet_files += 1
            if record["line_count"] > max_note_lines:
                oversized_files += 1
            if issues:
                files_with_issues += 1
                issue_count += len(issues)
            by_kind[kind] = by_kind.get(kind, 0) + 1
            if status:
                by_status[status] = by_status.get(status, 0) + 1

            files.append(
                {
                    "source_file": record["source_file"],
                    "line_count": record["line_count"],
                    "note_id": record["note_id"],
                    "kind": kind,
                    "explicit_kind": record["explicit_kind"],
                    "status": status,
                    "last_verified": last_verified,
                    "packet_compiled": packet is not None,
                    "schema_health": packet.schema_health if packet else "incomplete",
                    "freshness_state": packet.freshness_state if packet else "unknown",
                    "evidence_state": packet.evidence_state if packet else "missing",
                    "verification_required": packet.verification_required if packet else True,
                    "needs_verification": packet.needs_verification if packet else True,
                    "gaps": packet.gaps if packet else ["packet not compiled"],
                    "missing_sections": missing_sections,
                    "capability_specification": capability_specification,
                    "missing_capability_sections": missing_capability_sections,
                    "links": record["links"],
                    "broken_links": record["broken_links"],
                    "evidence_summary": evidence_report.summary(),
                    "evidence_anchors": [anchor.snapshot() for anchor in evidence_report.anchors],
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
                "oversized_files": oversized_files,
                "max_note_lines": max_note_lines,
                "files_with_issues": files_with_issues,
                "issue_count": issue_count,
                "by_kind": dict(sorted(by_kind.items())),
                "by_status": dict(sorted(by_status.items())),
            },
            "files": files,
        }

    def reindex(
        self,
        *,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, int]:
        # Serialize Chroma mutations and manifest writes. Document mutations use
        # a separate lock so an embedding pass cannot make MCP writes wait; the
        # atomic document operations let a concurrent reindex observe either the
        # old or new complete file and a queued targeted pass catches it up.
        with self._write_lock:
            return self._reindex_unlocked(
                cancel_event=cancel_event,
                progress_callback=progress_callback,
            )

    def reindex_paths(
        self,
        rel_paths: set[str],
        *,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, int]:
        normalized = {path.replace("\\", "/") for path in rel_paths}
        with self._write_lock:
            manifest = self._read_manifest()
            files = manifest.get("files", {})
            # Older manifests do not contain enough dependency information for
            # a safe targeted update. A note used as evidence by another note
            # also requires a full pass so its trust state is refreshed.
            if any("evidence_items" not in record for record in files.values()):
                return self._reindex_unlocked(
                    cancel_event=cancel_event,
                    progress_callback=progress_callback,
                )
            repository_root = getattr(
                self.settings,
                "repository_root",
                self.settings.wiki_root.parent,
            ).resolve()
            try:
                wiki_prefix = self.settings.wiki_root.resolve().relative_to(repository_root).as_posix()
            except ValueError:
                wiki_prefix = self.settings.wiki_root.name
            changed_targets = normalized | {
                f"{wiki_prefix}/{path}" for path in normalized
            }
            for owner, record in files.items():
                if owner in normalized:
                    continue
                evidence_snapshot = record.get("evidence_snapshot", {})
                dependency_hit = False
                for anchor in evidence_snapshot.values():
                    target = str(anchor.get("target", "")).replace("\\", "/").rstrip("/")
                    kind = str(anchor.get("kind", "path"))
                    if kind == "glob":
                        dependency_hit = any(fnmatch.fnmatch(path, target) for path in changed_targets)
                    elif kind == "dir":
                        dependency_hit = any(
                            path == target or path.startswith(f"{target}/")
                            for path in changed_targets
                        )
                    else:
                        dependency_hit = target in changed_targets
                    if dependency_hit:
                        break
                if dependency_hit:
                    return self._reindex_unlocked(
                        cancel_event=cancel_event,
                        progress_callback=progress_callback,
                    )
            return self._reindex_unlocked(
                target_paths=normalized,
                cancel_event=cancel_event,
                progress_callback=progress_callback,
            )

    def _reindex_unlocked(
        self,
        *,
        target_paths: set[str] | None = None,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, int]:
        manifest = self._read_manifest()
        prev = manifest.get("files", {})
        current: dict[str, Any] = dict(prev) if target_paths is not None else {}
        changed = 0
        removed = 0
        evidence_inspector = EvidenceInspector(
            getattr(self.settings, "repository_root", self.settings.wiki_root.parent),
            int(getattr(self.settings, "evidence_max_anchors", 12)),
            getattr(self.settings, "repository_root", self.settings.wiki_root.parent)
            / self.settings.wiki_root.name,
        )

        all_paths = [str(path).replace("\\", "/") for path in relative_md_paths(self.settings.wiki_root)]
        indexed_paths = set(all_paths)
        paths_to_visit = all_paths if target_paths is None else sorted(target_paths & indexed_paths)
        paths_to_remove = (
            set(prev) - indexed_paths if target_paths is None else target_paths - indexed_paths
        )
        total_work = len(paths_to_visit) + len(paths_to_remove)
        processed = 0
        if progress_callback:
            progress_callback(processed, total_work)

        for prev_path in sorted(paths_to_remove):
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("indexing cancelled")
            ids = prev.get(prev_path, {}).get("chunk_ids", [])
            if ids:
                self.collection.delete(ids=ids)
            current.pop(prev_path, None)
            removed += 1
            processed += 1
            if progress_callback:
                progress_callback(processed, total_work)

        pending: list[tuple[str, dict[str, Any], list[str], list[str], list[dict[str, Any]]]] = []
        for rel in paths_to_visit:
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("indexing cancelled")
            full = self.settings.wiki_root / PurePosixPath(rel)
            try:
                raw = full.read_text(encoding="utf-8")
            except FileNotFoundError:
                # A concurrent delete/rename may remove a path after the scan.
                # Treat it as removed; the queued targeted request will reconcile
                # any other dependency changes.
                old = prev.get(rel)
                ids = old.get("chunk_ids", []) if old else []
                if ids:
                    self.collection.delete(ids=ids)
                current.pop(rel, None)
                removed += 1
                processed += 1
                if progress_callback:
                    progress_callback(processed, total_work)
                continue
            digest = sha256_text(raw)
            old = prev.get(rel)
            if old and old.get("hash") == digest and old.get("schema_version") == INDEX_SCHEMA_VERSION:
                cached_evidence = old.get("evidence_items")
                if cached_evidence is not None:
                    evidence_report = evidence_inspector.inspect(
                        list(cached_evidence),
                        old.get("evidence_snapshot", {}),
                        verification_updated=False,
                    )
                    if old.get("evidence_snapshot", {}) == evidence_report.snapshot:
                        current[rel] = old
                        processed += 1
                        if progress_callback:
                            progress_callback(processed, total_work)
                        continue

            parsed = frontmatter.loads(raw)
            body = parsed.content
            sections = parse_semantic_sections(body)
            evidence_items = _list_items(sections.get("evidence", ""))
            previous_evidence_snapshot = old.get("evidence_snapshot", {}) if old else {}
            previous_verified = _normalise_date(old.get("frontmatter", {}).get("last_verified")) if old else None
            current_verified = _normalise_date(parsed.metadata.get("last_verified"))
            evidence_report = evidence_inspector.inspect(
                evidence_items,
                previous_evidence_snapshot,
                verification_updated=bool(old and current_verified != previous_verified),
            )

            doc_record = {
                "hash": digest,
                "schema_version": INDEX_SCHEMA_VERSION,
                "links": extract_links(body),
                "frontmatter": parsed.metadata,
                "chunk_ids": [],
                "evidence_snapshot": evidence_report.snapshot,
                "evidence_summary": evidence_report.summary(),
                "evidence_items": evidence_items,
            }

            if old and old.get("chunk_ids"):
                self.collection.delete(ids=old["chunk_ids"])

            packet = self.compile_context_packet(rel, parsed.metadata, body, evidence_report)
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
                        "note_status": str(parsed.metadata.get("status", "") or ""),
                        "context_packet": json.dumps(
                            packet.metadata["context_packet"],
                            default=self._json_default,
                        ),
                    }
                )

            raw_chunks = markdown_chunks(
                body,
                token_budget=min(self.settings.chunk_tokens, self.provider.max_input_tokens),
                token_count=self.provider.token_count,
                split_to_token_windows=self.provider.split_to_token_windows,
            )
            chunk_texts = [chunk.text for chunk in raw_chunks]
            chunk_ids = [f"{rel}::chunk::{idx}" for idx in range(len(chunk_texts))]
            chunk_metadatas = [
                {
                    "source_file": rel,
                    "chunk_id": idx,
                    "content_hash": digest,
                    "record_type": "chunk",
                    "section_rank": SECTION_PRIORITY["raw"],
                    "note_status": str(parsed.metadata.get("status", "") or ""),
                    "heading_path": " > ".join(chunk.heading_path),
                    "section_id": chunk.section_id,
                }
                for idx, chunk in enumerate(raw_chunks)
            ]
            index_ids = packet_ids + chunk_ids
            index_texts = packet_texts + chunk_texts
            index_metadatas = packet_metadatas + chunk_metadatas
            doc_record["chunk_ids"] = index_ids
            current[rel] = doc_record
            pending.append((rel, doc_record, index_ids, index_texts, index_metadatas))
            changed += 1
            processed += 1
            if progress_callback:
                progress_callback(processed, total_work)

        flat_ids: list[str] = []
        flat_texts: list[str] = []
        flat_metadatas: list[dict[str, Any]] = []
        for _, _, ids, texts, metadatas in pending:
            flat_ids.extend(ids)
            flat_texts.extend(texts)
            flat_metadatas.extend(metadatas)
        batch_size = int(getattr(self.settings, "embedding_batch_size", 64))
        for start in range(0, len(flat_texts), batch_size):
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("indexing cancelled")
            texts = flat_texts[start : start + batch_size]
            vectors = self.provider.embed(texts)
            self.collection.add(
                ids=flat_ids[start : start + batch_size],
                embeddings=vectors,
                documents=texts,
                metadatas=flat_metadatas[start : start + batch_size],
            )

        manifest["files"] = current
        self._write_manifest(manifest)
        # The lexical BM25 index is derived from the collection; drop the cache so
        # the next search rebuilds it in lockstep with the vector store.
        with self._lexical_lock:
            self._lexical_cache = None
        return {
            "changed": changed,
            "removed": removed,
            "total_files": len(current),
            "scanned": len(paths_to_visit),
            "mode": "targeted" if target_paths is not None else "full",
        }

    @staticmethod
    def _apply_source_diversity(
        eligible: list[SearchCandidate],
        k: int,
    ) -> list[SearchCandidate]:
        selected: list[SearchCandidate] = []
        selected_ids: set[int] = set()
        source_counts: dict[str, int] = {}

        # First pass preserves source diversity using each source's best result.
        for candidate in eligible:
            if candidate.source_file in source_counts:
                continue
            selected.append(candidate)
            selected_ids.add(id(candidate))
            source_counts[candidate.source_file] = 1
            if len(selected) >= k:
                return selected

        # Second pass adds at most one supporting result from an already selected source.
        for candidate in eligible:
            if id(candidate) in selected_ids:
                continue
            if source_counts.get(candidate.source_file, 0) >= MAX_RESULTS_PER_SOURCE:
                continue
            selected.append(candidate)
            source_counts[candidate.source_file] = source_counts.get(candidate.source_file, 0) + 1
            if len(selected) >= k:
                break
        return selected

    @classmethod
    def _order_ranked_candidates(
        cls,
        candidates: list[SearchCandidate],
        include_inactive: bool = False,
        min_relevance: float = 0.0,
    ) -> list[SearchCandidate]:
        eligible = [
            candidate
            for candidate in candidates
            if (include_inactive or candidate.note_status not in INACTIVE_NOTE_STATUSES)
            and candidate.ranking_score >= min_relevance
        ]
        eligible.sort(
            key=lambda candidate: (
                -candidate.ranking_score,
                -candidate.semantic_score,
                candidate.source_file,
                candidate.record_type,
                candidate.chunk_idx,
            )
        )
        return eligible

    @classmethod
    def _select_ranked_candidates(
        cls,
        candidates: list[SearchCandidate],
        k: int,
        include_inactive: bool = False,
        min_relevance: float = 0.0,
    ) -> list[SearchCandidate]:
        return cls._apply_source_diversity(
            cls._order_ranked_candidates(candidates, include_inactive, min_relevance),
            k,
        )

    @classmethod
    def _order_hybrid_candidates(
        cls,
        candidates: list[SearchCandidate],
        include_inactive: bool = False,
        min_relevance: float = 0.0,
    ) -> list[SearchCandidate]:
        """Order the fused dense+lexical pool by reciprocal-rank fusion.

        Admission still honours ``min_relevance`` on the best per-signal score, so
        a strong keyword match survives even when its dense cosine is weak, while
        low-signal dense noise is dropped exactly as in the dense-only path.
        """
        eligible = [
            candidate
            for candidate in candidates
            if (include_inactive or candidate.note_status not in INACTIVE_NOTE_STATUSES)
            and candidate.gate_score >= min_relevance
        ]
        eligible.sort(
            key=lambda candidate: (
                -candidate.hybrid_score,
                -candidate.fused_score,
                -candidate.gate_score,
                candidate.source_file,
                candidate.record_type,
                candidate.chunk_idx,
            )
        )
        return eligible

    @classmethod
    def _select_hybrid_candidates(
        cls,
        candidates: list[SearchCandidate],
        k: int,
        include_inactive: bool = False,
        min_relevance: float = 0.0,
    ) -> list[SearchCandidate]:
        return cls._apply_source_diversity(
            cls._order_hybrid_candidates(candidates, include_inactive, min_relevance),
            k,
        )

    def _rerank_pool(
        self,
        query: str,
        ordered: list[SearchCandidate],
    ) -> list[SearchCandidate]:
        """Reorder the strongest fused candidates with the cross-encoder.

        Only the top ``reranker_top_n`` are cross-encoded (the expensive step);
        the tail keeps its fusion order. Any failure degrades to the input order.
        """
        reranker = self.reranker
        if reranker is None or len(ordered) < 2:
            return ordered
        top_n = max(1, int(getattr(self.settings, "reranker_top_n", 20)))
        head = ordered[:top_n]
        tail = ordered[top_n:]
        try:
            scores = reranker.score(query, [self._rerank_text(c) for c in head])
        except Exception:
            LOGGER.warning("Cross-encoder rerank failed; using fusion order", exc_info=True)
            return ordered
        if not scores or len(scores) != len(head):
            return ordered
        reordered = [
            candidate
            for _, candidate in sorted(
                zip(scores, head),
                key=lambda pair: pair[0],
                reverse=True,
            )
        ]
        return reordered + tail

    def _ensure_lexical_index(self) -> dict[str, Any]:
        with self._lexical_lock:
            if self._lexical_cache is not None:
                return self._lexical_cache
            try:
                fetched = self.collection.get(include=["documents", "metadatas"])
            except Exception:
                fetched = {}
            ids = fetched.get("ids", []) or []
            docs = fetched.get("documents", []) or []
            metas = fetched.get("metadatas", []) or []
            documents: list[tuple[str, str]] = []
            meta_map: dict[str, dict[str, Any]] = {}
            doc_map: dict[str, str] = {}
            for position, doc_id in enumerate(ids):
                text = docs[position] if position < len(docs) else ""
                meta = metas[position] if position < len(metas) else {}
                if not isinstance(meta, dict):
                    meta = {}
                text = text or ""
                documents.append((doc_id, text))
                meta_map[doc_id] = meta
                doc_map[doc_id] = text
            cache = {
                "bm25": BM25Index.build(documents),
                "meta": meta_map,
                "doc": doc_map,
            }
            self._lexical_cache = cache
            return cache

    def _lexical_search(
        self,
        query: str,
        limit: int,
        scope: str,
    ) -> list[tuple[str, float, list[str], dict[str, Any], str]]:
        cache = self._ensure_lexical_index()
        bm25: BM25Index = cache["bm25"]
        if not len(bm25):
            return []
        hits = bm25.search(query, limit)
        results: list[tuple[str, float, list[str], dict[str, Any], str]] = []
        for doc_id, norm, matched in hits:
            meta = cache["meta"].get(doc_id, {})
            source_file = str(meta.get("source_file", ""))
            if scope and not _path_within_scope(source_file, scope):
                continue
            results.append((doc_id, norm, matched, meta, cache["doc"].get(doc_id, "")))
        return results

    def _candidate_from_metadata(
        self,
        doc_id: str,
        meta: dict[str, Any],
        doc_text: str,
    ) -> SearchCandidate:
        record_type = str(meta.get("record_type", "chunk"))
        try:
            chunk_idx = int(meta.get("chunk_id", 0))
        except (TypeError, ValueError):
            chunk_idx = 0
        packet = None
        if record_type == "packet":
            try:
                packet_raw = meta.get("context_packet")
                packet = json.loads(str(packet_raw)) if packet_raw else None
            except (TypeError, ValueError, json.JSONDecodeError):
                packet = None
        return SearchCandidate(
            source_file=str(meta.get("source_file", "")),
            chunk_idx=chunk_idx,
            semantic_score=0.0,
            context=str(doc_text or ""),
            record_type=record_type,
            packet=packet,
            metadata=meta,
        )

    def _rerank_text(self, candidate: SearchCandidate) -> str:
        if candidate.packet:
            parts = [
                str(candidate.packet.get("source", candidate.source_file)),
                str(candidate.packet.get("rule", "")),
                str(candidate.packet.get("summary", "")),
                str(candidate.packet.get("decision", "")),
                str(candidate.packet.get("context", "")),
            ]
            text = "\n".join(part for part in parts if part).strip()
            if text:
                return text
        return f"{candidate.source_file}\n{candidate.context}".strip()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        include_inactive: bool = False,
        path_prefix: str | None = None,
    ) -> list[SearchResult]:
        requested_k = top_k if top_k is not None and top_k > 0 else self.settings.top_k
        k = min(requested_k, int(getattr(self.settings, "max_top_k", 20)))
        vector = self.provider.embed([query])[0]
        res = self.collection.query(
            query_embeddings=[vector],
            n_results=min(max(k * 3, k), 60),
            include=["documents", "metadatas", "distances"],
        )

        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]

        scope = _normalise_path_prefix(path_prefix)
        candidates_by_id: dict[str, SearchCandidate] = {}
        dense_order: list[str] = []
        for i, chunk_id in enumerate(ids):
            score = 1.0 - float(distances[i]) if i < len(distances) else 0.0
            meta = metas[i] if i < len(metas) else {}
            # Chroma can return null metadata entries for some rows.
            if not isinstance(meta, dict):
                meta = {}
            source_file = str(meta.get("source_file", ""))
            if scope and not _path_within_scope(source_file, scope):
                continue
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
            candidate = SearchCandidate(
                source_file=source_file,
                chunk_idx=chunk_idx,
                semantic_score=score,
                context=str(doc_text),
                record_type=record_type,
                packet=packet,
                metadata=meta,
                dense_rank=len(dense_order),
            )
            candidates_by_id[str(chunk_id)] = candidate
            dense_order.append(str(chunk_id))

        hybrid_enabled = bool(getattr(self.settings, "hybrid_search", True))
        lexical_hits = (
            self._lexical_search(
                query,
                int(getattr(self.settings, "lexical_candidates", 50)),
                scope,
            )
            if hybrid_enabled
            else []
        )

        if not lexical_hits:
            # Dense-only path preserves exact behaviour when the lexical index is
            # empty (e.g. cold corpus) or hybrid retrieval is disabled.
            ordered = self._order_ranked_candidates(
                list(candidates_by_id.values()),
                include_inactive,
                self.settings.min_relevance,
            )
        else:
            lexical_min_score = float(getattr(self.settings, "lexical_min_score", 0.35))
            for lex_rank, (doc_id, norm, matched, meta, doc_text) in enumerate(lexical_hits):
                existing = candidates_by_id.get(doc_id)
                if existing is not None:
                    existing.lexical_rank = lex_rank
                    existing.lexical_score = norm
                    continue
                # A lexical-only hit must clear the score floor and represent a
                # discriminative match — a rare identifier/code/config key, or at
                # least two distinct query terms — so a single shared common word
                # cannot manufacture a match for an otherwise-negative query.
                if norm < lexical_min_score:
                    continue
                if len(matched) < 2 and not any(is_identifier_like(term) for term in matched):
                    continue
                candidate = self._candidate_from_metadata(doc_id, meta, doc_text)
                candidate.lexical_rank = lex_rank
                candidate.lexical_score = norm
                candidates_by_id[doc_id] = candidate

            rrf_k = int(getattr(self.settings, "rrf_k", 60))
            dense_weight = float(getattr(self.settings, "dense_weight", 1.0))
            lexical_weight = float(getattr(self.settings, "lexical_weight", 1.0))
            for candidate in candidates_by_id.values():
                fused = 0.0
                if candidate.dense_rank is not None:
                    fused += dense_weight / (rrf_k + candidate.dense_rank + 1)
                if candidate.lexical_rank is not None:
                    fused += lexical_weight / (rrf_k + candidate.lexical_rank + 1)
                candidate.fused_score = fused
            best_fused = max((c.fused_score for c in candidates_by_id.values()), default=0.0)
            if best_fused > 0.0:
                for candidate in candidates_by_id.values():
                    candidate.fused_score /= best_fused

            ordered = self._order_hybrid_candidates(
                list(candidates_by_id.values()),
                include_inactive,
                self.settings.min_relevance,
            )

        ordered = self._rerank_pool(query, ordered)
        ranked = self._apply_source_diversity(ordered, k)

        if self.settings.merge_adjacent_window <= 0:
            return [
                SearchResult(
                    source_file=candidate.source_file,
                    chunk_id=str(candidate.chunk_idx),
                    score=candidate.ranking_score,
                    context=candidate.context,
                    record_type=candidate.record_type,
                    context_packet=candidate.packet,
                    metadata=candidate.metadata,
                )
                for candidate in ranked
            ]

        needed_ids: set[str] = set()
        for candidate in ranked:
            if candidate.record_type != "chunk":
                continue
            for idx in range(max(0, candidate.chunk_idx - self.settings.merge_adjacent_window), candidate.chunk_idx + self.settings.merge_adjacent_window + 1):
                needed_ids.add(f"{candidate.source_file}::chunk::{idx}")

        neighbor_docs: dict[str, str] = {}
        if needed_ids:
            get_res = self.collection.get(ids=list(needed_ids), include=["documents"])
            fetched_ids = get_res.get("ids", [])
            fetched_docs = get_res.get("documents", [])
            for i, doc_id in enumerate(fetched_ids):
                if i < len(fetched_docs):
                    neighbor_docs[str(doc_id)] = str(fetched_docs[i])

        out: list[SearchResult] = []
        for candidate in ranked:
            if candidate.record_type != "chunk":
                out.append(
                    SearchResult(
                        source_file=candidate.source_file,
                        chunk_id="packet",
                        score=candidate.ranking_score,
                        context=candidate.packet["rule"] if candidate.packet and candidate.packet.get("rule") else candidate.context,
                        record_type=candidate.record_type,
                        context_packet=candidate.packet,
                        metadata=candidate.metadata,
                    )
                )
                continue

            merged_parts: list[str] = []
            for idx in range(max(0, candidate.chunk_idx - self.settings.merge_adjacent_window), candidate.chunk_idx + self.settings.merge_adjacent_window + 1):
                doc_id = f"{candidate.source_file}::chunk::{idx}"
                text = neighbor_docs.get(doc_id)
                if text:
                    merged_parts.append(text)

            merged_context = (
                merge_contexts_without_overlap(merged_parts)
                if merged_parts
                else candidate.context
            )
            out.append(
                SearchResult(
                    source_file=candidate.source_file,
                    chunk_id=str(candidate.chunk_idx),
                    score=candidate.ranking_score,
                    context=merged_context,
                    record_type=candidate.record_type,
                    context_packet=candidate.packet,
                    metadata=candidate.metadata,
                )
            )
        return out

    def list_docs(self) -> list[str]:
        manifest = self._read_manifest()
        return sorted(manifest.get("files", {}).keys())

    def wiki_signature(self) -> dict[str, list[int]]:
        """Cheap filesystem fingerprint of every wiki note (mtime + size).

        The watcher compares successive signatures to reindex only the notes that
        actually changed, instead of re-scanning and re-embedding the whole wiki
        on every interval. Editor-driven writes already trigger targeted reindex,
        so this only has to catch out-of-band edits.
        """
        signature: dict[str, list[int]] = {}
        for rel in relative_md_paths(self.settings.wiki_root):
            full = self.settings.wiki_root / rel
            try:
                stat = full.stat()
            except OSError:
                continue
            signature[str(rel).replace("\\", "/")] = [stat.st_mtime_ns, stat.st_size]
        return signature

    def _note_summary(self, rel: str, record: dict[str, Any]) -> dict[str, Any]:
        frontmatter_meta = record.get("frontmatter", {}) or {}
        raw_kind = str(frontmatter_meta.get("kind", "") or "").strip().lower()
        kind = raw_kind if raw_kind in NOTE_KINDS else None
        status = str(frontmatter_meta.get("status", "") or "").strip() or None
        last_verified = _normalise_date(frontmatter_meta.get("last_verified"))
        verified_date = _parse_last_verified(last_verified)
        if verified_date is None:
            freshness_state = "unknown"
        elif verified_date < date.today() - timedelta(days=self.settings.staleness_days):
            freshness_state = "stale"
        else:
            freshness_state = "current"
        return {
            "path": rel,
            "name": PurePosixPath(rel).name,
            "id": str(frontmatter_meta.get("id", "") or "").strip() or None,
            "kind": kind,
            "status": status,
            "last_verified": last_verified,
            "freshness_state": freshness_state,
            "evidence_summary": record.get("evidence_summary", ""),
        }

    def tree(
        self,
        path_prefix: str | None = None,
        max_depth: int | None = None,
    ) -> dict[str, Any]:
        """Return a navigable directory tree of indexed notes with light metadata.

        Navigation reads cached manifest frontmatter only; it never recompiles
        packets, so it is cheap enough to call before every retrieval decision.
        """
        scope = _normalise_path_prefix(path_prefix)
        manifest_files = self._read_manifest().get("files", {})
        notes = [
            self._note_summary(rel, record)
            for rel, record in sorted(manifest_files.items())
            if _path_within_scope(rel, scope)
        ]

        root: dict[str, Any] = {"type": "dir", "name": scope or "", "children": []}
        dir_index: dict[str, dict[str, Any]] = {"": root}

        def ensure_dir(parts: tuple[str, ...]) -> dict[str, Any]:
            key = ""
            parent = root
            for part in parts:
                key = f"{key}/{part}" if key else part
                node = dir_index.get(key)
                if node is None:
                    node = {"type": "dir", "name": part, "children": []}
                    dir_index[key] = node
                    parent["children"].append(node)
                parent = node
            return parent

        scope_parts = tuple(p for p in scope.split("/") if p) if scope and not scope.endswith(".md") else ()
        for note in notes:
            rel_parts = tuple(p for p in PurePosixPath(note["path"]).parts)
            relative_parts = rel_parts[len(scope_parts):] if rel_parts[: len(scope_parts)] == scope_parts else rel_parts
            dir_parts = relative_parts[:-1]
            if max_depth is not None and len(dir_parts) > max_depth:
                dir_parts = dir_parts[:max_depth]
            parent = ensure_dir(dir_parts)
            parent["children"].append({"type": "note", **note})

        by_kind: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for note in notes:
            key = note["kind"] or "untyped"
            by_kind[key] = by_kind.get(key, 0) + 1
            if note["status"]:
                by_status[note["status"]] = by_status.get(note["status"], 0) + 1

        return {
            "path": scope,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "total_notes": len(notes),
            "summary": {
                "by_kind": dict(sorted(by_kind.items())),
                "by_status": dict(sorted(by_status.items())),
            },
            "tree": root,
        }

    @staticmethod
    def compose_investigation_note(
        *,
        title: str,
        context: str,
        findings: list[str],
        use_this_when: str = "",
        eliminated_approaches: list[str] | None = None,
        scope_and_completeness: str = "",
        evidence: list[str] | None = None,
        retrieval_hints: list[str] | None = None,
        applies_to: list[str] | None = None,
        note_id: str | None = None,
        scope: str = "project-specific",
    ) -> str:
        """Render a well-formed, unverified `investigation` note for capture.

        Captured session memory is deliberately written as `status: pending`
        without `last_verified`, so retrieval and audits surface it as an
        advisory candidate that a later Maintain/Audit pass promotes or drops.
        """

        def bullet_block(items: list[str] | None, empty: str) -> str:
            cleaned = [str(item).strip() for item in (items or []) if str(item).strip()]
            if not cleaned:
                return empty
            return "\n".join(f"- {item}" for item in cleaned)

        resolved_id = _slugify_note_title(note_id or title)
        applies_lines = "".join(
            f"\n  - {str(item).strip()}"
            for item in (applies_to or [])
            if str(item).strip()
        )
        use_this = use_this_when.strip() or f"Retrieving prior findings about {title.strip()}."
        context_text = context.strip() or "Captured from a working session; context not yet consolidated."
        scope_text = (
            scope_and_completeness.strip()
            or "Unverified session capture; scope and completeness not yet confirmed."
        )
        frontmatter_lines = [
            "---",
            f"id: {resolved_id}",
            "kind: investigation",
            f"scope: {scope.strip() or 'project-specific'}",
            "status: pending",
            f"applies_to:{applies_lines}" if applies_lines else "applies_to: []",
            "---",
        ]
        body = [
            f"# {title.strip()}",
            "",
            "## Use this when",
            "",
            use_this,
            "",
            "## Context",
            "",
            context_text,
            "",
            "## Findings",
            "",
            bullet_block(findings, "- Not yet recorded."),
            "",
            "## Eliminated approaches",
            "",
            bullet_block(eliminated_approaches, "- None recorded."),
            "",
            "## Scope and completeness",
            "",
            scope_text,
            "",
            "## Evidence",
            "",
            bullet_block(evidence, "- Not yet corroborated against code."),
            "",
            "## Retrieval hints",
            "",
            bullet_block(retrieval_hints, f"- {title.strip()}"),
            "",
        ]
        return "\n".join(frontmatter_lines) + "\n\n" + "\n".join(body)

    def capture(
        self,
        *,
        title: str,
        context: str,
        findings: list[str],
        use_this_when: str = "",
        eliminated_approaches: list[str] | None = None,
        scope_and_completeness: str = "",
        evidence: list[str] | None = None,
        retrieval_hints: list[str] | None = None,
        applies_to: list[str] | None = None,
        path: str | None = None,
        expected_hash: str | None = None,
    ) -> dict[str, Any]:
        if not str(title or "").strip():
            return {"status": "error", "reason": "title_required"}
        capture_dir = str(getattr(self.settings, "capture_dir", "investigations")).strip("/") or "investigations"
        slug = _slugify_note_title(title)
        rel_path = str(path).replace("\\", "/").strip("/") if path else f"{capture_dir}/{slug}.md"
        content = self.compose_investigation_note(
            title=title,
            context=context,
            findings=findings,
            use_this_when=use_this_when,
            eliminated_approaches=eliminated_approaches,
            scope_and_completeness=scope_and_completeness,
            evidence=evidence,
            retrieval_hints=retrieval_hints,
            applies_to=applies_to,
            note_id=slug,
        )
        result = self.write_doc(rel_path, content, expected_hash)
        if result.get("status") == "ok":
            result["kind"] = "investigation"
            result["note_status"] = "pending"
        return result

    def _resolve_wiki_markdown_path(self, rel_path: str) -> Path:
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise ValueError("path must be a non-empty relative Markdown path")

        portable_path = rel_path.replace("\\", "/")
        posix_path = PurePosixPath(portable_path)
        windows_path = PureWindowsPath(rel_path)
        if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
            raise ValueError("absolute paths are not allowed")
        if any(part in {".", ".."} or part.startswith(".") for part in posix_path.parts):
            raise ValueError("hidden paths and traversal segments are not allowed")
        if posix_path.suffix != ".md":
            raise ValueError("path must target a .md Markdown document")

        wiki_root = self.settings.wiki_root.resolve()
        target = (wiki_root / Path(*posix_path.parts)).resolve()
        try:
            target.relative_to(wiki_root)
        except ValueError as error:
            raise ValueError("path must stay within the canonical wiki root") from error
        return target

    def read_doc(self, rel_path: str) -> dict[str, str]:
        target = self._resolve_wiki_markdown_path(rel_path)
        content = target.read_text(encoding="utf-8")
        return {
            "path": target.relative_to(self.settings.wiki_root.resolve()).as_posix(),
            "content": content,
            "content_hash": sha256_text(content),
        }

    def write_doc(
        self,
        rel_path: str,
        content: str,
        expected_hash: str | None = None,
    ) -> dict[str, Any]:
        target = self._resolve_wiki_markdown_path(rel_path)
        with self._document_lock:
            current_content = target.read_text(encoding="utf-8") if target.exists() else None
            current_hash = sha256_text(current_content) if current_content is not None else None
            if current_content is not None and expected_hash is None:
                return {
                    "status": "conflict",
                    "reason": "expected_hash_required",
                    "path": rel_path.replace("\\", "/"),
                    "current_hash": current_hash,
                }
            if expected_hash is not None and expected_hash != current_hash:
                return {
                    "status": "conflict",
                    "reason": "hash_mismatch",
                    "path": rel_path.replace("\\", "/"),
                    "expected_hash": expected_hash,
                    "current_hash": current_hash,
                }

            atomic_write_text(target, content)
            return {
                "status": "ok",
                "path": target.relative_to(self.settings.wiki_root.resolve()).as_posix(),
                "previous_hash": current_hash,
                "content_hash": sha256_text(content),
            }

    @staticmethod
    def _link_targets_path(link: str, rel_path: str) -> bool:
        # Wikilinks may include a display alias or a heading/block fragment.
        # Neither changes the note path that must be protected from rename/delete.
        normalized = link.split("|", 1)[0].split("#", 1)[0]
        normalized = normalized.strip().replace("\\", "/").strip("/")
        target = rel_path.replace("\\", "/").strip("/")
        return normalized in {target, str(PurePosixPath(target).with_suffix(""))}

    def _inbound_links(self, rel_path: str) -> list[str]:
        inbound: list[str] = []
        canonical = rel_path.replace("\\", "/")
        for candidate in relative_md_paths(self.settings.wiki_root):
            candidate_rel = candidate.as_posix()
            if candidate_rel == canonical:
                continue
            body = (self.settings.wiki_root / candidate).read_text(encoding="utf-8")
            if any(self._link_targets_path(link, canonical) for link in extract_links(body)):
                inbound.append(candidate_rel)
        return inbound

    def delete_doc(self, rel_path: str, expected_hash: str) -> dict[str, Any]:
        target = self._resolve_wiki_markdown_path(rel_path)
        with self._document_lock:
            if not target.exists():
                return {
                    "status": "conflict",
                    "reason": "not_found",
                    "path": rel_path.replace("\\", "/"),
                    "current_hash": None,
                }
            current_content = target.read_text(encoding="utf-8")
            current_hash = sha256_text(current_content)
            if expected_hash != current_hash:
                return {
                    "status": "conflict",
                    "reason": "hash_mismatch",
                    "path": rel_path.replace("\\", "/"),
                    "expected_hash": expected_hash,
                    "current_hash": current_hash,
                }
            inbound_links = self._inbound_links(rel_path)
            if inbound_links:
                return {
                    "status": "conflict",
                    "reason": "inbound_links_exist",
                    "path": rel_path.replace("\\", "/"),
                    "current_hash": current_hash,
                    "inbound_links": inbound_links,
                }
            target.unlink()
            return {
                "status": "ok",
                "path": rel_path.replace("\\", "/"),
                "deleted_hash": current_hash,
            }

    def rename_doc(
        self,
        source_path: str,
        destination_path: str,
        expected_hash: str,
    ) -> dict[str, Any]:
        source = self._resolve_wiki_markdown_path(source_path)
        destination = self._resolve_wiki_markdown_path(destination_path)
        with self._document_lock:
            if not source.exists():
                return {
                    "status": "conflict",
                    "reason": "not_found",
                    "source_path": source_path.replace("\\", "/"),
                    "current_hash": None,
                }
            current_content = source.read_text(encoding="utf-8")
            current_hash = sha256_text(current_content)
            if expected_hash != current_hash:
                return {
                    "status": "conflict",
                    "reason": "hash_mismatch",
                    "source_path": source_path.replace("\\", "/"),
                    "expected_hash": expected_hash,
                    "current_hash": current_hash,
                }
            if destination.exists():
                destination_content = destination.read_text(encoding="utf-8")
                return {
                    "status": "conflict",
                    "reason": "destination_exists",
                    "source_path": source_path.replace("\\", "/"),
                    "destination_path": destination_path.replace("\\", "/"),
                    "destination_hash": sha256_text(destination_content),
                }
            inbound_links = self._inbound_links(source_path)
            if inbound_links:
                return {
                    "status": "conflict",
                    "reason": "inbound_links_exist",
                    "source_path": source_path.replace("\\", "/"),
                    "current_hash": current_hash,
                    "inbound_links": inbound_links,
                }
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            return {
                "status": "ok",
                "source_path": source_path.replace("\\", "/"),
                "destination_path": destination.relative_to(self.settings.wiki_root.resolve()).as_posix(),
                "content_hash": current_hash,
            }


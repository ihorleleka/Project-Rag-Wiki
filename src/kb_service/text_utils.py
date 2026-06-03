from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

WIKILINK_PATTERN = re.compile(r"\[\[([^\]|#]+)")


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def chunks(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("chunk size must be greater than zero")
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    out: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        out.append(text[start:end])
        if end == length:
            break
        start = end - overlap
    return out


def extract_links(markdown: str) -> list[str]:
    seen = set()
    links: list[str] = []
    for m in WIKILINK_PATTERN.finditer(markdown):
        link = m.group(1).strip()
        if link and link not in seen:
            seen.add(link)
            links.append(link)
    return links


def relative_md_paths(root: Path) -> Iterable[Path]:
    for file in sorted(root.rglob("*.md")):
        if file.is_file():
            yield file.relative_to(root)

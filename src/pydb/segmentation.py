from __future__ import annotations

import re

from .config import ChunkingConfig
from .evidence import OffsetMaps, _from_units, _units, normalize_latex, normalize_latex_with_map, segment_source_text

_HEADING_RE = re.compile(r"\\(?:sub)*section\*?\s*\{([^{}]*)\}")
_SKIP_RE = re.compile(r"^\s*%|\\\b(?:begin|end)\s*\{|equation\b", re.I)


def _utf16(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _pages(span: dict) -> list[int]:
    page = span.get("page_range") or {}
    return list(range(page.get("start", 1), page.get("end", page.get("start", 1)) + 1)) if page else []


def verified_spans(original: str, sections: list[dict]) -> tuple[list[dict], str, OffsetMaps]:
    """Segment ``main.tex`` into verified sentence spans with section metadata.

    Returns ``(spans, normalized, maps)``. Spans and normalized offsets use the
    same UTF-16 unit convention as the evidence database, so callers can merge
    spans without recomputing the normalization.
    """
    norm, maps = normalize_latex_with_map(original)
    headings = []
    used = set()
    tex_headings = [{"title": normalize_latex(m.group(1)), "raw": m.group(1), "offset": m.start()} for m in _HEADING_RE.finditer(original)]
    by_heading = {}
    for h in tex_headings:
        candidates = [s for s in sections if re.sub(r"[^\w]", "", (s.get("title") or "").lower()) == re.sub(r"[^\w]", "", h["title"].lower()) and s["id"] not in used]
        if len(candidates) == 1:
            by_heading[h["offset"]] = candidates[0]; used.add(candidates[0]["id"])
    unresolved = [h for h in tex_headings if h["offset"] not in by_heading]
    remaining = [s for s in sections if s["id"] not in used]
    if len(unresolved) == len(remaining):
        for h, sec in zip(unresolved, remaining): by_heading[h["offset"]] = sec
    cursor = 0
    for h in tex_headings:
        sec = by_heading.get(h["offset"])
        if sec:
            prefix = normalize_latex(original[:h["offset"]])
            title = normalize_latex(h["raw"]).strip()
            at = norm.find(title, cursor)
            if at >= 0: cursor = at + len(title)
            headings.append((_utf16(norm[:at]) if at >= 0 else _utf16(prefix), sec))
    spans = []
    for span in segment_source_text(norm):
        if _SKIP_RE.match(span["text"]): continue
        sec = next((s for offset, s in reversed(headings) if offset <= span["start"]), None)
        spans.append({**span, "section": sec["id"] if sec else "front_matter", "title": sec.get("title", sec["id"]) if sec else "Front matter", "pages": _pages(sec) if sec else [], "trust": "verified_span", "keywords": "", "formula": ""})
    return spans, norm, maps


def merge_verified_spans(spans: list[dict], units: str, target_chars: int, max_gap: int) -> list[dict]:
    """Merge adjacent spans in the same section into passages of at most ``target_chars`` UTF-16 units.

    Only same-section neighbours separated by at most ``max_gap`` units merge.
    The merged text is the exact normalized slice, so offsets and evidence
    verification remain valid.
    """
    merged = []
    for span in spans:
        previous = merged[-1] if merged else None
        if (previous is not None and previous["section"] == span["section"]
                and span["start"] - previous["end"] <= max_gap
                and span["end"] - previous["start"] <= target_chars):
            merged[-1] = {**previous, "end": span["end"], "text": _from_units(units[previous["start"]:span["end"]])}
        else:
            merged.append(dict(span))
    return merged


def chunk_verified_spans(spans: list[dict], normalized: str, chunking: ChunkingConfig) -> list[dict]:
    if chunking.mode == "sentence":
        return list(spans)
    return merge_verified_spans(spans, _units(normalized), chunking.target_chars, chunking.max_gap)

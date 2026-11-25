from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

NORMALIZER_VERSION = "latex-v5"
EXTRACTOR_VERSION = "bundle-v4"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OffsetMaps:
    originalToNormalized: list[int]
    normalizedToOriginal: list[int]


def _units(value: str) -> str:
    raw = value.encode("utf-16-le", "surrogatepass")
    return "".join(chr(int.from_bytes(raw[i:i + 2], "little")) for i in range(0, len(raw), 2))


def _from_units(value: str) -> str:
    raw = b"".join(ord(char).to_bytes(2, "little") for char in value)
    return raw.decode("utf-16-le", "surrogatepass")


def normalize_latex_with_map(original: str) -> tuple[str, OffsetMaps]:
    source = _units(original)
    tokens: list[tuple[str, int, int]] = []
    command = re.compile(r"(?<!\\)%[^\n]*|\\(?:begin|end)\s*\{[^}]*\}|\\(?:cite|ref|label)\*?\{([^}]*)\}|\\(?:textbf|textit|emph|mathrm|mathbf)\{([^{}]*)\}|\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\s*|[{}$]|\\\(|\\\)|\\,|\\[%&_#]|.", re.S)
    for match in command.finditer(source):
        raw, at = match.group(0), match.start()
        inner = re.match(r"^\\(?:cite|ref|label)\*?\{([^}]*)\}$", raw) or re.match(r"^\\(?:textbf|textit|emph|mathrm|mathbf)\{([^{}]*)\}$", raw)
        if inner:
            text = inner.group(1)
        elif raw.startswith("%") or re.match(r"^\\(?:begin|end)\s*\{", raw):
            text = " "
        elif re.match(r"^\\[a-zA-Z]", raw) or raw in {"{", "}", "$"}:
            text = " "
        elif raw == r"\(": text = "("
        elif raw == r"\)": text = ")"
        elif raw == r"\,": text = " "
        elif re.match(r"^\\[%&_#]$", raw): text = raw[1]
        else: text = raw
        for char in text:
            tokens.append((char, at, match.end()))
    normalized = "".join(token[0] for token in tokens)
    whitespace = "\\u0009-\\u000d\\u0020\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff"
    whitespace = whitespace.encode("ascii").decode("unicode_escape")
    trim_start = len(normalized) - len(normalized.lstrip(whitespace))
    normalized = re.sub(f"^[{whitespace}]+|[{whitespace}]+$", "", normalized)
    normalized = re.sub("[" + chr(32) + chr(9) + "]+", " ", normalized)
    normalized = re.sub(chr(10) + "{3,}", chr(10) * 2, normalized)
    normalized_to_original = [tokens[min(i + trim_start, len(tokens) - 1)][2] if tokens else len(source) for i in range(len(normalized) + 1)]
    original_to_normalized = []
    for i in range(len(source) + 1):
        lo, hi = 0, len(normalized_to_original) - 1
        while lo < hi:
            mid = (lo + hi) >> 1
            if normalized_to_original[mid] < i: lo = mid + 1
            else: hi = mid
        original_to_normalized.append(lo)
    return _from_units(normalized), OffsetMaps(original_to_normalized, normalized_to_original)


def normalize_latex(original: str) -> str:
    return normalize_latex_with_map(original)[0]


def segment_source_text(normalized: str) -> list[dict]:
    source = _units(normalized)
    spans = []
    lines = re.compile(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*")
    # Allow a period inside a sentence when it is not followed by whitespace
    # (i.e., "i.e.", "e.g.", decimals). Unmatched text between matches is also
    # emitted, so no source text is ever dropped.
    sentences = re.compile(r"(?:[^.!?]|\.(?![.!?]*(?:\s|$)))+[.!?]+(?=\s|$)|[^.!?]+$")
    def emit(base: int, raw: str, start: int, end: int) -> None:
        text = raw[start:end].strip()
        if not text: return
        at = base + start + raw[start:end].index(text)
        spans.append({"start": at, "end": at + len(text), "text": _from_units(text)})
    for block in lines.finditer(source):
        raw, cursor = block.group(0), 0
        for item in sentences.finditer(raw):
            if item.start() > cursor: emit(block.start(), raw, cursor, item.start())
            emit(block.start(), raw, item.start(), item.end())
            cursor = item.end()
        if cursor < len(raw): emit(block.start(), raw, cursor, len(raw))
    return spans

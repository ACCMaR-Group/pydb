from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .config import COARSE_CHUNK_TARGET, DEFAULT_CHUNK_GAP, DEFAULT_CHUNK_TARGET, ChunkingConfig, Config
from .evidence import _units
from .segmentation import merge_verified_spans, verified_spans

ESTIMATOR_VERSION = "sampled-v1"
SAMPLE_PER_STRATUM = 24
STRATUM_COUNT = 4
BOOTSTRAP_ROUNDS = 200
BOOTSTRAP_SEED = 0
FIXED_PEAK_BYTES = 2 * 1024**3
INDEX_OVERHEAD = 1.5
BUDGET_FRACTION = 0.7
CHECKPOINT_BYTES_PER_VECTOR = 50_000
UNLIMITED = 1 << 60

_DEFAULT_CANDIDATES = [
    ("sentence", None, None),
    ("passage", DEFAULT_CHUNK_TARGET, DEFAULT_CHUNK_GAP),
    ("coarse", COARSE_CHUNK_TARGET, DEFAULT_CHUNK_GAP),
    ("custom", 4800, DEFAULT_CHUNK_GAP),
]


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if raw == "max": return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return None if value < 0 or value >= UNLIMITED else value


def _cgroup_headroom(root: Path) -> list[tuple[int, str]]:
    results = []
    limit, usage = _read_int(root / "memory.max"), _read_int(root / "memory.current")
    if limit is not None and usage is not None:
        results.append((max(0, limit - usage), "cgroup-v2"))
    limit, usage = _read_int(root / "memory" / "memory.limit_in_bytes"), _read_int(root / "memory" / "memory.usage_in_bytes")
    if limit is not None and usage is not None:
        results.append((max(0, limit - usage), "cgroup-v1"))
    return results


def detect_available_memory(meminfo: Path | str = "/proc/meminfo", cgroup_root: Path | str = "/sys/fs/cgroup") -> dict:
    """Return the smallest of MemAvailable and visible cgroup headrooms.

    Estimates remain advisory: other processes can consume memory after the
    check, and the index overhead is a model, not a measurement.
    """
    values: list[tuple[int, str]] = []
    try:
        for line in Path(meminfo).read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                values.append((int(line.split()[1]) * 1024, "meminfo"))
                break
    except (OSError, ValueError, IndexError):
        pass
    values.extend(_cgroup_headroom(Path(cgroup_root)))
    if not values:
        return {"availableBytes": None, "sources": []}
    value, source = min(values)
    return {"availableBytes": value, "sources": [name for _, name in values], "smallestSource": source}


def _select_sample(ordered_sizes: list[tuple[str, int]]) -> list[int]:
    n = len(ordered_sizes)
    if n <= 40:
        return list(range(n))
    chosen = set()
    for stratum in range(STRATUM_COUNT):
        lo, hi = (stratum * n) // STRATUM_COUNT, ((stratum + 1) * n) // STRATUM_COUNT
        count = hi - lo
        if count <= 0: continue
        k = min(SAMPLE_PER_STRATUM, count)
        positions = range(count) if k == count else (
            [0] if k == 1 else sorted({round(j * (count - 1) / (k - 1)) for j in range(k)}))
        chosen.update(lo + position for position in positions)
    return sorted(chosen)


def _candidates(chunking: ChunkingConfig) -> list[tuple[str, int | None, int | None]]:
    candidates = list(_DEFAULT_CANDIDATES)
    configured = (chunking.mode, chunking.target_chars, chunking.max_gap)
    if chunking.mode not in {"auto", "sentence"} and configured not in candidates:
        candidates.append(configured)
    return candidates


def _sample_paper(root: Path, pid: str, candidates: list[tuple[str, int | None, int | None]]) -> tuple[int, dict] | None:
    try:
        sections = json.loads((root / pid / "knowledge" / "sections.json").read_text(encoding="utf-8")).get("sections") or []
        original = (root / pid / "main.tex").read_text(encoding="utf-8")
        spans, normalized, _ = verified_spans(original, sections)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    units = _units(normalized)
    counts = {}
    for mode, target, gap in candidates:
        count = len(spans) if mode == "sentence" else len(merge_verified_spans(spans, units, target, gap))
        counts[(mode, target, gap)] = count + 1  # one article summary per paper
    return len(original.encode("utf-8")), counts


def estimate_build(config: Config, root: Path | None = None, ids: list[str] | None = None) -> dict:
    """Estimate embedding-eligible chunks by deterministic stratified sampling.

    Samples papers by main.tex byte rank, runs the real segmentation/merge code,
    and extrapolates within byte strata. Read-only and API-free.
    """
    root = (root or config.corpus_root).resolve()
    if ids is None:
        ids = sorted((p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("paper") and p.name[5:].isdigit()),
                     key=lambda value: int(value[5:]))
    sizes = []
    for pid in ids:
        try: size = (root / pid / "main.tex").stat().st_size
        except OSError: size = 0
        sizes.append((pid, max(1, size)))
    ordered = sorted(range(len(ids)), key=lambda i: (sizes[i][1], sizes[i][0]))
    candidates = _candidates(config.chunking)
    sample_ranks = _select_sample([(ids[i], sizes[i][1]) for i in ordered])
    per_stratum: list[dict] = []
    skipped = []
    for stratum in range(STRATUM_COUNT):
        lo, hi = (stratum * len(ordered)) // STRATUM_COUNT, ((stratum + 1) * len(ordered)) // STRATUM_COUNT
        entries = []
        for rank in sample_ranks:
            if not lo <= rank < hi: continue
            pid = ids[ordered[rank]]
            sampled = _sample_paper(root, pid, candidates)
            if sampled is None:
                skipped.append(pid); continue
            entries.append(sampled)
        per_stratum.append({
            "totalBytes": sum(sizes[ordered[r]][1] for r in range(lo, hi)),
            "samples": entries,
        })
    corpus_bytes = sum(size for _, size in sizes)
    sampled_bytes = sum(size for stratum in per_stratum for size, _ in stratum["samples"])
    estimates = []
    for mode, target, gap in candidates:
        key = (mode, target, gap)
        point, upper = _extrapolate(per_stratum, key)
        estimates.append({"mode": mode, "targetChars": target, "maxGap": gap,
                          "chunks": None if point is None else int(round(point)),
                          "chunksUpper": None if upper is None else int(round(upper))})
    return {
        "estimator": ESTIMATOR_VERSION,
        "corpusPapers": len(ids),
        "sampledPapers": sum(len(stratum["samples"]) for stratum in per_stratum),
        "skippedSamples": skipped,
        "sampledBytes": sampled_bytes,
        "corpusBytes": corpus_bytes,
        "coverage": (sampled_bytes / corpus_bytes) if corpus_bytes else None,
        "candidates": estimates,
        "readOnly": True,
    }


def _extrapolate(per_stratum: list[dict], key: tuple) -> tuple[float | None, float | None]:
    point = 0.0
    usable = 0
    for stratum in per_stratum:
        entries = stratum["samples"]
        size = sum(b for b, _ in entries)
        chunks = sum(c[key] for _, c in entries)
        if size <= 0: continue
        point += stratum["totalBytes"] * (chunks / size)
        usable += 1
    if not usable: return None, None
    rng = random.Random(BOOTSTRAP_SEED)
    totals = []
    for _ in range(BOOTSTRAP_ROUNDS):
        total = 0.0
        for stratum in per_stratum:
            entries = stratum["samples"]
            if not entries: continue
            sampled = [entries[rng.randrange(len(entries))] for _ in entries]
            size = sum(b for b, _ in sampled)
            if size <= 0: continue
            total += stratum["totalBytes"] * (sum(c[key] for _, c in sampled) / size)
        totals.append(total)
    totals.sort()
    upper = totals[min(len(totals) - 1, int(0.95 * len(totals)))] if totals else point
    return point, max(point, upper)


def memory_estimate(chunks: int | None, dimensions: int | None, available: int | None, batch_size: int) -> dict:
    measured = chunks is not None and dimensions is not None and available is not None
    vector_bytes = None if chunks is None or dimensions is None else chunks * dimensions * 4
    peak_bytes = None if vector_bytes is None else int(FIXED_PEAK_BYTES + vector_bytes * INDEX_OVERHEAD)
    budget_bytes = None if available is None else int(available * BUDGET_FRACTION)
    fits = None if not measured else peak_bytes <= budget_bytes
    return {
        "measured": measured,
        "assessedChunks": chunks,
        "dimensions": dimensions,
        "vectorBytes": vector_bytes,
        "peakBytes": peak_bytes,
        "budgetBytes": budget_bytes,
        "fits": fits,
        "checkpointBytes": None if chunks is None else chunks * CHECKPOINT_BYTES_PER_VECTOR,
        "embeddingRequests": None if chunks is None else math.ceil(chunks / batch_size),
        "overheadFactor": INDEX_OVERHEAD,
        "budgetFraction": BUDGET_FRACTION,
        "fixedPeakBytes": FIXED_PEAK_BYTES,
    }


def plan_estimate(config: Config, root: Path | None = None, ids: list[str] | None = None) -> dict:
    estimate = estimate_build(config, root, ids)
    dimensions = config.embedding.expected_dim() if config.embedding else None
    batch_size = config.embedding.batch_size if config.embedding else 64
    available = detect_available_memory()
    evaluated = []
    for candidate in estimate["candidates"]:
        memory = memory_estimate(candidate["chunksUpper"], dimensions, available["availableBytes"], batch_size)
        evaluated.append({**candidate, **memory})
    selected = next((item for item in evaluated
                     if (item["mode"], item["targetChars"], item["maxGap"]) ==
                     (config.chunking.mode, config.chunking.target_chars, config.chunking.max_gap)), None)
    recommendation = None
    reason = None
    if config.embedding is None:
        reason = "embedding is disabled; chunking still controls lexical and citation units"
    elif dimensions is None:
        reason = "embedding.expected_dimensions is required for a memory estimate"
    elif available["availableBytes"] is None:
        reason = "available memory could not be detected"
    else:
        # Prefer finer units when they fit; candidates are ordered fine -> coarse.
        recommendation = next((item for item in evaluated if item["fits"] is True), None)
        reason = ("finest mode whose upper estimate fits the memory budget" if recommendation
                  else "no candidate fits; consider a subset, lexical-only, or a larger machine")
    if config.chunking.mode == "auto":
        # No fitting candidate is still informative: report the coarsest candidate.
        selected_memory = recommendation or (evaluated[-1] if evaluated else None)
    else:
        selected_memory = selected
    return {
        "chunking": {"mode": config.chunking.mode, "targetChars": config.chunking.target_chars,
                     "maxGap": config.chunking.max_gap, "auto": config.chunking.mode == "auto"},
        "memory": {
            "availableBytes": available["availableBytes"],
            "sources": available["sources"],
            "dimensions": dimensions,
            "batchSize": batch_size,
            "estimate": estimate["estimator"],
            "sampledPapers": estimate["sampledPapers"],
            "corpusPapers": estimate["corpusPapers"],
            "coverage": estimate["coverage"],
            "sampledBytes": estimate["sampledBytes"],
            "corpusBytes": estimate["corpusBytes"],
            "skippedSamples": estimate["skippedSamples"],
            "selected": selected_memory,
            "fits": None if selected_memory is None else selected_memory["fits"],
            "reason": reason,
        },
        "candidates": evaluated,
        "recommendation": None if recommendation is None else {
            "mode": recommendation["mode"], "targetChars": recommendation["targetChars"],
            "maxGap": recommendation["maxGap"], "chunksUpper": recommendation["chunksUpper"],
            "peakBytes": recommendation["peakBytes"], "budgetBytes": recommendation["budgetBytes"],
        },
    }


def check_build_fit(config: Config, root: Path | None = None, ids: list[str] | None = None) -> dict:
    """Fail before ingestion when the configured mode is estimated to exceed the budget."""
    if config.embedding is None:
        return {"checked": False, "reason": "embedding is disabled"}
    dimensions = config.embedding.expected_dim()
    if dimensions is None:
        return {"checked": False, "reason": "embedding.expected_dimensions is not set"}
    available = detect_available_memory()["availableBytes"]
    if available is None:
        return {"checked": False, "reason": "available memory could not be detected"}
    estimate = estimate_build(config, root, ids)
    candidate = next((item for item in estimate["candidates"]
                      if (item["mode"], item["targetChars"], item["maxGap"]) ==
                      (config.chunking.mode, config.chunking.target_chars, config.chunking.max_gap)), None)
    if candidate is None or candidate["chunksUpper"] is None:
        return {"checked": False, "reason": "chunk estimate unavailable"}
    memory = memory_estimate(candidate["chunksUpper"], dimensions, available, config.embedding.batch_size)
    if memory["fits"] is False and not config.allow_over_budget:
        raise ValueError(
            f"Estimated vector index does not fit memory: chunks<={candidate['chunksUpper']}, "
            f"dimensions={dimensions}, peak<={memory['peakBytes']} bytes, budget={memory['budgetBytes']} bytes. "
            f"Use a coarser chunking mode, a subset, a larger machine, or set allow_over_budget: true.")
    return {"checked": True, "fits": memory["fits"], "overBudgetAllowed": memory["fits"] is False and config.allow_over_budget, **memory}

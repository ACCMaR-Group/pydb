from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .evidence import sha256


def _utf16_slice(value: str, start: int, end: int) -> str:
    raw = value.encode("utf-16-le")
    return raw[start * 2:end * 2].decode("utf-16-le")


def validate_build(build_path: str | Path, required_queries: list[dict] | None = None) -> dict:
    path = Path(build_path)
    manifest = None
    artifact_dir = path if path.is_dir() else path.parent
    manifest_path = path / "metadata.json" if path.is_dir() else path.parent / "metadata.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = Path(manifest["databasePath"])
        expected = path / declared.name if path.is_dir() else path.parent / declared.name
        if path.is_dir(): path = expected
        elif path.name != declared.name: raise ValueError("Build manifest database path mismatch")
        artifact_dir = path.parent
        if declared.resolve() != path.resolve(): raise ValueError("Build manifest database path mismatch")
        if required_queries is None: required_queries = manifest.get("validation", {}).get("configuredRequiredQueries", [])
    elif path.is_dir():
        candidates = list(path.glob("*.sqlite"))
        if len(candidates) != 1: raise ValueError("Build directory must contain exactly one SQLite database")
        path = candidates[0]
    if path.suffix != ".sqlite" or not path.is_file(): raise ValueError("Build database does not exist")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    errors: list[str] = []
    try:
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok": errors.append("integrity_check_failed")
        foreign = db.execute("PRAGMA foreign_key_check").fetchall()
        if foreign: errors.append("foreign_key_check_failed")
        meta = {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM database_meta")}
        run = db.execute("SELECT id,input_hash,extractor_version,normalizer_version,counts,quarantined FROM build_runs ORDER BY rowid DESC LIMIT 1").fetchone()
        if not run or meta.get("build_id") != run[0]: errors.append("manifest_identity_mismatch")
        if manifest is None:
            manifest = meta.get("build_report")
        if run and (run[2] != "bundle-v4" or run[3] != "latex-v3"): errors.append("version_mismatch")
        checked = 0
        if db.execute("SELECT COUNT(*) FROM papers p LEFT JOIN source_texts s ON s.paper_id=p.id WHERE s.paper_id IS NULL").fetchone()[0]: errors.append("source_text_missing")
        for paper_id, original, normalized, maps, source_hash, normalized_hash in db.execute("SELECT paper_id,original_text,normalized_text,offset_map,source_hash,normalized_hash FROM source_texts"):
            if sha256(original) != source_hash or sha256(normalized) != normalized_hash: errors.append("source_hash_mismatch")
            parsed = json.loads(maps)
            if not isinstance(parsed.get("originalToNormalized"), list) or len(parsed["originalToNormalized"]) != len(original.encode("utf-16-le")) // 2 + 1 or not isinstance(parsed.get("normalizedToOriginal"), list) or len(parsed["normalizedToOriginal"]) != len(normalized.encode("utf-16-le")) // 2 + 1: errors.append("offset_map_invalid")
            for start, end, quote, chunk_hash, trust in db.execute("SELECT start_offset,end_offset,text,source_hash,trust_level FROM chunks WHERE paper_id=?", (paper_id,)):
                if trust == "verified_span":
                    if start is None or end is None or chunk_hash != normalized_hash or not (0 <= start < end <= len(normalized.encode("utf-16-le")) // 2): errors.append("verified_span_invalid")
                    elif _utf16_slice(normalized, start, end) != quote: errors.append("span_mismatch")
                    else: checked += 1
        evidence_mismatch = db.execute("SELECT COUNT(*) FROM evidence e LEFT JOIN chunks c ON c.id=e.chunk_id WHERE c.id IS NULL OR e.paper_id!=c.paper_id OR e.trust_level!=c.trust_level OR e.source_hash!=c.source_hash OR (e.start_offset IS NOT c.start_offset) OR (e.end_offset IS NOT c.end_offset) OR (e.quote IS NOT c.text AND e.trust_level='verified_span')").fetchone()[0]
        missing_evidence = db.execute("SELECT COUNT(*) FROM chunks c LEFT JOIN evidence e ON e.chunk_id=c.id WHERE e.id IS NULL").fetchone()[0]
        if evidence_mismatch or missing_evidence: errors.append("evidence_chunk_mismatch")
        counts = {"papers": db.execute("SELECT COUNT(*) FROM papers").fetchone()[0], "chunks": db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], "evidence": db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0], "fts": db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]}
        fts_mismatch = db.execute("SELECT COUNT(*) FROM chunks c LEFT JOIN chunks_fts f ON f.chunk_id=c.id WHERE f.chunk_id IS NULL OR f.title IS NOT c.title OR f.text IS NOT c.text OR f.keywords IS NOT c.keywords OR f.formula_terms IS NOT c.formula_terms").fetchone()[0]
        if fts_mismatch: errors.append("fts_alignment_mismatch")
        if db.execute("SELECT COUNT(*) FROM chunks WHERE trust_level NOT IN ('verified_span','extracted_text','generated_summary')").fetchone()[0]: errors.append("unknown_trust_level")
        if run:
            saved_counts = json.loads(run[4])
            if any(counts[key] != saved_counts.get(key) for key in ("papers", "chunks")) or counts["chunks"] != counts["fts"] or counts["chunks"] != counts["evidence"]: errors.append("row_count_mismatch")
            if manifest and (manifest.get("quarantined") != json.loads(run[5]) or manifest.get("counts") != saved_counts): errors.append("quarantine_manifest_mismatch")
            db_vectors = meta.get("vector_search", {"enabled": False})
            if manifest and manifest.get("vectorSearch", {"enabled": False}) != db_vectors: errors.append("vector_manifest_mismatch")
        queries = []
        for gate in required_queries or []:
            query, wanted = gate["query"], gate["paper_id"]
            try:
                rows = db.execute("SELECT p.metadata,c.section,c.trust_level FROM chunks_fts f JOIN chunks c ON c.id=f.chunk_id JOIN papers p ON p.id=c.paper_id WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT 100", (query,)).fetchall()
                hits = [(section, trust) for md, section, trust in rows if json.loads(md).get("paperId") == wanted and (gate.get("section") is None or section == gate.get("section"))]
                matched = bool(hits)
                trust_levels = sorted({trust for _, trust in hits})
            except sqlite3.Error:
                matched = False
                trust_levels = []
            queries.append({"query": query, "paper_id": wanted, "section": gate.get("section"), "passed": matched, "matchedTrustLevels": trust_levels, "kind": "lexical_fts_smoke"})
        if any(not q["passed"] for q in queries): errors.append("required_query_failed")
        if manifest:
            if not manifest.get("complete") or manifest.get("buildId") != (run[0] if run else None) or manifest.get("inputHash") != (run[1] if run else None): errors.append("manifest_identity_mismatch")
            vector_meta = manifest.get("vectorSearch", {})
            if vector_meta.get("embeddingEnabled"):
                from .vector import validate_vectors
                checked_vectors = validate_vectors(path, artifact_dir, vector_meta)
                if not checked_vectors["valid"]: errors.extend(checked_vectors["errors"])
        return {"valid": not errors, "buildId": run[0] if run else None, "inputHash": run[1] if run else None, "integrity": integrity, "foreignKeyViolations": len(foreign), "verifiedSpans": checked, "requiredQueries": queries, "errors": errors}
    finally: db.close()

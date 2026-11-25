from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path

from .config import Config
from .estimation import check_build_fit
from .evidence import EXTRACTOR_VERSION, NORMALIZER_VERSION, sha256
from .segmentation import _pages, chunk_verified_spans, verified_spans
from .validation import validate_build
from .vector import validate_vectors

BUNDLE_FILES = ("manifest.json", "document.json", "sections.json", "quality.json", "retrieval.json", "inventory.json", "coherence.json")
SCHEMA = '''PRAGMA foreign_keys=ON;
CREATE TABLE database_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE papers(id TEXT PRIMARY KEY,build_id TEXT,document_id TEXT,doi TEXT,title TEXT,article_type TEXT,metadata TEXT NOT NULL);
CREATE TABLE sections(id TEXT PRIMARY KEY,paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,title TEXT,pages TEXT,content TEXT);
CREATE TABLE source_texts(paper_id TEXT PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,original_text TEXT NOT NULL,normalized_text TEXT NOT NULL,offset_map TEXT NOT NULL,source_hash TEXT NOT NULL,normalized_hash TEXT NOT NULL);
CREATE TABLE chunks(id TEXT PRIMARY KEY,paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,section TEXT,title TEXT,text TEXT NOT NULL,keywords TEXT,formula_terms TEXT,pages TEXT,trust_level TEXT NOT NULL,source_hash TEXT NOT NULL,start_offset INTEGER,end_offset INTEGER,quality_flags TEXT NOT NULL DEFAULT '[]');
CREATE TABLE evidence(id TEXT PRIMARY KEY,chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,section TEXT,pages TEXT,trust_level TEXT NOT NULL,source_hash TEXT NOT NULL,start_offset INTEGER,end_offset INTEGER,quote TEXT);
CREATE TABLE quality_signals(paper_id TEXT PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,flags TEXT NOT NULL,score REAL NOT NULL DEFAULT 0);
CREATE TABLE build_runs(id TEXT PRIMARY KEY,input_hash TEXT,extractor_version TEXT,normalizer_version TEXT,started_at TEXT,elapsed_ms INTEGER,counts TEXT,quarantined TEXT);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_evidence_chunk ON evidence(chunk_id);
CREATE INDEX IF NOT EXISTS idx_evidence_paper ON evidence(paper_id);
CREATE INDEX IF NOT EXISTS idx_sections_paper ON sections(paper_id);
CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED,title,text,keywords,formula_terms);'''

INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_chunk ON evidence(chunk_id)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_paper ON evidence(paper_id)",
    "CREATE INDEX IF NOT EXISTS idx_sections_paper ON sections(paper_id)",
)


def _ensure_indexes(db_path) -> None:
    """Add lookup indexes to resumed databases built before they were part of the schema."""
    with sqlite3.connect(db_path) as db:
        for statement in INDEX_STATEMENTS:
            db.execute(statement)


def _identity(config: Config, ids: list[str]) -> tuple[str, str]:
    rows = []
    for pid in ids:
        root = config.corpus_root / pid
        for name in (*BUNDLE_FILES, "../main.tex"):
            file = root / "knowledge" / name if name != "../main.tex" else root / "main.tex"
            try:
                resolved = file.resolve(strict=True)
                if not resolved.is_relative_to(root.resolve()):
                    raise ValueError("Bundle file escapes paper directory")
                digest = sha256(resolved.read_text(encoding="utf-8"))
            except FileNotFoundError:
                if name in {"coherence.json", "../main.tex"}: digest = "OPTIONAL_MISSING"
                else: digest = "MISSING"
            rows.append(f"{pid}/{name}:{digest}")
    semantics = {**config.semantic_dict(), "required_queries": config.required_queries or []}
    ih = hashlib.sha256(("\n".join(sorted(rows)) + json.dumps(semantics, sort_keys=True)).encode()).hexdigest()
    bid = "build-" + hashlib.sha256(f"{ih}\n{EXTRACTOR_VERSION}\n{NORMALIZER_VERSION}".encode()).hexdigest()[:12]
    return bid, ih


def _resumable_state(db_path, build_id, input_hash):
    """Return (counts, quarantined) for a completed ingestion of this exact build.

    Ingestion writes its build_runs row only after every selected paper has been
    processed, so a matching row means the database can be reused without
    re-ingesting. Identity covers corpus digests, extractor/normalizer versions
    and semantic recipe settings, so partial or stale databases never match.
    """
    try:
        with sqlite3.connect(db_path) as db:
            row = db.execute("SELECT input_hash,counts,quarantined FROM build_runs WHERE id=? ORDER BY rowid DESC LIMIT 1", (build_id,)).fetchone()
            meta = db.execute("SELECT value FROM database_meta WHERE key='build_id'").fetchone()
        if not row or row[0] != input_hash or not meta or json.loads(meta[0]) != build_id: return None
        counts, quarantined = json.loads(row[1]), json.loads(row[2])
        if not isinstance(counts, dict) or counts.get("papers", 0) <= 0 or not isinstance(quarantined, list): return None
        return counts, quarantined
    except (sqlite3.Error, ValueError, TypeError):
        return None


def _finalize(config, db_path, work, build_id, input_hash, counts, quarantined, out, final, started_at, memory_fit=None):
    if counts.get("papers", 0) <= 0 or len(quarantined) > config.max_quarantined_papers: raise ValueError("Build failed quarantine/completeness threshold")
    _ensure_indexes(db_path)
    vector_meta = {}
    if config.embedding:
        from .vector import build_vectors
        vector_meta = build_vectors(config, db_path, work, build_id)
    gate = validate_build(db_path, config.required_queries)
    if not gate["valid"]: raise ValueError("Built database validation failed")
    vector_gate = validate_vectors(db_path, work, vector_meta) if vector_meta else {"valid": True, "errors": []}
    if not vector_gate["valid"]: raise ValueError("Built vector artifact validation failed")
    target = work / f"{config.alias}.sqlite"
    if vector_meta.get("embeddingEnabled") or vector_meta.get("enabled"):
        db_vector_meta = {**vector_meta, "indexPath": "chroma"} if vector_meta.get("enabled") else vector_meta
        with sqlite3.connect(target) as db:
            db.execute("INSERT OR REPLACE INTO database_meta VALUES (?,?)", ("vector_search", json.dumps(db_vector_meta)))
    report = {"complete": True, "buildId": build_id, "inputHash": input_hash, "databasePath": str(out / build_id / target.name), "alias": config.alias, "startedAt": started_at, "counts": counts, "quarantined": quarantined, "runtime": {"recipePath": str(config.recipe_path), "pythonExecutable": config.python_executable}, "chunking": config.semantic_dict()["chunking"], "memoryEstimate": memory_fit, "vectorSearch": vector_meta or {"enabled": False}, "validation": {"valid": gate["valid"] and vector_gate["valid"], "requiredQueries": gate["requiredQueries"], "configuredRequiredQueries": config.required_queries or [], "errors": gate["errors"] + vector_gate["errors"]}}
    with sqlite3.connect(target) as db:
        db.execute("INSERT OR REPLACE INTO database_meta VALUES (?,?)", ("build_report", json.dumps(report)))
    (work / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(work, final)
    shutil.rmtree(work, ignore_errors=True)
    return report


def build(config: Config) -> dict:
    root = config.corpus_root.resolve(strict=True)
    if not root.is_dir(): raise ValueError("Corpus root must be a directory")
    ids = config.paper_ids or sorted((p.name for p in root.iterdir() if re.fullmatch(r"paper\d+", p.name)), key=lambda x: int(x[5:]))
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"paper\d+", x) for x in ids): raise ValueError("Invalid paper IDs")
    for pid in ids:
        if not (root / pid).resolve().is_relative_to(root): raise ValueError("Paper path escapes corpus root")
    if config.chunking.mode == "auto":
        raise ValueError("chunking.mode auto is resolved by `pydb plan`; set explicit chunking values before building")
    memory_fit = check_build_fit(config, root, ids)
    if memory_fit.get("checked") is False and memory_fit.get("reason") != "embedding is disabled":
        print(f"pydb: warning: memory fit was not checked: {memory_fit.get('reason')}", file=__import__("sys").stderr)
    if memory_fit.get("overBudgetAllowed"):
        print("pydb: warning: estimated vector index exceeds the detected memory budget; allow_over_budget is set", file=__import__("sys").stderr)
    build_id, input_hash = _identity(config, ids)
    out = config.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    final = out / build_id
    if final.exists():
        report = json.loads((final / "metadata.json").read_text())
        if not report.get("complete") or report.get("inputHash") != input_hash: raise ValueError("Existing immutable build identity mismatch")
        checked = validate_build(final, config.required_queries)
        if not checked["valid"]: raise ValueError("Existing immutable build failed validation")
        return report
    work = out / f".work-{build_id}"; work.mkdir(exist_ok=True)
    db_path = work / f"{config.alias}.sqlite"
    resumed = _resumable_state(db_path, build_id, input_hash) if config.resume and db_path.exists() else None
    if db_path.exists() and resumed is None: db_path.unlink()
    if resumed is not None:
        counts, quarantined = resumed
        print(f"pydb: resuming completed ingestion ({counts['papers']} papers, {counts['chunks']} chunks)", file=__import__("sys").stderr)
        return _finalize(config, db_path, work, build_id, input_hash, counts, quarantined, out, final, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), memory_fit)
    started = time.time(); started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); quarantined = []; papers = chunks = 0
    with sqlite3.connect(db_path) as db:
        db.executescript(SCHEMA)
        db.execute("INSERT OR REPLACE INTO database_meta VALUES (?,?)", ("offset_convention", json.dumps("normalized-v2; offsets index normalized_text; offset_map stores originalToNormalized and normalizedToOriginal boundary arrays")))
        for pid in ids:
            try:
                paper_chunks = 0
                with db:
                    knowledge = root / pid / "knowledge"
                    obj = {}
                    for name in BUNDLE_FILES:
                        file = knowledge / name
                        if name == "coherence.json" and not file.exists(): obj[name] = {}; continue
                        resolved = file.resolve(strict=True)
                        if not resolved.is_relative_to((root / pid).resolve()): raise ValueError("Bundle file escapes paper directory")
                        obj[name] = json.loads(resolved.read_text(encoding="utf-8"))
                    manifest, doc, section_bundle = obj["manifest.json"], obj["document.json"], obj["sections.json"]
                    quality, retrieval, inventory = obj["quality.json"], obj["retrieval.json"], obj["inventory.json"]
                    sections = section_bundle.get("sections")
                    if manifest.get("schema_version") != 4 or not doc.get("doi") or not doc.get("title") or not isinstance(sections, list): raise ValueError("Invalid bundle-v4 document")
                    texpath = root / pid / "main.tex"
                    if texpath.exists() and not texpath.resolve(strict=True).is_relative_to((root / pid).resolve()): raise ValueError("Source file escapes paper directory")
                    original = texpath.read_text(encoding="utf-8") if texpath.exists() else ""
                    spans, norm, maps = verified_spans(original, sections); nh = sha256(norm); sh = sha256(original)
                    paper = f"{config.alias}:{build_id}:{manifest['document_id']}"
                    db.execute("INSERT INTO papers VALUES (?,?,?,?,?,?,?)", (paper, build_id, manifest["document_id"], doc["doi"], doc["title"], (doc.get("article_type") or {}).get("category"), json.dumps({"paperId": pid, "schemaVersion": 4})))
                    db.execute("INSERT INTO source_texts VALUES (?,?,?,?,?,?)", (paper, original, norm, json.dumps({"originalToNormalized": maps.originalToNormalized, "normalizedToOriginal": maps.normalizedToOriginal}), sh, nh))
                    flags = []
                    conv = quality.get("conversion", {})
                    if not original: flags.append("missing_source_text")
                    for key, value in (("extractor", conv.get("extractor")), ("latex_unicode_replacements", conv.get("latex_unicode_replacements")), ("latex_sanity_repairs", conv.get("latex_sanity_repairs")), ("abstract_repair_confidence", (conv.get("abstract_repair") or {}).get("confidence"))):
                        if value is not None: flags.append(f"{key}:{value}")
                    db.execute("INSERT INTO quality_signals VALUES (?,?,?)", (paper, json.dumps(flags), max(0, 1 - (conv.get("latex_sanity_repairs", 0) or 0)*.05 - (conv.get("latex_unicode_replacements", 0) or 0)*.01)))
                    secmap = {}
                    for sec in sections:
                        sid = f"{paper}:{sec['id']}"; secmap[sec["id"]] = sid
                        content = " ".join(filter(None, [sec.get("purpose"), *(sec.get("knowledge", {}).get("key_points", [])), *(c.get("statement", "") for c in sec.get("knowledge", {}).get("claims", []))]))
                        db.execute("INSERT INTO sections VALUES (?,?,?,?,?)", (sid, paper, sec.get("title", sec["id"]), json.dumps(_pages(sec)), content))
                    snips = chunk_verified_spans(spans, norm, config.chunking)
                    records = retrieval.get("records", retrieval.get("retrieval", retrieval.get("chunks", [])))
                    for item in records:
                        text = item.get("content", item.get("text", item.get("summary", "")))
                        if text: snips.append({"text": text, "section": item.get("section_id", item.get("section", "retrieval")), "title": item.get("section_title", item.get("title", "Retrieved content")), "pages": _pages(item) or item.get("pages", []), "trust": "generated_summary" if item.get("generated") or item.get("kind") == "article_summary" else "extracted_text", "keywords": " ".join(item.get("keywords", [])), "formula": " ".join(item.get("formula_terms", []))})
                    for fig in [*inventory.get("figures", []), *(x for x in records if x.get("kind") == "figure")]:
                        for part in fig.get("content_parts", []):
                            if part.get("text"): snips.append({"text": part["text"], "section": fig.get("section_id", "figure"), "title": fig.get("caption", "Figure"), "pages": fig.get("pages", []), "trust": "extracted_text" if part.get("provenance_class") == "extractive" else "generated_summary", "keywords": " ".join(fig.get("keywords", [])), "formula": " ".join(fig.get("formula_terms", []))})
                    claims = [*inventory.get("claims", []), *inventory.get("chunks", [])]
                    claims += [{**c, "section_id": s["id"]} for s in inventory.get("sections", []) for c in s.get("claims", [])]
                    for claim in claims:
                        text = claim.get("statement", claim.get("text", ""))
                        if text: snips.append({"text": text, "section": claim.get("section_id", "inventory"), "title": claim.get("section_title", "Inventory claim"), "pages": claim.get("evidence_pages", []), "trust": "extracted_text", "keywords": " ".join(claim.get("keywords", [])), "formula": " ".join(claim.get("formula_terms", []))})
                    for sec in sections:
                        k = sec.get("knowledge", {}); text = " ".join(filter(None, [sec.get("purpose"), *k.get("key_points", []), *(c.get("statement", "") for c in k.get("claims", []))]))
                        if text: snips.append({"text": text, "section": sec["id"], "title": sec.get("title", sec["id"]), "pages": _pages(sec), "trust": "extracted_text", "keywords": " ".join(k.get("retrieval_keywords", [])), "formula": " ".join(t.get("term", "") for t in k.get("terms", []))})
                    if (doc.get("article_summary") or {}).get("text"): snips.append({"text": doc["article_summary"]["text"], "section": "summary", "title": doc["title"], "pages": doc["article_summary"].get("source_pages", []), "trust": "generated_summary", "keywords": " ".join(doc.get("keywords", [])), "formula": ""})
                    seen = set()
                    for item in snips:
                        key = sha256(item["section"] + "\0" + item["text"])
                        if key in seen: continue
                        seen.add(key); cid = f"{paper}:chunk:{paper_chunks}"
                        start, end = item.get("start"), item.get("end")
                        db.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (cid,paper,item["section"],item["title"],item["text"],item["keywords"],item["formula"],json.dumps(item["pages"]),item["trust"],nh,start,end,json.dumps(flags if item["trust"] != "generated_summary" else ["generated"])))
                        db.execute("INSERT INTO chunks_fts VALUES (?,?,?,?,?)", (cid,item["title"],item["text"],item["keywords"],item["formula"]))
                        db.execute("INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?)", (cid+":e",cid,paper,item["section"],json.dumps(item["pages"]),item["trust"],nh,start,end,item["text"] if start is not None else None))
                        paper_chunks += 1
                    chunks += paper_chunks
                    papers += 1
                    print(f"pydb: ingested {pid} ({papers}/{len(ids)})", file=__import__("sys").stderr)
            except Exception:
                quarantined.append({"paperId": pid, "reason": "invalid_or_unreadable_bundle"})
        counts = {"directories": len(ids), "papers": papers, "chunks": chunks, "quarantined": len(quarantined)}
        metadata = {"schema_version": 1, "build_id": build_id, "db_id": config.alias, "alias": config.alias, "corpus_root": str(root), "counts": counts, "extractor_version": EXTRACTOR_VERSION, "normalizer_version": NORMALIZER_VERSION, "capabilities": {"lexical": True, "structured_recipes": False, "numeric_filters": False, "route_queries": False, "embeddings": bool(config.embedding)}}
        for k,v in metadata.items(): db.execute("INSERT OR REPLACE INTO database_meta VALUES (?,?)", (k,json.dumps(v)))
        db.execute("INSERT INTO build_runs VALUES (?,?,?,?,?,?,?,?)", (build_id,input_hash,EXTRACTOR_VERSION,NORMALIZER_VERSION,time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime(started)),int((time.time()-started)*1000),json.dumps(counts),json.dumps(quarantined)))
    return _finalize(config, db_path, work, build_id, input_hash, counts, quarantined, out, final, started_at, memory_fit)

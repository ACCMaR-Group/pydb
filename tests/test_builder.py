import json
import sqlite3
from pathlib import Path

import pytest

from pydb.builder import build
from pydb.config import Config
from pydb.validation import validate_build


def fixture(root, papers=None):
    root.mkdir(parents=True, exist_ok=True)
    papers = papers or [
        ("paper1", r"\section{Results} Alpha result. Next sentence.", "results", "Results"),
        ("paper2", r"\section{Methods} Beta method. More details.", "methods", "Methods"),
    ]
    for pid, tex, sid, title in papers:
        knowledge = root / pid / "knowledge"
        knowledge.mkdir(parents=True)
        (knowledge.parent / "main.tex").write_text(tex, encoding="utf-8")
        bundle = {
            "manifest.json": {"schema_version": 4, "document_id": pid},
            "document.json": {"doi": f"10.1/{pid}", "title": f"Test {pid}"},
            "sections.json": {"sections": [{"id": sid, "title": title, "page_range": {"start": 1, "end": 1}}]},
            "quality.json": {},
            "retrieval.json": {"records": [{"content": f"summary {pid}", "generated": True}]},
            "inventory.json": {},
            "coherence.json": {},
        }
        for name, value in bundle.items():
            (knowledge / name).write_text(json.dumps(value), encoding="utf-8")
    return root


def config(corpus, out, required=None):
    return Config(corpus, None, out, "science", None, False, 0, out.parent / "input.yaml", "python", required or [])


def test_build_and_retrieval_validation(tmp_path):
    corpus = fixture(tmp_path / "corpus", [("paper1", r"\section{Results} The 😀 value is 42. Next sentence.", "results", "Results")])
    cfg = config(corpus, tmp_path / "out", [{"query": "42", "paper_id": "paper1", "section": "results"}])
    report = build(cfg)
    assert report["complete"]
    assert validate_build(report["databasePath"], cfg.required_queries)["valid"]
    assert validate_build(Path(report["databasePath"]).parent)["valid"]
    assert validate_build(Path(report["databasePath"]))["requiredQueries"][0]["matchedTrustLevels"] == ["verified_span"]
    failed = validate_build(report["databasePath"], [{"query": "unmatched", "paper_id": "paper1"}])
    assert not failed["valid"] and failed["requiredQueries"][0]["passed"] is False
    with sqlite3.connect(report["databasePath"]) as db:
        assert {row[0] for row in db.execute("SELECT DISTINCT trust_level FROM chunks")} == {"verified_span", "generated_summary"}
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        db.execute("UPDATE chunks SET text='tampered' WHERE trust_level='generated_summary'")
        db.commit()
    assert "fts_alignment_mismatch" in validate_build(report["databasePath"])["errors"]


def test_multisection_pages_and_section_mapping(tmp_path):
    corpus = fixture(tmp_path / "corpus", [("paper1", "\\section{Intro} First text.\\section{Results} Second text.", "intro", "Introduction")])
    sec = corpus / "paper1" / "knowledge" / "sections.json"
    sec.write_text(json.dumps({"sections": [
        {"id": "intro", "title": "Introduction", "page_range": {"start": 1, "end": 2}},
        {"id": "results", "title": "Results", "page_range": {"start": 3, "end": 4}},
    ]}), encoding="utf-8")
    report = build(config(corpus, tmp_path / "out"))
    with sqlite3.connect(report["databasePath"]) as db:
        rows = db.execute("SELECT section,text,pages FROM chunks WHERE trust_level='verified_span' ORDER BY start_offset").fetchall()
    assert [(row[0], row[1]) for row in rows] == [("intro", "Intro First text."), ("results", "Results Second text.")]
    assert [json.loads(row[2]) for row in rows] == [[1, 2], [3, 4]]


def test_rejected_retrieval_gate_keeps_final_absent(tmp_path):
    corpus = fixture(tmp_path / "corpus", [("paper1", r"\section{Results} Some evidence.", "results", "Results")])
    out = tmp_path / "out"
    cfg = config(corpus, out, [{"query": "no_such_term", "paper_id": "paper1", "section": "results"}])
    with pytest.raises(ValueError, match="validation"):
        build(cfg)
    assert list(out.glob("build-*/metadata.json")) == []
    assert list(out.glob(".work-*/science.sqlite"))


def test_resume_reuses_completed_ingestion(tmp_path, monkeypatch, capsys):
    import pydb.builder as builder
    corpus = fixture(tmp_path / "corpus", [("paper1", r"\section{Results} First evidence.", "results", "Results")])
    cfg = config(corpus, tmp_path / "out")
    calls = []
    real_normalize = builder.normalize_latex_with_map
    def counting(text):
        calls.append(text)
        return real_normalize(text)
    monkeypatch.setattr(builder, "normalize_latex_with_map", counting)
    real_validate = builder.validate_build
    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated interruption after ingestion")
    monkeypatch.setattr(builder, "validate_build", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build(cfg)
    assert calls
    monkeypatch.setattr(builder, "validate_build", real_validate)
    report = build(cfg)
    assert report["complete"] and report["counts"]["papers"] == 1
    assert len(calls) == 1
    assert "resuming completed ingestion" in capsys.readouterr().err
    with sqlite3.connect(report["databasePath"]) as db:
        assert db.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1


def test_resume_false_forces_fresh_ingestion(tmp_path, monkeypatch):
    import pydb.builder as builder
    from dataclasses import replace
    corpus = fixture(tmp_path / "corpus", [("paper1", r"\section{Results} First evidence.", "results", "Results")])
    cfg = config(corpus, tmp_path / "out")
    calls = []
    real_normalize = builder.normalize_latex_with_map
    def counting(text):
        calls.append(text)
        return real_normalize(text)
    monkeypatch.setattr(builder, "normalize_latex_with_map", counting)
    real_validate = builder.validate_build
    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated interruption after ingestion")
    monkeypatch.setattr(builder, "validate_build", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build(cfg)
    assert len(calls) == 1
    monkeypatch.setattr(builder, "validate_build", real_validate)
    report = build(replace(cfg, resume=False))
    assert report["complete"] and len(calls) == 2


def test_bundle_symlink_cannot_escape_paper_directory(tmp_path):
    corpus = fixture(tmp_path / "corpus", [("paper1", r"\\section{Results} First evidence.", "results", "Results")])
    knowledge = corpus / "paper1" / "knowledge"
    outside = tmp_path / "outside.json"
    outside.write_text((knowledge / "document.json").read_text(encoding="utf-8"), encoding="utf-8")
    (knowledge / "document.json").unlink()
    (knowledge / "document.json").symlink_to(outside)
    cfg = Config(corpus, None, tmp_path / "out", "science", None, False, 1, tmp_path / "input.yaml", "python", [])
    with pytest.raises(ValueError, match="escapes"):
        build(cfg)
    assert list((tmp_path / "out").glob("build-*/metadata.json")) == []


def test_failed_second_paper_rolls_back_partial_rows(tmp_path):
    corpus = fixture(tmp_path / "corpus", [
        ("paper1", r"\section{Results} First evidence.", "results", "Results"),
        ("paper2", r"\section{Methods} Second evidence.", "methods", "Methods"),
    ])
    # Fail late while traversing inventory, after the paper/source/sections
    # and initial verified chunks have been written in the paper transaction.
    bad = corpus / "paper2" / "knowledge" / "inventory.json"
    bad.write_text(json.dumps({"figures": [None]}), encoding="utf-8")
    cfg = Config(corpus, None, tmp_path / "out", "science", None, False, 1, tmp_path / "input.yaml", "python", [])
    report = build(cfg)
    assert report["counts"]["directories"] == 2
    assert report["counts"]["papers"] == 1
    assert report["counts"]["quarantined"] == 1
    with sqlite3.connect(report["databasePath"]) as db:
        assert db.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == report["counts"]["chunks"]
        assert db.execute("SELECT COUNT(*) FROM source_texts").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM sections").fetchone()[0] == 1

import json
from dataclasses import replace

import pytest

from pydb.config import ChunkingConfig, Config, EmbeddingConfig
from pydb.estimation import check_build_fit, detect_available_memory, estimate_build, plan_estimate


def make_corpus(root, count=3):
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        paper = root / f"paper{index}"
        knowledge = paper / "knowledge"
        knowledge.mkdir(parents=True)
        (paper / "main.tex").write_text("\\section{Results} First sentence. Second sentence. Third sentence.", encoding="utf-8")
        (knowledge / "sections.json").write_text(json.dumps({"sections": [
            {"id": "results", "title": "Results", "page_range": {"start": 1, "end": 1}}
        ]}), encoding="utf-8")
    return root


def make_config(root, tmp_path, embedding=None, chunking=None, **kwargs):
    return Config(root, None, tmp_path / "out", "science", embedding, False, 0,
                  tmp_path / "input.yaml", "python", [], chunking=chunking or ChunkingConfig("passage", 1200, 100), **kwargs)


def test_detect_available_memory_min_of_limits(tmp_path):
    meminfo = tmp_path / "meminfo"
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    meminfo.write_text("MemAvailable: 1000 kB\n", encoding="utf-8")
    (cgroup / "memory.max").write_text("4000000\n", encoding="utf-8")
    (cgroup / "memory.current").write_text("1000000\n", encoding="utf-8")
    result = detect_available_memory(meminfo=meminfo, cgroup_root=cgroup)
    assert result["availableBytes"] == 1000 * 1024
    assert result["smallestSource"] == "meminfo"
    meminfo.write_text("MemAvailable: 10000 kB\n", encoding="utf-8")
    result = detect_available_memory(meminfo=meminfo, cgroup_root=cgroup)
    assert result["availableBytes"] == 3000000
    assert result["smallestSource"] == "cgroup-v2"
    (cgroup / "memory.max").write_text("max\n", encoding="utf-8")
    assert detect_available_memory(meminfo=meminfo, cgroup_root=cgroup)["availableBytes"] == 10000 * 1024


def test_detect_available_memory_cgroup_v1_and_missing(tmp_path):
    meminfo = tmp_path / "meminfo"
    cgroup = tmp_path / "cgroup"
    (cgroup / "memory").mkdir(parents=True)
    (cgroup / "memory" / "memory.limit_in_bytes").write_text("8000\n", encoding="utf-8")
    (cgroup / "memory" / "memory.usage_in_bytes").write_text("2000\n", encoding="utf-8")
    meminfo.write_text("MemAvailable: 100 kB\n", encoding="utf-8")
    result = detect_available_memory(meminfo=meminfo, cgroup_root=cgroup)
    assert result["availableBytes"] == 6000
    assert result["smallestSource"] == "cgroup-v1"
    (cgroup / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n", encoding="utf-8")
    meminfo.unlink()
    assert detect_available_memory(meminfo=meminfo, cgroup_root=cgroup)["availableBytes"] is None


def test_estimate_build_is_deterministic_and_ordered(tmp_path):
    root = make_corpus(tmp_path / "corpus", 3)
    config = make_config(root, tmp_path)
    first, second = estimate_build(config), estimate_build(config)
    assert first == second
    assert first["sampledPapers"] == first["corpusPapers"] == 3
    assert first["readOnly"] is True
    counts = {item["mode"]: item["chunks"] for item in first["candidates"]}
    assert counts["sentence"] > counts["passage"] >= counts["coarse"]
    assert counts["sentence"] is not None


def test_plan_estimate_recommends_finest_fitting_mode(tmp_path, monkeypatch):
    import pydb.estimation as estimation
    root = make_corpus(tmp_path / "corpus", 3)
    embedding = EmbeddingConfig("https://api.example/v1", "mock", expected_dimensions=2)
    config = make_config(root, tmp_path, embedding=embedding)
    monkeypatch.setattr(estimation, "detect_available_memory", lambda **_: {"availableBytes": 10**12, "sources": ["test"]})
    plan = plan_estimate(config)
    assert plan["chunking"]["mode"] == "passage"
    assert plan["memory"]["dimensions"] == 2
    assert plan["memory"]["fits"] is True
    assert plan["recommendation"]["mode"] == "sentence"
    monkeypatch.setattr(estimation, "detect_available_memory", lambda **_: {"availableBytes": 1, "sources": ["test"]})
    plan = plan_estimate(replace(config, chunking=ChunkingConfig("auto")))
    assert plan["memory"]["fits"] is False
    assert plan["recommendation"] is None


def test_plan_estimate_without_dimensions_is_unmeasured(tmp_path):
    root = make_corpus(tmp_path / "corpus", 2)
    config = make_config(root, tmp_path, embedding=EmbeddingConfig("https://api.example/v1", "mock"))
    plan = plan_estimate(config)
    assert plan["memory"]["dimensions"] is None
    assert plan["memory"]["fits"] is None
    assert plan["recommendation"] is None


def test_check_build_fit_fails_before_ingestion_unless_allowed(tmp_path, monkeypatch):
    import pydb.estimation as estimation
    root = make_corpus(tmp_path / "corpus", 2)
    embedding = EmbeddingConfig("https://api.example/v1", "mock", expected_dimensions=2)
    config = make_config(root, tmp_path, embedding=embedding)
    monkeypatch.setattr(estimation, "detect_available_memory", lambda **_: {"availableBytes": 1, "sources": ["test"]})
    with pytest.raises(ValueError, match="does not fit"):
        check_build_fit(config)
    allowed = check_build_fit(replace(config, allow_over_budget=True))
    assert allowed["overBudgetAllowed"] is True
    unmeasured = check_build_fit(replace(config, embedding=EmbeddingConfig("https://api.example/v1", "mock")))
    assert unmeasured["checked"] is False

import json

from pydb.cli import main
from pydb.config import EmbeddingConfig, load_config


def test_plan_and_validate_cli(tmp_path, capsys):
    (tmp_path / "paper1").mkdir()
    recipe = tmp_path / "input.yaml"
    recipe.write_text(f"corpus_root: {tmp_path}\noutput_dir: {tmp_path / 'out'}\n")
    assert main(["plan", str(recipe)]) == 0
    plan = json.loads(capsys.readouterr().out)
    output = capsys.readouterr().out
    assert plan["alias"] == "science"
    assert plan["corpus"]["valid"] is True
    assert plan["sqlite"]["fts5Available"] is True
    assert plan["staging"]["willWrite"] is False
    assert "api_key" not in output
    assert not (tmp_path / "out").exists()


def test_plan_reports_chroma_dependency_without_importing_it(tmp_path, capsys, monkeypatch):
    import importlib.util
    recipe = tmp_path / "input.yaml"
    (tmp_path / "paper1").mkdir()
    recipe.write_text(f"corpus_root: {tmp_path}\noutput_dir: {tmp_path / 'out'}\nembedding:\n  enabled: true\n  base_url: https://example.invalid/v1\n  model: tiny\n  api_key: literal-secret\nchroma:\n  enabled: true\n")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert main(["plan", str(recipe)]) == 1
    result = capsys.readouterr()
    assert "chromadb is not installed" in result.err
    assert "literal-secret" not in result.out + result.err
    assert not (tmp_path / "out").exists()


def _publish_fixture(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    root = tmp_path / "sciturbo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "import-science-db.ts").write_text("", encoding="utf-8")
    return build, root


def test_publish_allows_long_import_and_surfaces_errors(tmp_path, monkeypatch, capsys):
    import pydb.cli as cli
    build, root = _publish_fixture(tmp_path)
    monkeypatch.setattr(cli, "validate_build", lambda path: {"valid": True})
    captured = {}
    class Proc:
        returncode = 0
        stdout = "ok\n"
        stderr = ""
    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return Proc()
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.main(["publish", str(build), "--sciturbo-root", str(root)]) == 0
    assert captured["timeout"] == 3600
    assert cli.main(["publish", str(build), "--sciturbo-root", str(root), "--in-place", "--root", str(tmp_path)]) == 0
    assert "--in-place" in captured["command"] and "--root" in captured["command"]
    assert cli.main(["publish", str(build), "--sciturbo-root", str(root), "--root", str(tmp_path)]) == 1
    assert "--root requires --in-place" in capsys.readouterr().err
    class Failed(Proc):
        returncode = 1
        stdout = ""
        stderr = "importer exploded\n"
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs: Failed())
    assert cli.main(["publish", str(build), "--sciturbo-root", str(root)]) == 1
    assert "importer exploded" in capsys.readouterr().err


def test_plan_reports_memory_and_chunking_without_writing(tmp_path, capsys):
    paper = tmp_path / "paper1"
    (paper / "knowledge").mkdir(parents=True)
    (paper / "main.tex").write_text("\\section{Results} First sentence. Second sentence.", encoding="utf-8")
    (paper / "knowledge" / "sections.json").write_text(json.dumps({"sections": [
        {"id": "results", "title": "Results", "page_range": {"start": 1, "end": 1}}
    ]}), encoding="utf-8")
    recipe = tmp_path / "input.yaml"
    recipe.write_text(
        f"corpus_root: {tmp_path}\noutput_dir: {tmp_path / 'out'}\n"
        "embedding:\n  enabled: true\n  base_url: https://api.example/v1\n  model: demo\n  expected_dimensions: 4096\n"
        "chunking:\n  mode: passage\n")
    assert main(["plan", str(recipe)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["planning"]["chunking"]["mode"] == "passage"
    assert plan["planning"]["memory"]["dimensions"] == 4096
    assert plan["planning"]["memory"]["estimate"] == "sampled-v1"
    assert plan["planning"]["memory"]["sampledPapers"] == 1
    assert "recommendation" in plan["planning"]
    assert not (tmp_path / "out").exists()


def test_plan_fails_for_empty_corpus_without_writing(tmp_path, capsys):
    recipe = tmp_path / "input.yaml"
    recipe.write_text(f"corpus_root: {tmp_path}\noutput_dir: {tmp_path / 'out'}\n")
    assert main(["plan", str(recipe)]) == 1
    assert "no selected paper" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_plan_fails_for_missing_corpus_without_writing(tmp_path, capsys):
    missing = tmp_path / "missing-corpus"
    recipe = tmp_path / "input.yaml"
    recipe.write_text(f"corpus_root: {missing}\noutput_dir: {tmp_path / 'out'}\n")
    assert main(["plan", str(recipe)]) == 1
    assert "Corpus root" in capsys.readouterr().err
    assert not missing.exists()
    assert not (tmp_path / "out").exists()

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

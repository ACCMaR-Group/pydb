import os

import pytest

from pydb.config import load_config, resolve_api_key


def test_optional_disabled_embedding_and_sanitized_semantics(tmp_path):
    recipe = tmp_path / "input.yaml"
    recipe.write_text("corpus_root: corpus\noutput_dir: out\nembedding:\n  enabled: false\n")
    config = load_config(recipe)
    assert config.embedding is None
    assert "api_key" not in str(config.semantic_dict())


def test_environment_key_is_resolved_only_on_demand(tmp_path, monkeypatch):
    recipe = tmp_path / "input.yaml"
    recipe.write_text("corpus_root: corpus\noutput_dir: out\nembedding:\n  enabled: true\n  base_url: https://api.example/v1\n  model: demo\n  api_key: ${PYDB_TEST_KEY}\n")
    config = load_config(recipe)
    assert "PYDB_TEST_KEY" not in str(config.semantic_dict())
    assert "api_key" not in repr(config)
    monkeypatch.setenv("PYDB_TEST_KEY", "not-for-logs")
    assert resolve_api_key(config.embedding) == "not-for-logs"


def test_rejects_ambiguous_key_and_unsafe_url(tmp_path):
    recipe = tmp_path / "input.yaml"
    common = "corpus_root: corpus\noutput_dir: out\nembedding:\n  enabled: true\n  model: demo\n"
    recipe.write_text(common + "  base_url: https://api.example\n  api_key: literal\n  api_key_env: KEY\n")
    with pytest.raises(ValueError, match="only one"):
        load_config(recipe)
    recipe.write_text(common + "  base_url: https://user:secret@api.example\n")
    with pytest.raises(ValueError, match="credentials"):
        load_config(recipe)


@pytest.mark.parametrize("body", [
    "embedding:\n  enabled: 'false'\n",
    "embedding:\n  enabled: true\n  base_url: https://api.example/v1\n  model: demo\n  dimensions: true\n",
    "embedding:\n  enabled: true\n  base_url: https://api.example/v1\n  model: demo\n  retries: 999999999\n",
    "embedding:\n  enabled: true\n  base_url: https://api.example/v1\n  model: demo\n  timeout: .nan\n",
    "unknown_option: true\n",
])
def test_rejects_malformed_recipe_boundaries(tmp_path, body):
    recipe = tmp_path / "input.yaml"
    recipe.write_text("corpus_root: corpus\noutput_dir: out\n" + body)
    with pytest.raises(ValueError):
        load_config(recipe)


def test_chroma_requires_embedding(tmp_path):
    recipe = tmp_path / "input.yaml"
    recipe.write_text("corpus_root: corpus\noutput_dir: out\nchroma:\n  enabled: true\n")
    with pytest.raises(ValueError, match="requires embeddings"):
        load_config(recipe)

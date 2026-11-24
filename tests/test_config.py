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


def test_resume_defaults_true_and_does_not_change_semantic_identity(tmp_path):
    base = "corpus_root: corpus\noutput_dir: out\n"
    recipe = tmp_path / "input.yaml"
    recipe.write_text(base)
    default = load_config(recipe)
    assert default.resume is True
    recipe.write_text(base + "resume: false\n")
    disabled = load_config(recipe)
    assert disabled.resume is False
    assert disabled.semantic_dict() == default.semantic_dict()
    recipe.write_text(base + "resume: 'false'\n")
    with pytest.raises(ValueError, match="resume must be boolean"):
        load_config(recipe)


def test_chunking_modes_defaults_and_validation(tmp_path):
    recipe = tmp_path / "input.yaml"
    base = "corpus_root: corpus\noutput_dir: out\n"
    recipe.write_text(base)
    config = load_config(recipe)
    assert (config.chunking.mode, config.chunking.target_chars, config.chunking.max_gap) == ("passage", 1200, 100)
    recipe.write_text(base + "chunking:\n  mode: sentence\n")
    config = load_config(recipe)
    assert config.chunking.mode == "sentence" and config.chunking.target_chars is None
    recipe.write_text(base + "chunking:\n  mode: coarse\n")
    assert load_config(recipe).chunking.target_chars == 2400
    recipe.write_text(base + "chunking:\n  mode: custom\n  target_chars: 900\n  max_gap: 50\n")
    config = load_config(recipe)
    assert (config.chunking.target_chars, config.chunking.max_gap) == (900, 50)
    recipe.write_text(base + "chunking:\n  mode: auto\n")
    assert load_config(recipe).chunking.mode == "auto"
    for body in ("chunking:\n  mode: sentence\n  target_chars: 100\n",
                 "chunking:\n  mode: custom\n",
                 "chunking:\n  mode: nope\n",
                 "chunking:\n  mode: passage\n  target_chars: 0\n",
                 "chunking:\n  mode: passage\n  max_gap: -1\n"):
        recipe.write_text(base + body)
        with pytest.raises(ValueError):
            load_config(recipe)


def test_expected_dimensions_and_chunking_enter_semantics(tmp_path):
    recipe = tmp_path / "input.yaml"
    base = "corpus_root: corpus\noutput_dir: out\nembedding:\n  enabled: true\n  base_url: https://api.example/v1\n  model: demo\n"
    recipe.write_text(base + "  expected_dimensions: 4096\n")
    config = load_config(recipe)
    assert config.embedding.expected_dim() == 4096
    passage = config.semantic_dict()
    assert passage["chunking"]["mode"] == "passage"
    assert passage["embedding"]["expected_dimensions"] == 4096
    recipe.write_text(base + "  expected_dimensions: 4096\nchunking:\n  mode: sentence\n")
    assert load_config(recipe).semantic_dict() != passage
    recipe.write_text(base + "  dimensions: 4096\n  expected_dimensions: 1024\n")
    with pytest.raises(ValueError, match="disagree"):
        load_config(recipe)
    recipe.write_text(base + "  expected_dimensions: 0\n")
    with pytest.raises(ValueError, match="expected_dimensions"):
        load_config(recipe)


def test_allow_over_budget_is_boolean_and_not_semantic(tmp_path):
    recipe = tmp_path / "input.yaml"
    base = "corpus_root: corpus\noutput_dir: out\n"
    recipe.write_text(base)
    default = load_config(recipe).semantic_dict()
    recipe.write_text(base + "allow_over_budget: true\n")
    config = load_config(recipe)
    assert config.allow_over_budget is True
    assert config.semantic_dict() == default
    recipe.write_text(base + "allow_over_budget: 'true'\n")
    with pytest.raises(ValueError, match="allow_over_budget"):
        load_config(recipe)


def test_chroma_requires_embedding(tmp_path):
    recipe = tmp_path / "input.yaml"
    recipe.write_text("corpus_root: corpus\noutput_dir: out\nchroma:\n  enabled: true\n")
    with pytest.raises(ValueError, match="requires embeddings"):
        load_config(recipe)

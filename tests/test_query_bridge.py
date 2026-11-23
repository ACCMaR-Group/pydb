import sys
import types

import pytest

from pydb.config import load_config
from pydb.vector import _collection_metadata, embedding_fingerprint, query_candidates


@pytest.fixture
def query_setup(tmp_path, monkeypatch):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        f"corpus_root: corpus\noutput_dir: {tmp_path}\n"
        "alias: science\n"
        "embedding:\n  enabled: true\n  base_url: https://recipe.example/v1\n"
        "  model: recipe-model\n  api_key_env: RECIPE_SECRET\n  max_distance: 0.25\n"
        "chroma:\n  enabled: true\n"
    )
    build_id = "build-0123456789ab"
    index = tmp_path / build_id / "chroma"
    index.mkdir(parents=True)
    config = load_config(recipe)
    client_state = {"metadata": _collection_metadata(build_id, embedding_fingerprint(config)), "calls": []}

    class Collection:
        metadata = client_state["metadata"]

        def count(self):
            return 1

        def query(self, **kwargs):
            return {"ids": [["candidate"]], "distances": [[0.1]]}

    class Client:
        def get_collection(self, name, embedding_function=None):
            assert name == "evidence"
            return Collection()

    chromadb = types.SimpleNamespace(
        PersistentClient=lambda **kwargs: Client(),
        config=types.SimpleNamespace(Settings=lambda **kwargs: object()),
    )
    monkeypatch.setitem(sys.modules, "chromadb", chromadb)
    monkeypatch.setattr("pydb.vector.embed_texts", lambda embedding, texts: client_state["calls"].append((embedding, texts)) or [[1.0, 0.0]])
    request = {
        "protocolVersion": 1,
        "buildId": build_id,
        "embeddingFingerprint": embedding_fingerprint(config),
        "recipePath": str(recipe),
        "indexPath": str(index),
        "collection": "evidence",
        "query": "find evidence",
        "limit": 4,
    }
    return request, client_state, config


def test_bridge_profile_query_uses_explicit_key_without_environment(query_setup, monkeypatch):
    request, state, _ = query_setup
    monkeypatch.delenv("RECIPE_SECRET", raising=False)
    profile = {
        "base_url": "https://recipe.example/v1",
        "model": "recipe-model",
        "api_key": "${RECIPE_SECRET}",
        "document_prefix": "",
        "query_prefix": "",
    }
    request["embeddingConfig"] = profile
    request["embeddingFingerprint"] = embedding_fingerprint(load_config(request["recipePath"]))
    # The profile fingerprint is recipe-compatible and the key is never returned.
    response = query_candidates(request)
    embedding, texts = state["calls"][0]
    assert embedding.api_key == "${RECIPE_SECRET}"
    assert embedding.api_key_env is None
    assert texts == ["find evidence"]
    assert "${RECIPE_SECRET}" not in repr(response)


def test_bridge_profile_api_key_keeps_literal_backslash_sequence(query_setup):
    request, state, config = query_setup
    literal_key = "literal\\nsequence"
    request["embeddingConfig"] = {
        "base_url": "https://recipe.example/v1",
        "model": "recipe-model",
        "api_key": literal_key,
        "document_prefix": "",
        "query_prefix": "",
    }
    request["embeddingFingerprint"] = embedding_fingerprint(config)
    query_candidates(request)
    assert state["calls"][0][0].api_key == literal_key
    assert state["calls"][0][0].api_key_env is None


def test_bridge_profile_without_key_clears_recipe_environment_key(query_setup, monkeypatch):
    request, state, _ = query_setup
    monkeypatch.delenv("RECIPE_SECRET", raising=False)
    request["embeddingConfig"] = {
        "base_url": "https://recipe.example/v1",
        "model": "recipe-model",
        "document_prefix": "",
        "query_prefix": "",
    }
    response = query_candidates(request)
    assert state["calls"][0][0].api_key is None
    assert state["calls"][0][0].api_key_env is None
    assert response["candidates"][0]["chunkId"] == "candidate"


def test_bridge_fingerprint_mismatch_fails_before_embedding(query_setup):
    request, state, _ = query_setup
    request["embeddingConfig"] = {
        "base_url": "https://recipe.example/v1",
        "model": "other-model",
        "document_prefix": "",
        "query_prefix": "",
    }
    with pytest.raises(ValueError, match="fingerprint"):
        query_candidates(request)
    assert not state["calls"]


@pytest.mark.parametrize("profile", [
    {"base_url": "file:///tmp", "model": "model", "document_prefix": "", "query_prefix": ""},
    {"base_url": "https://api.example/v1", "model": "model", "api_key": "bad\nkey", "document_prefix": "", "query_prefix": ""},
    {"base_url": "https://api.example/v1", "model": "model", "api_key": "bad\rkey", "document_prefix": "", "query_prefix": ""},
])
def test_bridge_invalid_endpoint_or_key_is_sanitized(query_setup, profile):
    request, state, _ = query_setup
    request["embeddingConfig"] = profile
    with pytest.raises(ValueError) as error:
        query_candidates(request)
    assert "bad" not in str(error.value)
    assert "file:///" not in str(error.value)
    assert not state["calls"]


def test_custom_config_home_validates_published_index(query_setup, tmp_path):
    request, _, config = query_setup
    build_id = request["buildId"]
    config_home = tmp_path / "bridge-config"
    published = config_home / "databases" / config.alias / "builds" / build_id / "chroma"
    published.mkdir(parents=True)
    request["indexPath"] = str(published)
    request["configHome"] = str(config_home)
    response = query_candidates(request)
    assert response["buildId"] == build_id


def test_config_home_must_be_absolute(query_setup):
    request, _, _ = query_setup
    request["configHome"] = "relative/config"
    with pytest.raises(ValueError, match="config home"):
        query_candidates(request)

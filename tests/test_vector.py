import json
import os
import sqlite3

import pytest

from pydb.config import Config, EmbeddingConfig
from pydb.vector import build_vectors, embedding_fingerprint, query_candidates, validate_vectors

chromadb = pytest.importorskip("chromadb")


def _tiny_db(path):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE chunks(id,text,trust_level,section)")
        db.executemany("INSERT INTO chunks VALUES(?,?,?,?)", [
            ("stable-id", "verified material finding", "verified_span", "results"),
            ("summary-id", "generated summary", "generated_summary", "summary"),
            ("bad-id", "unverified", "extracted_text", "results"),
        ])


def test_embedding_only_creates_secret_free_checkpoint(tmp_path, monkeypatch):
    dbpath = tmp_path / "tiny.sqlite"
    _tiny_db(dbpath)
    emb = EmbeddingConfig("https://api.example/v1", "model", api_key="do-not-persist")
    config = Config(tmp_path, None, tmp_path, "science", emb, False, 0, tmp_path / "input.yaml", "/usr/bin/python3")
    import pydb.vector as vector
    calls = []
    def deterministic(c, texts):
        calls.append(list(texts))
        return [[float(i), 2.0] for i, _ in enumerate(texts)]
    monkeypatch.setattr(vector, "embed_texts", deterministic)
    metadata = build_vectors(config, dbpath, tmp_path / "work", "build-a")
    checkpoint_path = tmp_path / "work" / "vectors-checkpoint.sqlite"
    with sqlite3.connect(checkpoint_path) as checkpoint_db:
        checkpoint_rows = checkpoint_db.execute("SELECT chunk_id,vector FROM vectors").fetchall()
    checkpoint = json.dumps(checkpoint_rows)
    assert metadata["enabled"] is False and metadata["embeddingEnabled"] is True
    build_vectors(config, dbpath, tmp_path / "work", "build-a")
    assert len(calls) == 1
    assert "api_key" not in metadata and "do-not-persist" not in checkpoint
    assert "bad-id" not in checkpoint
    assert metadata["embeddingFingerprint"] == embedding_fingerprint(config)
    assert validate_vectors(dbpath, tmp_path / "work", metadata)["valid"]


def test_vectors_are_reused_across_build_ids_by_text_hash(tmp_path, monkeypatch):
    dbpath = tmp_path / "tiny.sqlite"
    _tiny_db(dbpath)
    emb = EmbeddingConfig("https://api.example/v1", "mock")
    config = Config(tmp_path, None, tmp_path, "science", emb, False, 0, tmp_path / "input.yaml", "python", [])
    import pydb.vector as vector
    calls = []
    def fake(c, texts):
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]
    monkeypatch.setattr(vector, "embed_texts", fake)
    build_vectors(config, dbpath, tmp_path / "work", "build-a")
    assert len(calls) == 1
    def fail(c, texts):
        raise AssertionError("re-embedded reusable text")
    monkeypatch.setattr(vector, "embed_texts", fail)
    metadata = build_vectors(config, dbpath, tmp_path / "work", "build-b")
    assert metadata["reusedVectors"] == 2 and metadata["embeddedVectors"] == 0
    with sqlite3.connect(tmp_path / "work" / "vectors-checkpoint.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM vectors WHERE build_id='build-b'").fetchone()[0] == 2
    assert validate_vectors(dbpath, tmp_path / "work", metadata)["valid"]


def test_embedding_fingerprint_includes_expected_dimensions(tmp_path):
    from pydb.config import Config
    def config(dim):
        return Config(tmp_path, None, tmp_path, "science",
                      EmbeddingConfig("https://api.example/v1", "mock", expected_dimensions=dim),
                      False, 0, tmp_path / "input.yaml", "python", [])
    assert embedding_fingerprint(config(4096)) != embedding_fingerprint(config(1024))


def test_chroma_resume_reuses_checkpoint_without_reembedding(tmp_path, monkeypatch):
    dbpath = tmp_path / "tiny.sqlite"
    _tiny_db(dbpath)
    recipe = tmp_path / "input.yaml"
    recipe.write_text(
        f"corpus_root: corpus\noutput_dir: {tmp_path}\n"
        "embedding:\n  enabled: true\n  base_url: https://api.example/v1\n  model: mock\n"
        "chroma:\n  enabled: true\n"
    )
    from pydb.config import load_config
    config = load_config(recipe)
    supplied = {"verified material finding": [1.0, 0.0], "generated summary": [0.0, 1.0]}
    import pydb.vector as vector
    calls = []
    def fake(c, texts):
        calls.append(list(texts))
        return [supplied[t] for t in texts]
    monkeypatch.setattr(vector, "embed_texts", fake)
    metadata = build_vectors(config, dbpath, tmp_path / "build-0123456789ab", "build-0123456789ab")
    assert len(calls) == 1
    build_vectors(config, dbpath, tmp_path / "build-0123456789ab", "build-0123456789ab")
    assert len(calls) == 1
    index = tmp_path / "build-0123456789ab" / "chroma"
    client = chromadb.PersistentClient(path=str(index), settings=chromadb.config.Settings(anonymized_telemetry=False))
    collection = client.get_collection("evidence", embedding_function=None)
    assert set(collection.get(include=[])["ids"]) == {"stable-id", "summary-id"}
    assert validate_vectors(dbpath, tmp_path / "build-0123456789ab", metadata)["valid"]


def test_real_chroma_index_and_query_use_supplied_vectors(tmp_path, monkeypatch):
    dbpath = tmp_path / "tiny.sqlite"
    _tiny_db(dbpath)
    recipe = tmp_path / "input.yaml"
    recipe.write_text(
        f"corpus_root: corpus\noutput_dir: {tmp_path}\n"
        "embedding:\n  enabled: true\n  base_url: https://api.example/v1\n  model: mock\n  max_distance: 0.1\n"
        "chroma:\n  enabled: true\n"
    )
    from pydb.config import load_config
    config = load_config(recipe)
    supplied = {"verified material finding": [1.0, 0.0], "generated summary": [0.0, 1.0]}
    import pydb.vector as vector
    monkeypatch.setattr(vector, "embed_texts", lambda c, texts: [supplied[t] for t in texts])
    metadata = build_vectors(config, dbpath, tmp_path / "build-0123456789ab", "build-0123456789ab")
    assert metadata["enabled"] is True
    index = tmp_path / "build-0123456789ab" / metadata["indexPath"]
    client = chromadb.PersistentClient(path=str(index), settings=chromadb.config.Settings(anonymized_telemetry=False))
    collection = client.get_collection("evidence", embedding_function=None)
    assert collection.count() == 2
    assert "bad-id" not in collection.get(include=[])["ids"]
    assert validate_vectors(dbpath, tmp_path / "build-0123456789ab", metadata)["valid"]
    query_vector = [1.0, 0.0]
    monkeypatch.setattr(vector, "embed_texts", lambda c, texts: [query_vector])
    request = {
        "protocolVersion": 1,
        "buildId": "build-0123456789ab",
        "embeddingFingerprint": metadata["embeddingFingerprint"],
        "recipePath": str(recipe),
        "indexPath": str(index),
        "collection": "evidence",
        "query": "verified material finding",
        "limit": 2,
    }
    response = query_candidates(request)
    assert response["candidates"][0]["chunkId"] == "stable-id"
    assert response["buildId"] == "build-0123456789ab"
    published = tmp_path / "config-home" / "databases" / "science" / "builds" / "build-0123456789ab" / "chroma"
    published.parent.mkdir(parents=True)
    import shutil
    shutil.copytree(index, published)
    monkeypatch.setenv("SCITURBO_CONFIG_DIR", str(tmp_path / "config-home"))
    request["indexPath"] = str(published)
    assert query_candidates(request)["candidates"][0]["chunkId"] == "stable-id"
    request["indexPath"] = str(tmp_path / "untrusted" / "chroma")
    with pytest.raises(ValueError, match="path"):
        query_candidates(request)

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from pathlib import Path

from .config import Config, load_config, query_embedding_config
from .embedding import embed_texts

PROTOCOL_VERSION = 1


def embedding_fingerprint(config: Config) -> str:
    if config.embedding is None:
        return ""
    identity = {"base_url": config.embedding.base_url, "model": config.embedding.model,
                "dimensions": config.embedding.dimensions, "document_prefix": config.embedding.document_prefix,
                "query_prefix": config.embedding.query_prefix}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _collection_metadata(build_id: str, fingerprint: str) -> dict:
    return {"protocolVersion": PROTOCOL_VERSION, "buildId": build_id,
            "embeddingFingerprint": fingerprint, "hnsw:space": "cosine"}


def _valid_vector(vector: list, dimensions: int | None) -> bool:
    return isinstance(vector, list) and bool(vector) and (dimensions is None or len(vector) == dimensions) and all(
        type(value) in (int, float) and math.isfinite(value) for value in vector)


def build_vectors(config: Config, db_path: str | Path, work_dir: str | Path, build_id: str) -> dict:
    if config.embedding is None:
        return {"enabled": False}
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    fingerprint = embedding_fingerprint(config)
    with sqlite3.connect(db_path) as db:
        rows = db.execute("SELECT id,text,trust_level FROM chunks WHERE trust_level='verified_span' OR (trust_level='generated_summary' AND section='summary') ORDER BY id").fetchall()
    checkpoint_path = work / "vectors-checkpoint.sqlite"
    with sqlite3.connect(checkpoint_path) as checkpoint:
        checkpoint.execute("CREATE TABLE IF NOT EXISTS vectors (build_id TEXT, fingerprint TEXT, chunk_id TEXT, text_hash TEXT, vector TEXT, PRIMARY KEY(build_id,fingerprint,chunk_id))")
        identities = [(row[0], hashlib.sha256(row[1].encode()).hexdigest()) for row in rows]
        pending = []
        vectors = {}
        expected_dimensions = config.embedding.dimensions
        for row, (chunk_id, text_hash) in zip(rows, identities):
            saved = checkpoint.execute("SELECT text_hash,vector FROM vectors WHERE build_id=? AND fingerprint=? AND chunk_id=?", (build_id, fingerprint, chunk_id)).fetchone()
            if saved and saved[0] == text_hash:
                try:
                    value = json.loads(saved[1])
                    if expected_dimensions is None and _valid_vector(value, None):
                        expected_dimensions = len(value)
                    if _valid_vector(value, expected_dimensions):
                        vectors[chunk_id] = value
                        continue
                except Exception:
                    pass
            pending.append((row, text_hash))
        for start in range(0, len(pending), config.embedding.batch_size):
            batch = pending[start:start + config.embedding.batch_size]
            embeddings = embed_texts(config.embedding, [config.embedding.document_prefix + item[0][1] for item in batch])
            if len(embeddings) != len(batch):
                raise ValueError("Invalid embedding batch")
            for (row, text_hash), vector in zip(batch, embeddings):
                if expected_dimensions is None and _valid_vector(vector, None):
                    expected_dimensions = len(vector)
                if not _valid_vector(vector, expected_dimensions):
                    raise ValueError("Invalid embedding batch")
                vectors[row[0]] = vector
                checkpoint.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?,?)", (build_id, fingerprint, row[0], text_hash, json.dumps(vector, separators=(",", ":"))))
            checkpoint.commit()
    metadata = {"enabled": False, "embeddingEnabled": True, "protocolVersion": PROTOCOL_VERSION,
                "buildId": build_id, "embeddingFingerprint": fingerprint,
                "recipePath": str(config.recipe_path), "pythonExecutable": config.python_executable,
                "checkpointPath": checkpoint_path.name}
    if config.chroma:
        try:
            import chromadb
        except ImportError:
            raise RuntimeError("Chroma requested but chromadb is not installed") from None
        index = work / "chroma"
        client = chromadb.PersistentClient(path=str(index), settings=chromadb.config.Settings(anonymized_telemetry=False))
        expected = _collection_metadata(build_id, fingerprint)
        try:
            collection = client.get_collection("evidence", embedding_function=None)
            if collection.metadata != expected:
                raise ValueError("Chroma collection identity mismatch")
        except ValueError:
            raise
        except Exception:
            collection = client.create_collection("evidence", metadata=expected, embedding_function=None)
        if rows:
            ordered = [row[0] for row in rows]
            max_batch_size = client.get_max_batch_size()
            row_map = {row[0]: row for row in rows}
            for start in range(0, len(ordered), max_batch_size):
                ids = ordered[start:start + max_batch_size]
                collection.upsert(ids=ids, embeddings=[vectors[cid] for cid in ids],
                                  documents=[row_map[cid][1] for cid in ids])
        metadata.update({"enabled": True, "indexPath": "chroma", "collection": "evidence"})
    return metadata


def validate_vectors(db_path: str | Path, artifact_dir: str | Path, metadata: dict) -> dict:
    if not metadata.get("embeddingEnabled"):
        return {"valid": not metadata.get("enabled", False), "errors": []}
    build_id, fingerprint = metadata.get("buildId"), metadata.get("embeddingFingerprint")
    errors = []
    artifact_dir = Path(artifact_dir)
    with sqlite3.connect(db_path) as db:
        rows = db.execute("SELECT id,text FROM chunks WHERE trust_level='verified_span' OR (trust_level='generated_summary' AND section='summary') ORDER BY id").fetchall()
    checkpoint = artifact_dir / metadata.get("checkpointPath", "")
    if not checkpoint.is_file():
        return {"valid": False, "errors": ["vector_checkpoint_missing"]}
    with sqlite3.connect(checkpoint) as db:
        saved = db.execute("SELECT chunk_id,text_hash,vector FROM vectors WHERE build_id=? AND fingerprint=? ORDER BY chunk_id", (build_id, fingerprint)).fetchall()
    expected = [(cid, hashlib.sha256(text.encode()).hexdigest()) for cid, text in rows]
    if [(row[0], row[1]) for row in saved] != expected:
        errors.append("vector_checkpoint_mapping_mismatch")
    dimensions = None
    for _, _, encoded in saved:
        try:
            vector = json.loads(encoded)
            if not _valid_vector(vector, dimensions):
                errors.append("vector_checkpoint_invalid")
                break
            dimensions = len(vector) if dimensions is None else dimensions
        except Exception:
            errors.append("vector_checkpoint_invalid")
            break
    if not metadata.get("enabled") or errors:
        return {"valid": not errors, "errors": errors}
    index = artifact_dir / str(metadata.get("indexPath", ""))
    if not index.is_dir():
        return {"valid": False, "errors": ["vector_index_missing"]}
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(index), settings=chromadb.config.Settings(anonymized_telemetry=False))
        collection = client.get_collection(metadata.get("collection", ""), embedding_function=None)
        if collection.metadata != _collection_metadata(build_id, fingerprint):
            errors.append("vector_index_identity_mismatch")
        got = set(collection.get(include=[])['ids'])
        if got != {row[0] for row in rows}:
            errors.append("vector_index_mapping_mismatch")
        return {"valid": not errors, "errors": errors}
    except Exception:
        return {"valid": False, "errors": ["vector_index_invalid"]}


def query_candidates(request: dict) -> dict:
    if request.get("protocolVersion") != PROTOCOL_VERSION:
        raise ValueError("Unsupported vector query protocol")
    if len(json.dumps(request).encode("utf-8")) > 65536:
        raise ValueError("Vector query request is too large")
    config = load_config(request.get("recipePath", os.environ.get("SCITURBO_PYDB_RECIPE", "")))
    if config.embedding is None:
        raise ValueError("Embedding is disabled")
    embedding = config.embedding
    if "embeddingConfig" in request:
        embedding = query_embedding_config(request["embeddingConfig"], embedding)
    fingerprint = embedding_fingerprint(Config(
        config.corpus_root, config.paper_ids, config.output_dir, config.alias, embedding,
        config.chroma, config.max_quarantined_papers, config.recipe_path,
        config.python_executable, config.required_queries))
    if fingerprint != request.get("embeddingFingerprint"):
        raise ValueError("Embedding fingerprint mismatch")
    if request.get("collection") != "evidence":
        raise ValueError("Invalid collection")
    index_path = Path(request["indexPath"])
    recipe_path = Path(request["recipePath"])
    build_id = request.get("buildId", "")
    local_index = (config.output_dir / build_id / "chroma").resolve()
    if "configHome" in request:
        raw_config_home = request["configHome"]
        if not isinstance(raw_config_home, str) or not Path(raw_config_home).is_absolute():
            raise ValueError("Invalid config home")
        config_home = Path(raw_config_home).resolve()
    else:
        config_home = Path(os.environ.get("SCITURBO_CONFIG_DIR", Path.home() / ".sciturbo")).expanduser().resolve()
    published_index = (config_home / "databases" / config.alias / "builds" / build_id / "chroma").resolve()
    if (not re.fullmatch(r"build-[a-f0-9]{12}", build_id) or not recipe_path.is_absolute()
            or not index_path.is_absolute() or not index_path.is_dir() or index_path.name != "chroma"
            or index_path.resolve() not in {local_index, published_index}):
        raise ValueError("Vector index path is missing or invalid")
    import chromadb
    client = chromadb.PersistentClient(path=str(index_path), settings=chromadb.config.Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(request["collection"], embedding_function=None)
    except Exception:
        raise ValueError("Vector collection does not exist") from None
    if collection.metadata != _collection_metadata(request["buildId"], fingerprint):
        raise ValueError("Vector index identity mismatch")
    if collection.count() == 0:
        return {"protocolVersion": PROTOCOL_VERSION, "buildId": request["buildId"], "embeddingFingerprint": fingerprint, "candidates": []}
    if not isinstance(request.get("query"), str) or not request["query"].strip():
        raise ValueError("Vector query must be nonempty text")
    vector = embed_texts(embedding, [embedding.query_prefix + request["query"]])[0]
    result = collection.query(query_embeddings=[vector], n_results=max(1, min(int(request.get("limit", 10)), 200)))
    max_distance = config.embedding.max_distance
    candidates = [{"chunkId": cid, "distance": float(dist)} for cid, dist in zip(result["ids"][0], result["distances"][0]) if max_distance is None or float(dist) <= max_distance]
    return {"protocolVersion": PROTOCOL_VERSION, "buildId": request["buildId"],
            "embeddingFingerprint": fingerprint, "candidates": candidates}

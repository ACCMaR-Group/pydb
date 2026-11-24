from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit

import yaml

_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

CHUNKING_MODES = ("sentence", "passage", "coarse", "custom", "auto")
DEFAULT_CHUNK_TARGET = 1200
DEFAULT_CHUNK_GAP = 100
COARSE_CHUNK_TARGET = 2400


@dataclass(frozen=True)
class ChunkingConfig:
    mode: str
    target_chars: int | None = None
    max_gap: int | None = None

    def __repr__(self) -> str:
        return f"ChunkingConfig(mode={self.mode!r}, target_chars={self.target_chars!r}, max_gap={self.max_gap!r})"


DEFAULT_CHUNKING = ChunkingConfig("passage", DEFAULT_CHUNK_TARGET, DEFAULT_CHUNK_GAP)


@dataclass(frozen=True, repr=False)
class EmbeddingConfig:
    base_url: str
    model: str
    api_key: str | None = None
    api_key_env: str | None = None
    dimensions: int | None = None
    expected_dimensions: int | None = None
    batch_size: int = 64
    timeout: float = 30.0
    retries: int = 2
    document_prefix: str = ""
    query_prefix: str = ""
    max_distance: float | None = None

    def expected_dim(self) -> int | None:
        """Dimension asserted locally. Never sent to the server."""
        return self.dimensions if self.dimensions is not None else self.expected_dimensions

    def __repr__(self) -> str:
        return (f"EmbeddingConfig(base_url={self.base_url!r}, model={self.model!r}, "
                f"dimensions={self.dimensions!r}, batch_size={self.batch_size!r}, "
                f"timeout={self.timeout!r}, retries={self.retries!r}, max_distance={self.max_distance!r})")


@dataclass(frozen=True)
class Config:
    corpus_root: Path
    paper_ids: list[str] | None
    output_dir: Path
    alias: str
    embedding: EmbeddingConfig | None
    chroma: bool
    max_quarantined_papers: int
    recipe_path: Path
    python_executable: str
    required_queries: list[dict] = None
    resume: bool = True
    chunking: ChunkingConfig = DEFAULT_CHUNKING
    allow_over_budget: bool = False

    def __repr__(self) -> str:
        return (f"Config(corpus_root={self.corpus_root!r}, paper_ids={self.paper_ids!r}, "
                f"output_dir={self.output_dir!r}, alias={self.alias!r}, embedding={self.embedding!r}, "
                f"chroma={self.chroma!r}, recipe_path={self.recipe_path!r})")

    def semantic_dict(self) -> dict:
        emb = None
        if self.embedding:
            emb = {
                "base_url": self.embedding.base_url,
                "model": self.embedding.model,
                "dimensions": self.embedding.dimensions,
                "expected_dimensions": self.embedding.expected_dimensions,
                "batch_size": self.embedding.batch_size,
                "timeout": self.embedding.timeout,
                "retries": self.embedding.retries,
                "document_prefix": self.embedding.document_prefix,
                "query_prefix": self.embedding.query_prefix,
                "max_distance": self.embedding.max_distance,
            }
        return {"corpus_root": str(self.corpus_root), "paper_ids": self.paper_ids,
                "output_dir": str(self.output_dir), "alias": self.alias,
                "embedding": emb, "chroma": self.chroma,
                "max_quarantined_papers": self.max_quarantined_papers,
                "chunking": {"mode": self.chunking.mode, "target_chars": self.chunking.target_chars,
                             "max_gap": self.chunking.max_gap}}


def resolve_api_key(config: EmbeddingConfig) -> str | None:
    if config.api_key_env:
        value = os.environ.get(config.api_key_env)
        if not value:
            raise ValueError("Configured embedding API key environment variable is missing")
        if "\r" in value or "\n" in value:
            raise ValueError("Configured embedding API key contains invalid header characters")
        return value
    if config.api_key is not None and ("\r" in config.api_key or "\n" in config.api_key):
        raise ValueError("Configured embedding API key contains invalid header characters")
    return config.api_key


def _endpoint(value: str) -> str:
    p = urlsplit(value)
    if p.scheme not in {"https", "http"} or not p.hostname:
        raise ValueError("Embedding base_url must be an HTTP(S) URL")
    if p.username or p.password or p.query or p.fragment:
        raise ValueError("Embedding base_url must not contain credentials, query, or fragment")
    if p.scheme != "https" and p.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Embedding base_url must use HTTPS except for loopback")
    return value.rstrip("/")


def _load_embedding(e: dict) -> EmbeddingConfig | None:
    if not isinstance(e, dict):
        raise ValueError("embedding must be a mapping")
    embedding_allowed = {"enabled", "base_url", "model", "api_key", "api_key_env", "dimensions", "expected_dimensions", "batch_size", "timeout", "retries", "document_prefix", "query_prefix", "max_distance"}
    if set(e) - embedding_allowed:
        raise ValueError("embedding contains unknown fields")
    enabled = e.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("embedding.enabled must be boolean")
    if not enabled:
        return None
    if "api_key" in e and "api_key_env" in e:
        raise ValueError("Specify only one embedding API key source")
    key, key_env = e.get("api_key"), e.get("api_key_env")
    if key_env is not None and (not isinstance(key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env)):
        raise ValueError("Invalid embedding API key environment reference")
    if isinstance(key, str):
        match = _ENV_REF.fullmatch(key)
        if match:
            if key_env is not None:
                raise ValueError("Specify only one embedding API key source")
            key_env, key = match.group(1), None
    if key is not None and not isinstance(key, str):
        raise ValueError("Embedding api_key must be a string")
    if key is not None and ("\r" in key or "\n" in key):
        raise ValueError("Configured embedding API key contains invalid header characters")
    if not isinstance(e.get("base_url"), str) or not isinstance(e.get("model"), str) or not e["model"].strip():
        raise ValueError("Embedding base_url and model are required strings")
    dimensions = e.get("dimensions")
    expected_dimensions = e.get("expected_dimensions")
    batch_size, retries = e.get("batch_size", 64), e.get("retries", 2)
    timeout = e.get("timeout", 30)
    if dimensions is not None and (type(dimensions) is not int or dimensions <= 0):
        raise ValueError("Embedding dimensions must be a positive integer")
    if expected_dimensions is not None and (type(expected_dimensions) is not int or expected_dimensions <= 0):
        raise ValueError("Embedding expected_dimensions must be a positive integer")
    if dimensions is not None and expected_dimensions is not None and dimensions != expected_dimensions:
        raise ValueError("embedding.dimensions and expected_dimensions disagree")
    if type(batch_size) is not int or not 1 <= batch_size <= 2048:
        raise ValueError("Invalid embedding batch size")
    if type(retries) is not int or not 0 <= retries <= 5:
        raise ValueError("Invalid embedding retry count")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError("Invalid embedding timeout")
    for name in ("document_prefix", "query_prefix"):
        if name in e and not isinstance(e[name], str):
            raise ValueError(f"embedding.{name} must be a string")
    max_distance = e.get("max_distance")
    if max_distance is not None and (type(max_distance) not in (int, float) or not math.isfinite(max_distance) or not 0 <= max_distance <= 2):
        raise ValueError("embedding.max_distance must be between 0 and 2")
    return EmbeddingConfig(
        base_url=_endpoint(e["base_url"]), model=e["model"], api_key=key,
        api_key_env=key_env, dimensions=dimensions, expected_dimensions=expected_dimensions,
        batch_size=batch_size, timeout=float(timeout), retries=retries,
        document_prefix=e.get("document_prefix", ""), query_prefix=e.get("query_prefix", ""),
        max_distance=float(max_distance) if max_distance is not None else None)


def query_embedding_config(profile: dict, recipe_embedding: EmbeddingConfig) -> EmbeddingConfig:
    allowed = {"base_url", "model", "api_key", "dimensions", "expected_dimensions", "document_prefix", "query_prefix"}
    required = {"base_url", "model", "document_prefix", "query_prefix"}
    if not isinstance(profile, dict) or set(profile) - allowed or required - set(profile):
        raise ValueError("Invalid embedding configuration")
    if "api_key" in profile and not isinstance(profile["api_key"], str):
        raise ValueError("Embedding api_key must be a string")
    raw = {
        "enabled": True,
        "base_url": profile.get("base_url"),
        "model": profile.get("model"),
        "api_key": profile.get("api_key"),
        "dimensions": profile.get("dimensions"),
        "expected_dimensions": recipe_embedding.expected_dimensions,
        "document_prefix": profile.get("document_prefix"),
        "query_prefix": profile.get("query_prefix"),
        "batch_size": recipe_embedding.batch_size,
        "timeout": recipe_embedding.timeout,
        "retries": recipe_embedding.retries,
        "max_distance": recipe_embedding.max_distance,
    }
    if "api_key" not in profile:
        raw.pop("api_key")
    parsed = _load_embedding(raw)
    return replace(parsed, api_key=profile.get("api_key"), api_key_env=None)


def _load_chunking(raw) -> ChunkingConfig:
    if raw is None:
        return DEFAULT_CHUNKING
    if not isinstance(raw, dict):
        raise ValueError("chunking must be a mapping")
    if set(raw) - {"mode", "target_chars", "max_gap"}:
        raise ValueError("chunking contains unknown fields")
    mode = raw.get("mode", "passage")
    if mode not in CHUNKING_MODES:
        raise ValueError("chunking.mode must be sentence, passage, coarse, custom, or auto")
    target, gap = raw.get("target_chars"), raw.get("max_gap")
    if target is not None and (type(target) is not int or target <= 0):
        raise ValueError("chunking.target_chars must be a positive integer")
    if gap is not None and (type(gap) is not int or gap < 0):
        raise ValueError("chunking.max_gap must be a nonnegative integer")
    if mode in {"sentence", "auto"}:
        if target is not None or gap is not None:
            raise ValueError(f"chunking.mode {mode} does not accept target_chars or max_gap")
        return ChunkingConfig(mode)
    if mode == "custom":
        if target is None or gap is None:
            raise ValueError("chunking.mode custom requires target_chars and max_gap")
        return ChunkingConfig(mode, target, gap)
    default_target = DEFAULT_CHUNK_TARGET if mode == "passage" else COARSE_CHUNK_TARGET
    return ChunkingConfig(mode, target if target is not None else default_target,
                          gap if gap is not None else DEFAULT_CHUNK_GAP)


def load_config(path: str | Path) -> Config:
    recipe = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(recipe.read_text(encoding="utf-8")) or {}
    except Exception:
        raise ValueError("Unable to read recipe YAML") from None
    if not isinstance(raw, dict):
        raise ValueError("Recipe must be a YAML mapping")
    allowed = {"corpus_root", "paper_ids", "output_dir", "alias", "embedding", "chroma",
               "max_quarantined_papers", "python_executable", "validation", "resume",
               "chunking", "allow_over_budget"}
    if set(raw) - allowed:
        raise ValueError("Recipe contains unknown fields")
    if not isinstance(raw.get("corpus_root"), str) or not isinstance(raw.get("output_dir"), str):
        raise ValueError("corpus_root and output_dir must be paths")
    base = recipe.parent
    def pathval(key: str) -> Path:
        value = Path(raw[key]).expanduser()
        return (base / value).resolve() if not value.is_absolute() else value.resolve()
    e = raw.get("embedding") or {}
    embedding = _load_embedding(e)
    chroma_raw = raw.get("chroma") or {}
    if not isinstance(chroma_raw, dict) or set(chroma_raw) - {"enabled"}:
        raise ValueError("chroma must contain only enabled")
    chroma_value = chroma_raw.get("enabled", False)
    if type(chroma_value) is not bool:
        raise ValueError("chroma.enabled must be boolean")
    chroma = chroma_value
    if chroma and embedding is None:
        raise ValueError("Chroma requires embeddings to be enabled")
    ids = raw.get("paper_ids")
    if ids is not None and (not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not re.fullmatch(r"paper[0-9]+", item) for item in ids) or len(set(ids)) != len(ids)):
        raise ValueError("paper_ids must be a nonempty list of unique paper-number identifiers")
    alias = raw.get("alias", "science")
    if not isinstance(alias, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", alias):
        raise ValueError("Invalid database alias")
    quarantined = raw.get("max_quarantined_papers", 0)
    if type(quarantined) is not int or quarantined < 0:
        raise ValueError("max_quarantined_papers must be a nonnegative integer")
    interpreter = raw.get("python_executable", sys.executable)
    if not isinstance(interpreter, str) or not interpreter:
        raise ValueError("python_executable must be a nonempty string")
    resume = raw.get("resume", True)
    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    chunking = _load_chunking(raw.get("chunking"))
    allow_over_budget = raw.get("allow_over_budget", False)
    if type(allow_over_budget) is not bool:
        raise ValueError("allow_over_budget must be boolean")
    validation = raw.get("validation") or {}
    if not isinstance(validation, dict) or set(validation) - {"required_queries"}:
        raise ValueError("validation contains unsupported fields")
    required = validation.get("required_queries", [])
    if not isinstance(required, list):
        raise ValueError("validation.required_queries must be an array of query mappings")
    for item in required:
        if (not isinstance(item, dict) or set(item) - {"query", "paper_id", "section"}
                or not isinstance(item.get("query"), str) or not item["query"].strip()
                or not isinstance(item.get("paper_id"), str) or not item["paper_id"].strip()
                or ("section" in item and not isinstance(item["section"], str))):
            raise ValueError("Each required query needs query, paper_id, and optional section")
    return Config(pathval("corpus_root"), list(ids) if ids is not None else None,
                  pathval("output_dir"), alias, embedding, chroma, quarantined, recipe, interpreter, required,
                  resume=resume, chunking=chunking, allow_over_budget=allow_over_budget)

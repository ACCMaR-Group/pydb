from __future__ import annotations

import math
import time
import urllib.error
import urllib.request
import json

from .config import EmbeddingConfig, resolve_api_key


_MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def embed_texts(config: EmbeddingConfig, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    key = resolve_api_key(config)
    result: list[list[float] | None] = [None] * len(texts)
    expected_dimensions = config.dimensions
    for start in range(0, len(texts), config.batch_size):
        batch = texts[start:start + config.batch_size]
        payload = {"model": config.model, "input": batch, "encoding_format": "float"}
        if config.dimensions is not None:
            payload["dimensions"] = config.dimensions
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(config.base_url + "/embeddings", body, headers)
        for attempt in range(config.retries + 1):
            try:
                with _OPENER.open(request, timeout=config.timeout) as response:
                    raw = response.read(_MAX_RESPONSE_BYTES + 1)
                    if len(raw) > _MAX_RESPONSE_BYTES:
                        raise ValueError("Embedding response exceeds size limit")
                    data = json.loads(raw)
                rows = data["data"]
                if len(rows) != len(batch):
                    raise ValueError("Embedding response count mismatch")
                for row in rows:
                    idx = row["index"]
                    vector = row["embedding"]
                    if type(idx) is not int or idx < 0 or idx >= len(batch) or result[start + idx] is not None:
                        raise ValueError("Invalid embedding response ordering")
                    if not isinstance(vector, list) or any(type(x) not in (int, float) for x in vector):
                        raise ValueError("Invalid embedding vector")
                    vector = [float(x) for x in vector]
                    if not vector or not all(math.isfinite(x) for x in vector):
                        raise ValueError("Invalid embedding vector")
                    if expected_dimensions is None:
                        expected_dimensions = len(vector)
                    if len(vector) != expected_dimensions:
                        raise ValueError("Embedding dimension mismatch")
                    result[start + idx] = vector
                if any(x is None for x in result[start:start + len(batch)]):
                    raise ValueError("Embedding response ordering incomplete")
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
                retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code == 429 or exc.code >= 500
                if attempt >= config.retries or not retryable:
                    if isinstance(exc, urllib.error.HTTPError):
                        detail = ""
                        try: detail = exc.read(300).decode("utf-8", "replace").strip().replace("\n", " ")
                        except Exception: pass
                        raise RuntimeError(f"Embedding request failed (HTTP {exc.code})" + (f": {detail}" if detail else "")) from None
                    raise RuntimeError(f"Embedding request failed ({type(exc).__name__}: {exc})") from None
                time.sleep(min(0.25 * (2 ** attempt), 2))
            except Exception:
                raise ValueError("Invalid embedding response") from None
    return [x for x in result if x is not None]

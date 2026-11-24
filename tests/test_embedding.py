import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from pydb.config import EmbeddingConfig, load_config
from pydb.embedding import embed_texts
from pydb.vector import build_vectors


def serve(handler):
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_embedding_local_mock_batches_and_orders():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = [{"index": i, "embedding": [float(len(text)), 1.0]} for i, text in enumerate(payload["input"])][::-1]
            body = json.dumps({"data": data}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass

    server, thread = serve(Handler)
    try:
        cfg = EmbeddingConfig(f"http://127.0.0.1:{server.server_port}/v1", "mock", batch_size=2)
        assert embed_texts(cfg, ["a", "bbb", "cc"]) == [[1.0, 1.0], [3.0, 1.0], [2.0, 1.0]]
    finally:
        server.shutdown(); thread.join(); server.server_close()


@pytest.mark.parametrize("key_config", [
    "api_key: literal-test-secret",
    "api_key: ${PYDB_EXPORTED_TEST_KEY}",
    "api_key_env: PYDB_EXPORTED_TEST_KEY",
])
def test_recipe_key_reaches_local_endpoint_and_stays_out_of_artifacts(tmp_path, monkeypatch, key_config):
    sentinel = "PYDB_SECRET_SENTINEL_7f51"
    monkeypatch.setenv("PYDB_EXPORTED_TEST_KEY", sentinel)
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.headers.get("Authorization"))
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"data": [{"index": 0, "embedding": [1.0, 0.0]}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, *args): pass
    server, thread = serve(Handler)
    try:
        recipe = tmp_path / "input.yaml"
        key_line = key_config.replace("${", "'${").replace("}", "}'") if "${" in key_config else key_config
        recipe.write_text("corpus_root: corpus\noutput_dir: out\nembedding:\n  enabled: true\n  base_url: http://127.0.0.1:%d/v1\n  model: mock\n  %s\n" % (server.server_port, key_line))
        config = load_config(recipe)
        assert embed_texts(config.embedding, ["tiny verified evidence"]) == [[1.0, 0.0]]
        expected = "Bearer " + ("literal-test-secret" if key_config.startswith("api_key: literal") else sentinel)
        assert received == [expected]
        if "api_key: ${" in key_config or "api_key_env:" in key_config:
            monkeypatch.delenv("PYDB_EXPORTED_TEST_KEY", raising=False)
            with pytest.raises(ValueError, match="environment variable is missing") as error:
                embed_texts(config.embedding, ["missing key"])
            assert sentinel not in str(error.value)
            monkeypatch.setenv("PYDB_EXPORTED_TEST_KEY", sentinel)
        db_path = tmp_path / "tiny.sqlite"
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE chunks(id,text,trust_level,section)")
            db.execute("INSERT INTO chunks VALUES('tiny','tiny verified evidence','verified_span','results')")
        metadata = build_vectors(config, db_path, tmp_path / "work", "build-key-test")
        files = [p for p in (tmp_path / "work").rglob("*") if p.is_file()]
        artifacts = json.dumps(metadata) + "".join(p.read_bytes().decode("utf-8", errors="ignore") for p in files)
        assert sentinel not in artifacts
        assert "literal-test-secret" not in artifacts
        assert sentinel not in repr(config) and "literal-test-secret" not in repr(config)
    finally:
        server.shutdown(); thread.join(); server.server_close()


def test_redirect_is_not_followed_or_forwarded_to_target():
    reached = threading.Event()
    class Target(BaseHTTPRequestHandler):
        def do_POST(self):
            reached.set()
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"data":[]}')
        def log_message(self, *args): pass
    target, target_thread = serve(Target)
    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(307)
            self.send_header("Location", f"http://127.0.0.1:{target.server_port}/steal")
            self.end_headers()
        def log_message(self, *args): pass
    proxy, proxy_thread = serve(Redirect)
    try:
        cfg = EmbeddingConfig(f"http://127.0.0.1:{proxy.server_port}/v1", "mock", api_key="secret")
        with pytest.raises(RuntimeError, match="request failed"):
            embed_texts(cfg, ["text"])
        assert not reached.is_set()
    finally:
        proxy.shutdown(); proxy_thread.join(); proxy.server_close()
        target.shutdown(); target_thread.join(); target.server_close()


def test_expected_dimensions_validate_server_vectors():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"data": [{"index": 0, "embedding": [1.0, 2.0]}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, *args): pass
    server, thread = serve(Handler)
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1"
        assert embed_texts(EmbeddingConfig(url, "mock", expected_dimensions=2), ["text"]) == [[1.0, 2.0]]
        with pytest.raises(ValueError, match="Invalid embedding response"):
            embed_texts(EmbeddingConfig(url, "mock", expected_dimensions=3), ["text"])
    finally:
        server.shutdown(); thread.join(); server.server_close()


def test_http_error_reports_status_and_detail_without_key():
    sentinel = "PYDB_SECRET_SENTINEL_HTTP"
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"error": {"message": "Model does not support dimensions"}}).encode()
            self.send_response(400); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, *args): pass
    server, thread = serve(Handler)
    try:
        cfg = EmbeddingConfig(f"http://127.0.0.1:{server.server_port}/v1", "mock", api_key=sentinel, dimensions=4096, retries=0)
        with pytest.raises(RuntimeError) as error:
            embed_texts(cfg, ["text"])
        message = str(error.value)
        assert "HTTP 400" in message and "dimensions" in message
        assert sentinel not in message
    finally:
        server.shutdown(); thread.join(); server.server_close()


@pytest.mark.parametrize("vector", [["1.0"], [True]])
def test_rejects_non_numeric_vectors(vector):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"data": [{"index": 0, "embedding": vector}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, *args): pass
    server, thread = serve(Handler)
    try:
        cfg = EmbeddingConfig(f"http://127.0.0.1:{server.server_port}/v1", "mock")
        with pytest.raises(ValueError, match="Invalid embedding response"):
            embed_texts(cfg, ["first"])
    finally:
        server.shutdown(); thread.join(); server.server_close()


def test_rejects_inconsistent_dimensions_between_batches():
    class Handler(BaseHTTPRequestHandler):
        count = 0
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            type(self).count += 1
            vector = [1.0] if type(self).count == 1 else [1.0, 2.0]
            body = json.dumps({"data": [{"index": 0, "embedding": vector}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, *args): pass
    server, thread = serve(Handler)
    try:
        cfg = EmbeddingConfig(f"http://127.0.0.1:{server.server_port}/v1", "mock", batch_size=1)
        with pytest.raises(ValueError, match="Invalid embedding response"):
            embed_texts(cfg, ["first", "second"])
    finally:
        server.shutdown(); thread.join(); server.server_close()

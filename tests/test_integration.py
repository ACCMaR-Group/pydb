import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from pydb.builder import build
from pydb.config import load_config
from pydb.validation import validate_build


pytest.importorskip("chromadb")
BUN = shutil.which("bun")
SCITURBO_ROOT = Path(os.environ.get("SCITURBO_TEST_ROOT", Path(__file__).resolve().parents[2] / "sciturbo"))
ISOLATED_PYTHON = os.environ.get("SCITURBO_TEST_PYTHON", sys.executable)


def _bundle(root: Path):
    paper = root / "paper1"
    knowledge = paper / "knowledge"
    knowledge.mkdir(parents=True)
    quote = "The thermal response improves at 42 C."
    (paper / "main.tex").write_text(quote, encoding="utf-8")
    entries = {
        "manifest.json": {"schema_version": 4, "document_id": "integration-doc"},
        "document.json": {"doi": "10.1234/integration", "title": "Thermal study"},
        "sections.json": {"sections": [{"id": "results", "title": "Results", "page_range": {"start": 1, "end": 1}}]},
        "quality.json": {},
        "retrieval.json": {},
        "inventory.json": {},
        "coherence.json": {},
    }
    for name, value in entries.items():
        (knowledge / name).write_text(json.dumps(value), encoding="utf-8")
    return quote


def test_full_local_embedding_chroma_and_bun_hybrid(tmp_path, monkeypatch):
    if not BUN or not SCITURBO_ROOT.is_dir():
        pytest.skip("Bun or the sciturbo checkout is unavailable")
    secret = "PYDB_INTEGRATION_SECRET_3a6e"
    monkeypatch.setenv("PYDB_INTEGRATION_KEY", secret)
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.headers.get("Authorization"), body["input"]))
            vectors = [{"index": i, "embedding": [0.0, 1.0] if text.strip() == "Results" else [1.0, 0.0]} for i, text in enumerate(body["input"])]
            response = json.dumps({"data": vectors}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(response))); self.end_headers(); self.wfile.write(response)
        def log_message(self, *args): pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        corpus = tmp_path / "corpus"
        quote = _bundle(corpus)
        recipe = tmp_path / "input.yaml"
        recipe.write_text(
            f"corpus_root: {corpus}\noutput_dir: {tmp_path / 'builds'}\nalias: science\n"
            "max_quarantined_papers: 0\nembedding:\n  enabled: true\n"
            f"  base_url: http://127.0.0.1:{server.server_port}/v1\n"
            "  model: deterministic-local\n  api_key_env: PYDB_INTEGRATION_KEY\n  max_distance: 0.1\n"
            "chroma:\n  enabled: true\n"
        )
        config = load_config(recipe)
        report = build(config)
        assert report["complete"] and report["vectorSearch"]["enabled"]
        assert validate_build(report["databasePath"])["valid"]
        assert requests and all(auth == f"Bearer {secret}" for auth, _ in requests)
        first_request_count = len(requests)
        repeat = build(config)
        assert repeat["buildId"] == report["buildId"]
        assert len(requests) == first_request_count
        build_dir = Path(report["databasePath"]).parent
        files = [p for p in build_dir.rglob("*") if p.is_file()]
        serialized = "".join(p.read_bytes().decode("utf-8", errors="ignore") for p in files)
        assert secret not in serialized
        with sqlite3.connect(report["databasePath"]) as db:
            row = db.execute("SELECT id,text,start_offset,end_offset,source_hash FROM chunks WHERE trust_level='verified_span' AND text LIKE '%thermal%'").fetchone()
            build_id = json.loads(db.execute("SELECT value FROM database_meta WHERE key='build_id'").fetchone()[0])
            vector_meta = json.loads(db.execute("SELECT value FROM database_meta WHERE key='vector_search'").fetchone()[0])
        assert row and quote in row[1] and row[2] < row[3]
        assert "api_key" not in json.dumps(vector_meta) and secret not in repr(config)
        config_home = tmp_path / "isolated-config-home"
        publish = subprocess.run([ISOLATED_PYTHON, "-m", "pydb", "publish", str(build_dir), "--sciturbo-root", str(SCITURBO_ROOT), "--config-home", str(config_home)],
                                 cwd=SCITURBO_ROOT, env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "SCITURBO_CONFIG_DIR": str(config_home)},
                                 capture_output=True, text=True, timeout=60, check=False)
        assert publish.returncode == 0, publish.stderr
        published_dir = config_home / "databases" / "science" / "builds" / report["buildId"]
        assert validate_build(published_dir)["valid"]
        env = os.environ.copy()
        env["SCITURBO_CONFIG_DIR"] = str(config_home)
        env["SCITURBO_PYDB_PYTHON"] = ISOLATED_PYTHON
        env["SCITURBO_PYDB_RECIPE"] = str(recipe)
        script = (
            "import { Database } from 'bun:sqlite'; "
            "import { searchEvidence, verifySearchEvidence } from './src/science/search.ts'; "
            f"const db=new Database({json.dumps(report['databasePath'])},{{readonly:true}}); "
            "const result=verifySearchEvidence(db,searchEvidence(db,{query:'temperature sensitivity shifts',limit:3})); "
            "console.log(JSON.stringify({result,buildId:db.query(\"select value from database_meta where key='build_id'\").get()}));db.close();"
        )
        completed = subprocess.run([BUN, "-e", script], cwd=SCITURBO_ROOT, env=env,
                                   capture_output=True, text=True, timeout=45, check=False)
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        assert result["result"]["results"][0]["chunkId"] == row[0]
        assert result["result"]["results"][0].get("provenance") == "vector"
        assert result["result"]["results"][0]["text"] == quote
        assert result["result"]["results"][0]["evidence"][0]["quote"] == quote
        assert result["result"]["results"][0]["evidence"][0]["sourceHash"] == row[4]
        assert secret not in completed.stdout and secret not in completed.stderr
        assert build_id == report["buildId"]
        published_script = script.replace(json.dumps(report["databasePath"]), json.dumps(str(published_dir / Path(report["databasePath"]).name)))
        published_run = subprocess.run([BUN, "-e", published_script], cwd=SCITURBO_ROOT, env=env,
                                        capture_output=True, text=True, timeout=45, check=False)
        assert published_run.returncode == 0, published_run.stderr
        published_result = json.loads(published_run.stdout.strip().splitlines()[-1])
        assert published_result["result"]["results"][0].get("provenance") == "vector"
        assert published_result["result"]["results"][0]["chunkId"] == row[0]
    finally:
        server.shutdown(); thread.join(); server.server_close()

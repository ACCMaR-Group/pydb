from __future__ import annotations

import argparse
import json
import importlib.util
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from .builder import build
from .config import load_config
from .estimation import plan_estimate
from .validation import validate_build


def _preflight(config) -> dict:
    root = config.corpus_root.resolve()
    if not root.is_dir():
        raise ValueError("Corpus root must be an existing directory")
    all_ids = sorted((p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("paper") and p.name[5:].isdigit()), key=lambda value: int(value[5:]))
    selected = config.paper_ids if config.paper_ids is not None else all_ids
    if not selected:
        raise ValueError("Corpus contains no selected paper directories")
    for paper_id in selected:
        paper = (root / paper_id).resolve()
        if not paper.is_dir() or not paper.is_relative_to(root):
            raise ValueError("Selected paper directory is missing or escapes corpus root")
    db = sqlite3.connect(":memory:")
    try:
        db.execute("CREATE VIRTUAL TABLE preflight_fts USING fts5(text)")
    except sqlite3.Error:
        raise ValueError("SQLite FTS5 is unavailable") from None
    finally:
        db.close()
    chroma = {"requested": config.chroma, "available": None}
    if config.chroma:
        chroma["available"] = importlib.util.find_spec("chromadb") is not None
        if not chroma["available"]:
            raise ValueError("Chroma is enabled but chromadb is not installed")
    return {
        "corpus": {"path": str(root), "valid": True, "selectedPapers": len(selected)},
        "staging": {"outputDir": str(config.output_dir), "alias": config.alias, "willWrite": False},
        "sqlite": {"fts5Available": True},
        "embedding": {"enabled": config.embedding is not None, "keyConfigured": bool(config.embedding and (config.embedding.api_key is not None or config.embedding.api_key_env is not None))},
        "chroma": chroma,
        "retrievalGates": len(config.required_queries or []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pydb")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "build"):
        p = sub.add_parser(name); p.add_argument("recipe")
    for name in ("validate", "publish"):
        p = sub.add_parser(name); p.add_argument("build")
        if name == "publish":
            p.add_argument("--sciturbo-root", required=True)
            p.add_argument("--config-home")
            p.add_argument("--in-place", action="store_true")
            p.add_argument("--root", action="append", default=[])
    p = sub.add_parser("query")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            config = load_config(args.recipe)
            result = {"corpusRoot": str(config.corpus_root), "paperIds": config.paper_ids, "outputDir": str(config.output_dir), "alias": config.alias, "recipePath": str(config.recipe_path), **_preflight(config), "planning": plan_estimate(config, config.corpus_root, config.paper_ids)}
            print(json.dumps(result, indent=2))
        elif args.command == "build":
            print(json.dumps(build(load_config(args.recipe)), indent=2))
        elif args.command == "validate":
            result = validate_build(args.build)
            print(json.dumps(result, indent=2)); return 0 if result["valid"] else 1
        elif args.command == "query":
            from .vector import query_candidates
            payload = sys.stdin.buffer.read(65537)
            if len(payload) > 65536: raise ValueError("Vector query request is too large")
            request = json.loads(payload)
            print(json.dumps(query_candidates(request), separators=(",", ":")))
        elif args.command == "publish":
            result = validate_build(args.build)
            print(json.dumps(result, indent=2))
            if not result["valid"]: return 1
            root = Path(args.sciturbo_root).resolve(strict=True)
            script = root / "scripts/import-science-db.ts"
            if not script.is_file(): raise ValueError("Sciturbo external import module not found")
            command = ["bun", str(script), str(Path(args.build).resolve())]
            if args.config_home: command.extend(["--config-home", str(Path(args.config_home).resolve())])
            if args.in_place:
                command.append("--in-place")
                for root in args.root: command.extend(["--root", str(Path(root).resolve())])
            elif args.root:
                raise ValueError("--root requires --in-place")
            # Import copies the database, checkpoint, and vector index; a large
            # build can take many minutes. Do not use a short fixed timeout.
            proc = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=3600, check=False)
            if proc.returncode:
                detail = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")
                raise RuntimeError("Sciturbo publication failed" + (f": {detail[:500]}" if detail else ""))
            print(proc.stdout)
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"pydb: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"pydb: command failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

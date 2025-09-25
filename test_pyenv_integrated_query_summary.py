#!/usr/bin/env python3
"""
Test script for the integrated enhanced query execution with summarization.
Pure venv/pyenv mode (no conda).

Behavior:
  • If a project venv exists (default: ./.venv/llcpp) and we're not already using it,
    re-exec this script with that venv's python BEFORE importing heavy deps.
  • Otherwise, run with the current Python.

Usage (from project root):
  python test_integrated_query_summary.py
  python test_integrated_query_summary.py --venv-path .venv\\llcpp
  python test_integrated_query_summary.py --no-auto-venv
  python test_integrated_query_summary.py --query "EV battery makers in Pune"
"""

from __future__ import annotations
import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

# ------------------------------ Venv helpers ------------------------------

def infer_project_root() -> Path:
    return Path(__file__).resolve().parent

def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python3"

def venv_exists(venv_dir: Path) -> bool:
    return venv_dir.is_dir() and venv_python(venv_dir).exists()

def already_using_python(py_exe: Path) -> bool:
    try:
        return Path(sys.executable).resolve() == py_exe.resolve()
    except Exception:
        return False

def reexec_with_python(python_exe: Path) -> None:
    """
    Re-exec the current script with the given python, forwarding CLI args.
    We also set PYTHONPATH=project_root for clean relative imports.
    """
    project_root = infer_project_root()
    # Filter out venv control flags so we don't loop on re-exec
    filtered: list[str] = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a in ("--venv-path",):
            skip_next = True
            continue
        if a.startswith("--venv-path=") or a == "--no-auto-venv":
            continue
        filtered.append(a)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(project_root))
    cmd = [str(python_exe), str(Path(__file__).resolve()), *filtered]
    print(f"🔁 Re-executing inside venv: {python_exe}")
    rc = subprocess.call(cmd, env=env, cwd=str(project_root))
    sys.exit(rc)

# ------------------------------ CLI & bootstrap ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run enhanced query execution with summarization (venv-aware).")
    p.add_argument("--query", default="small scale electrical companies", help="Query text to run.")
    p.add_argument("--config", default=None, help="Path to config JSON (if your load_config supports it).")
    p.add_argument("--no-llm-validation", action="store_true", help="Disable LLM validation.")

    # Venv controls
    default_venv = infer_project_root() / ".venv" / "llcpp"
    p.add_argument("--venv-path", default=str(default_venv), help="Path to a project venv (default: .venv/llcpp).")
    p.add_argument("--no-auto-venv", action="store_true",
                   help="Do not auto-reexec into the venv even if it exists.")
    return p.parse_args()

def ensure_venv_exec(args: argparse.Namespace) -> None:
    """
    If a venv exists at --venv-path and we're not using it, re-exec with that venv's python.
    """
    if args.no_auto_venv:
        return
    venv_dir = Path(args.venv_path)
    if not venv_exists(venv_dir):
        return
    py = venv_python(venv_dir)
    if not already_using_python(py):
        reexec_with_python(py)

# ------------------------------ Main ------------------------------

def main() -> None:
    args = parse_args()

    # Ensure we're in the right venv BEFORE importing project modules
    ensure_venv_exec(args)

    # Import AFTER env is ensured (so pandas & friends come from the venv)
    from src.query_engine.enhanced_query_with_summary import execute_enhanced_query_with_summary
    from src.load_config import load_config

    project_root = infer_project_root()

    # Load configuration/logger
    try:
        if args.config:
            (config, logger) = load_config(args.config)
        else:
            (config, logger) = load_config()
    except TypeError:
        (config, logger) = load_config()

    enable_llm_validation = not args.no_llm_validation
    test_query = args.query

    logger.debug("🧪 TESTING INTEGRATED QUERY EXECUTION WITH SUMMARIZATION")
    logger.debug("=" * 80)
    logger.info(f"Test Query: '{test_query}'")
    # Show env info
    logger.debug(f"🐍 sys.executable: {sys.executable}")
    logger.debug(f"🌱 VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'none')}")
    logger.debug(f"📦 PYTHONPATH: {os.environ.get('PYTHONPATH', 'none')}")
    logger.debug("=" * 80)

    try:
        result = execute_enhanced_query_with_summary(
            user_query=test_query,
            config=config,
            logger=logger,
        )

        logger.info("\n🎯 FINAL INTEGRATED RESULTS:")

        if isinstance(result, dict) and "error" in result:
            logger.error(f"❌ Query failed: {result['error']}")
        else:
            logger.debug("✅ Query executed successfully")
            if isinstance(result, dict):
                logger.info(f" Confidence: {result.get('confidence', 0):.2f}")
                if result.get('results'):
                    results = result['results']
                    logger.info(f" Companies: {results.get('companies_count', 0)}")

                # Show enhanced summary
                if result.get('enhanced_summary'):
                    logger.info(f"\n Enhanced Summary:")
                    print(result['enhanced_summary'])

                # Show intent answer (the detailed company information)
                if result.get('intent_answer'):
                    logger.info(f"\n Intent Answer:")
                    print(result['intent_answer'])

        logger.info("\n✅ Integration test completed successfully!")

    except Exception as e:
        import traceback
        logger.error(f"❌ Integration test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()

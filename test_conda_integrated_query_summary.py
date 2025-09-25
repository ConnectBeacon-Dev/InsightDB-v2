#!/usr/bin/env python3
"""
Test script for the integrated enhanced query execution with summarization.
Auto-detects a local conda env under ./.conda and re-executes itself inside it
using `conda run -p <prefix>` BEFORE importing heavy deps (like pandas).

Usage (from project root):
  python test_integrated_query_summary.py
  # or explicitly:
  python test_integrated_query_summary.py --conda-prefix .conda\\llcpp
  python test_integrated_query_summary.py --conda-name llcpp
  python test_integrated_query_summary.py --query "EV battery makers in Pune"
"""

from __future__ import annotations
import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional


# ------------------------------ Conda helpers ------------------------------

def find_conda_exe() -> Optional[str]:
    exe = shutil.which("conda")
    if exe:
        return exe
    # common Windows shells may need this hook sourced; we still just report nicely
    return None


def norm(p: Path) -> str:
    return str(p.resolve())


def infer_project_root() -> Path:
    # Project root is the folder containing THIS file.
    return Path(__file__).resolve().parent


def detect_local_conda_prefix(preferred_name: str = "llcpp") -> Optional[Path]:
    """
    Look for a local conda env under ./.conda/** with preference to 'llcpp'.
    Returns the prefix path or None if not found.
    """
    root = infer_project_root()
    conda_dir = root / ".conda"
    if not conda_dir.is_dir():
        return None

    # Prefer llcpp
    preferred = conda_dir / preferred_name
    if preferred.is_dir():
        return preferred

    # Otherwise pick the first child directory
    for child in sorted(conda_dir.iterdir()):
        if child.is_dir():
            return child
    return None


def already_in_prefix(prefix: Path) -> bool:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return False
    try:
        return Path(conda_prefix).resolve() == prefix.resolve()
    except Exception:
        return False


def reexec_in_conda_prefix(prefix: Path) -> None:
    """
    Re-exec the current script inside the given conda prefix using:
      conda run -p <prefix> python <script> [filtered args]
    """
    conda_exe = find_conda_exe()
    if not conda_exe:
        print(
            "❌ Could not find `conda` on PATH.\n"
            "   Open an Anaconda/Miniconda PowerShell prompt or ensure your shell has Conda available.\n"
            "   Example:\n"
            "     PowerShell: & $Env:USERPROFILE\\miniconda3\\shell\\condabin\\conda-hook.ps1 | Out-Null\n",
            file=sys.stderr,
        )
        sys.exit(2)

    # Filter out conda control flags to avoid recursion
    filtered = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a in ("--conda-prefix", "--conda-name"):
            skip_next = True
            continue
        if a.startswith("--conda-prefix=") or a.startswith("--conda-name="):
            continue
        filtered.append(a)

    cmd = [
        conda_exe, "run", "-p", norm(prefix),  # use prefix (works even if env has no registered name)
        "python", norm(Path(__file__)),
        *filtered,
    ]
    print(f"🔁 Re-executing inside conda env: {norm(prefix)}")
    # Optional: echo command
    # print("CMD:", " ".join(cmd))

    rc = subprocess.call(cmd)
    sys.exit(rc)


# ------------------------------ CLI & bootstrap ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run enhanced query execution with summarization.")
    p.add_argument("--query", default="small scale electrical companies", help="Query text to run.")
    p.add_argument("--config", default=None, help="Path to config JSON (if your load_config supports it).")
    p.add_argument("--no-llm-validation", action="store_true", help="Disable LLM validation.")

    # Conda controls
    p.add_argument("--conda-prefix", default=None, help="Full path to a conda env prefix.")
    p.add_argument("--conda-name", default=None, help="Conda env name to use (if on this machine).")
    p.add_argument("--no-auto-conda", action="store_true",
                   help="Do not auto-detect ./.conda/** even if present.")

    return p.parse_args()


def ensure_conda(args: argparse.Namespace) -> None:
    """
    Decide which env (if any) to use and re-exec inside it if we're not already there.
    Priority:
      1) --conda-prefix (path)
      2) --conda-name (name)
      3) auto-detect ./.conda/** (prefer 'llcpp'), unless --no-auto-conda
      4) else do nothing (run in current Python)
    """
    # 1) Explicit prefix
    if args.conda_prefix:
        prefix = Path(args.conda_prefix)
        if not prefix.exists():
            print(f"❌ Provided --conda-prefix does not exist: {prefix}", file=sys.stderr)
            sys.exit(2)
        if not already_in_prefix(prefix):
            reexec_in_conda_prefix(prefix)
        return

    # 2) Named env
    if args.conda_name:
        # Convert name -> prefix via `conda env list` is overkill; just rely on conda run -n
        # However, we still want to avoid recursion if we're already in it.
        current_name = os.environ.get("CONDA_DEFAULT_ENV")
        if current_name != args.conda_name:
            conda_exe = find_conda_exe()
            if not conda_exe:
                print("❌ `conda` not found for --conda-name.", file=sys.stderr)
                sys.exit(2)
            # re-exec via name
            filtered = []
            skip_next = False
            for a in sys.argv[1:]:
                if skip_next:
                    skip_next = False
                    continue
                if a in ("--conda-prefix", "--conda-name"):
                    skip_next = True
                    continue
                if a.startswith("--conda-prefix=") or a.startswith("--conda-name="):
                    continue
                filtered.append(a)
            cmd = [conda_exe, "run", "-n", args.conda_name, "python", str(Path(__file__).resolve()), *filtered]
            print(f"🔁 Re-executing inside conda env (name): {args.conda_name}")
            rc = subprocess.call(cmd)
            sys.exit(rc)
        return

    # 3) Auto-detect ./.conda/** unless disabled
    if not args.no_auto_conda:
        prefix = detect_local_conda_prefix(preferred_name="llcpp")
        if prefix and not already_in_prefix(prefix):
            reexec_in_conda_prefix(prefix)


def main() -> None:
    args = parse_args()

    # Ensure we're inside the right conda env BEFORE we import project modules
    ensure_conda(args)

    # Import AFTER env is ensured (so pandas is available from the env)
    from src.query_engine.enhanced_query_with_summary import execute_enhanced_query_with_summary
    from src.load_config import load_config

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
    logger.debug(f"📦 CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'none')}")
    logger.debug(f"📦 CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', 'none')}")
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
                logger.info(f"📊 Confidence: {result.get('confidence', 0):.2f}")
                if result.get('results'):
                    results = result['results']
                    logger.info(f"📈 Companies: {results.get('companies_count', 0)}")
                
                # Show enhanced summary
                if result.get('enhanced_summary'):
                    logger.info(f"\n📋 Enhanced Summary:")
                    print(result['enhanced_summary'])
                
                # Show intent answer (the detailed company information)
                if result.get('intent_answer'):
                    logger.info(f"\n🎯 Intent Answer:")
                    print(result['intent_answer'])

        logger.info("\n✅ Integration test completed successfully!")

    except Exception as e:
        import traceback
        logger.error(f"❌ Integration test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()

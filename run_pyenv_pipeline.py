#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pyenv_pipeline.py — chain the InsightDB-v2 steps using your local venv (pure pyenv/venv).

It runs, in order:
  1) -m src.load_config
  2) -m src.company_index.create_integrated_search_file
  3) -m src.company_index.company_tfidf_api
  4) -m src.query_engine.enhanced_query_with_summary

Behavior:
- If already inside the venv (.venv/llcpp), steps run directly with sys.executable.
- Otherwise, each step is launched through: python bootstrap_pyenv_local.py run -- <step>
- Logs go to logs/pyenv_pipeline.log
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
BOOTSTRAP    = PROJECT_ROOT / "bootstrap_pyenv_local.py"
VENV_PY      = PROJECT_ROOT / ".venv" / "llcpp" / ("Scripts/python.exe" if os.name == "nt" else "bin/python3")

LOG_DIR  = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "pyenv_pipeline.log"

# Run each step as a module (this matched what worked for you)
STEPS = [
    ["-m", "src.load_config"],
    ["-m", "src.company_index.create_integrated_search_file"],
    ["-m", "src.company_index.company_tfidf_api"],
    ["-m", "src.query_engine.enhanced_query_with_summary"],
]

def in_target_venv() -> bool:
    try:
        if VENV_PY.exists() and Path(sys.executable).resolve() == VENV_PY.resolve():
            return True
    except Exception:
        pass
    ve = os.environ.get("VIRTUAL_ENV")
    return bool(ve and (PROJECT_ROOT / ".venv") in Path(ve).parents)

def build_runner() -> list[str]:
    # If already in venv, use the current interpreter; else delegate to bootstrap.
    if in_target_venv():
        return [sys.executable]
    if not BOOTSTRAP.exists():
        print(f"❌ bootstrap not found: {BOOTSTRAP}")
        sys.exit(2)
    return [sys.executable, str(BOOTSTRAP), "run", "--"]

def run_step(cmd: list[str], logf) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    line = " ".join(cmd)
    print(line)
    logf.write(f"\n>>> {line}\n")
    logf.flush()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return proc.returncode

def main():
    runner = build_runner()

    with LOG_FILE.open("a", encoding="utf-8") as logf:
        logf.write("\n" + "=" * 70 + "\n")
        logf.write(f"Pyenv pipeline run @ {datetime.now()}\n")
        logf.write(f"Runner: {' '.join(runner)}\n")
        logf.write("=" * 70 + "\n")

        for i, step in enumerate(STEPS, 1):
            cmd = runner + step
            short = " ".join(step)
            print(f"[{i}/{len(STEPS)}] {short}")
            logf.write(f"[{i}/{len(STEPS)}] {' '.join(cmd)}\n")
            rc = run_step(cmd, logf)
            if rc != 0:
                print(f"❌ Step failed (exit {rc})")
                logf.write(f"FAILED at step {i} (exit {rc})\n")
                sys.exit(rc)

        print("✅ Pipeline finished successfully.")
        logf.write("SUCCESS\n")

if __name__ == "__main__":
    main()

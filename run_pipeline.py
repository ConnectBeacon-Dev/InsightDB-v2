#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automation pipeline for InsightDB-v2.

Runs the following inside the .conda/llcpp environment:
  1. python src/load_config.py
  2. python -m src.input_data_processing.generate_grouped_csvs_with_data
  3. python -m src.company_index.company_search_api create
  4. python -m src.query_engine.enhanced_query_with_summary

All stdout/stderr are appended to logs/automation.log
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_DIR = PROJECT_ROOT / ".conda" / "llcpp"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "automation.log"

# Try to detect conda.exe
def detect_conda_exe():
    candidates = [
        Path("C:/ProgramData/miniconda3/Scripts/conda.exe"),
        Path("C:/ProgramData/Miniconda3/Scripts/conda.exe"),
        Path.home() / "Miniconda3" / "Scripts" / "conda.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # fallback to PATH
    try:
        out = subprocess.run(["where", "conda"], check=True,
                             capture_output=True, text=True)
        path = Path(out.stdout.strip().splitlines()[0])
        return str(path)
    except Exception:
        return None

CONDA_EXE = detect_conda_exe()
if not CONDA_EXE:
    print("❌ Could not detect conda.exe. Please install Miniconda/Conda.")
    sys.exit(1)

COMMANDS = [
    ["python", "src/load_config.py"],
    ["python", "-m", "src.input_data_processing.generate_grouped_csvs_with_data"],
    ["python", "-m", "src.company_index.company_search_api", "create"],
    ["python", "-m", "src.query_engine.enhanced_query_with_summary"],
]

def run_in_env(cmd, logf):
    """Run a command inside the project-local conda env."""
    full_cmd = [CONDA_EXE, "run", "-p", str(ENV_DIR)] + cmd
    logf.write(f"\n>>> {' '.join(full_cmd)}\n")
    logf.flush()
    proc = subprocess.run(full_cmd, stdout=logf, stderr=logf)
    return proc.returncode

def main():
    with open(LOG_FILE, "a", encoding="utf-8") as logf:
        logf.write("\n" + "="*50 + "\n")
        logf.write(f"Pipeline run {datetime.now()}\n")
        logf.write("="*50 + "\n")
        logf.flush()

        for idx, cmd in enumerate(COMMANDS, start=1):
            step = f"[{idx}/{len(COMMANDS)}] {' '.join(cmd)}"
            print(step)
            logf.write(step + "\n")
            rc = run_in_env(cmd, logf)
            if rc != 0:
                print(f"❌ Step failed: {' '.join(cmd)} (exit {rc})")
                logf.write(f"FAILED at step {idx} (exit {rc})\n")
                sys.exit(rc)

        print("✅ Pipeline finished successfully.")
        logf.write("SUCCESS\n")

if __name__ == "__main__":
    main()

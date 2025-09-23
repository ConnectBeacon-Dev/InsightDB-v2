#!/usr/bin/env python
# Run me from inside InsightDB-v2
# Creates a project-local conda env at .conda/llcpp and installs everything.

import os, sys, shutil, subprocess
from pathlib import Path

PROJECT_ROOT = Path.cwd()
REQ = PROJECT_ROOT / "requirements.txt"
if not REQ.exists():
    print("❌ requirements.txt not found. Run this from inside your InsightDB-v2 folder.")
    sys.exit(1)

ENV_DIR = PROJECT_ROOT / ".conda" / "llcpp"       # project-local env path
PY_VER = "3.11"
CHANNEL = "conda-forge"                           # avoids Anaconda TOS

QWEN_FILE = PROJECT_ROOT / "models" / "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
QWEN_FETCH = ["python", "models/get_qwen_model.py"]
ST_DIR = PROJECT_ROOT / "models" / "sentence-transformers" / "sentence-transformers_all-mpnet-base-v2"
ST_FETCH = ["python", "models/get_sentence_transformer_models.py"]

def run(cmd, check=True, capture=False):
    print(f"➡️  {' '.join(str(c) for c in cmd)}")
    if capture:
        p = subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(p.stdout, end="")
        return p.stdout
    else:
        return subprocess.run(cmd, check=check)

def detect_conda_exe():
    # 1) Ask PATH
    try:
        base = run(["conda", "info", "--base"], capture=True).strip().splitlines()[-1].strip()
    except Exception:
        base = ""
    candidates = []
    if base:
        if os.name == "nt":
            candidates.append(Path(base) / "Scripts" / "conda.exe")
        else:
            candidates.append(Path(base) / "bin" / "conda")
    # 2) Common Windows/system paths
    if os.name == "nt":
        candidates += [
            Path(os.environ.get("USERPROFILE", "")) / "Miniconda3" / "Scripts" / "conda.exe",
            Path("C:/ProgramData/miniconda3/Scripts/conda.exe"),
            Path("C:/ProgramData/Miniconda3/Scripts/conda.exe"),
        ]
    # 3) Last resort: where/which
    if os.name == "nt":
        try:
            out = run(["where", "conda"], capture=True)
            for line in out.splitlines():
                p = Path(line.strip())
                if p.exists():
                    candidates.append(p)
        except Exception:
            pass
    else:
        try:
            out = run(["which", "conda"], capture=True).strip()
            if out:
                candidates.append(Path(out))
        except Exception:
            pass

    for c in candidates:
        if c and c.exists():
            return str(c)
    return None

def force_rmtree(p: Path):
    if not p.exists():
        return
    def onerror(func, path, exc_info):
        try:
            os.chmod(path, 0o700)
        except Exception:
            pass
        try:
            func(path)
        except Exception:
            pass
    shutil.rmtree(p, onerror=onerror)

def main():
    conda_exe = detect_conda_exe()
    if not conda_exe:
        print("❌ Could not find conda. Please install Miniconda (user or system) first.")
        sys.exit(1)
    print(f"✅ Using Conda: {conda_exe}")

    # Fresh local env
    print(f"\n== Creating project-local env at {ENV_DIR} ==")
    # Clean any leftovers (safe; inside your repo)
    force_rmtree(ENV_DIR.parent)  # removes .conda entirely if present
    ENV_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Create env with python
    run([conda_exe, "create", "-p", str(ENV_DIR), "-c", CHANNEL, f"python={PY_VER}", "-y"])

    # Install llama-cpp-python via conda (prebuilt)
    run([conda_exe, "install", "-p", str(ENV_DIR), "-c", CHANNEL, "llama-cpp-python", "-y"])

    # Pip requirements inside the env (no activation needed)
    run([conda_exe, "run", "-p", str(ENV_DIR), "python", "-m", "pip", "install", "-r", str(REQ)])

    # Models
    #if not QWEN_FILE.exists():
    #    print("• Qwen model missing → fetching...")
    #    run([conda_exe, "run", "-p", str(ENV_DIR)] + QWEN_FETCH)
    #else:
    #    print(f"• Found Qwen model: {QWEN_FILE}")

    #if not ST_DIR.exists():
    #    print("• Sentence-Transformer missing → fetching...")
    #    run([conda_exe, "run", "-p", str(ENV_DIR)] + ST_FETCH)
    #else:
    #    print(f"• Found Sentence-Transformer: {ST_DIR}")

    # Smoke test
    run([conda_exe, "run", "-p", str(ENV_DIR), "python", "-c",
         "import llama_cpp,sys; print('llama-cpp-python OK', sys.version)"])

    print("\n🎉 All set.")
    print("Run your app (no activation needed):")
    #print(f'  "{conda_exe}" run -p "{ENV_DIR}" python src\\company_index\\company_search_api.py create')
    print("\nIf you prefer activation for an interactive session:")
    if os.name == "nt":
        base = Path(conda_exe).parent.parent
        print(f'  call "{base}\\condabin\\conda.bat" activate "{ENV_DIR}"')
    else:
        print(f'  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "{ENV_DIR}"')

if __name__ == "__main__":
    main()

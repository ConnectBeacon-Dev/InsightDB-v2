import os
import sys
import subprocess
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
BASE_DIR                = Path.cwd()
VENV_PYTHON             = BASE_DIR / "venv" / "Scripts" / "python.exe"  # after setup_env.ps1

PS_SETUP_ENV            = Path("setup_env.ps1")
PS_SETUP_LLAMA_MODELS   = Path("tools") / "setup_llama_and_models.ps1"

PY_GET_ST_MODELS        = Path("models") / "get_sentence_transformer_models.py"
PY_GET_QWEN_MODELS      = Path("models") / "get_qwen_model.py"

QWEN_GGUF_PATH          = Path("models") / "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
ST_DIR_PATH             = Path("models") / "sentence-transformers" / "sentence-transformers_all-mpnet-base-v2"

PY_LOAD_CONFIG          = Path("src") / "load_config.py"
PY_GEN_GROUPED_CSVS_MOD = "src.input_data_processing.generate_grouped_csvs_with_data"
PY_TEST_GENERATED_CSVS  = Path("test_generated_csvs.py")
PY_COMPANY_INDEX_CREATE = Path("src") / "company_index" / "company_search_api.py"

# ----------------------------
# Helpers
# ----------------------------
def run_powershell(script_path: Path) -> None:
    print(f"\n[PS] Running: {script_path}")
    result = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"PowerShell script failed: {script_path}")

def run_python(pyfile_or_mod, args=None, as_module=False) -> None:
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Virtual env python not found: {VENV_PYTHON}")

    args = args or []
    if as_module:
        cmd = [str(VENV_PYTHON), "-m", str(pyfile_or_mod), *args]
        shown = f"{VENV_PYTHON} -m {pyfile_or_mod} " + " ".join(args)
    else:
        cmd = [str(VENV_PYTHON), str(pyfile_or_mod), *args]
        shown = f"{VENV_PYTHON} {pyfile_or_mod} " + " ".join(args)

    print(f"\n[PY] Running: {shown}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Python step failed: {shown} (exit={result.returncode})")

def run_venv_python(args: list[str], show: str) -> None:
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Virtual env python not found: {VENV_PYTHON}")
    print(f"\n[PY] Running: {show}")
    result = subprocess.run([str(VENV_PYTHON), *args], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Python step failed: {show} (exit={result.returncode})")

def run_python_file(pyfile: Path, args: list[str] | None = None) -> None:
    if not pyfile.exists():
        raise FileNotFoundError(f"Python file not found: {pyfile}")
    args = args or []
    shown = f"{VENV_PYTHON} {pyfile} " + " ".join(args)
    run_venv_python([str(pyfile), *args], shown)
    
def run_python_module(module_path: str, args: list[str] | None = None) -> None:
    """Run a Python module (inside venv) using -m package.module syntax."""
    args = args or []
    shown = f"{VENV_PYTHON} -m {module_path} " + " ".join(args)
    print(f"\n[PY] Running: {shown}")
    result = subprocess.run([str(VENV_PYTHON), "-m", module_path, *args], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Python step failed: {shown} (exit={result.returncode})")

def dir_has_st_model(path: Path) -> bool:
    return (path / "config.json").exists() and (
        (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()
    )

# ----------------------------
# Main flow
# ----------------------------
def main():
    # Installation
    run_powershell(PS_SETUP_ENV)
    run_powershell(PS_SETUP_LLAMA_MODELS)

    # Sentence-Transformer models
    run_python(PY_GET_ST_MODELS)

    # Qwen check + download
    print(f"\n[CHK] Qwen GGUF present? {QWEN_GGUF_PATH}")
    if QWEN_GGUF_PATH.exists():
        print(f"[OK] Qwen model already present: {QWEN_GGUF_PATH}")
    else:
        print(f"[MISS] Qwen model not found → running: {PY_GET_QWEN_MODELS}")
        run_python(PY_GET_QWEN_MODELS)
        if not QWEN_GGUF_PATH.exists():
            raise RuntimeError(f"Qwen GGUF still missing after download: {QWEN_GGUF_PATH}")
        print(f"[OK] Qwen model ready at: {QWEN_GGUF_PATH}")

    # Sentence-Transformers check
    print(f"\n[CHK] Sentence-Transformers model folder: {ST_DIR_PATH}")
    if dir_has_st_model(ST_DIR_PATH):
        print(f"[OK] ST model is present and valid.")
    else:
        raise RuntimeError(f"Sentence-Transformers model missing or incomplete at: {ST_DIR_PATH}")

    # run as module (so package imports like "from src..." work)
    run_python_module("src.load_config")

    # generate grouped CSVs
    run_python_module("src.input_data_processing.generate_grouped_csvs_with_data")

    # test generated CSVs (this is a standalone script, so file mode is fine)
    run_python_file(PY_TEST_GENERATED_CSVS)

    # company index creation (must be module mode for src imports to work)
    run_python_module("src.company_index.company_search_api", ["create"])

    print("\n✅ All steps completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Automation failed: {e}")
        sys.exit(1)

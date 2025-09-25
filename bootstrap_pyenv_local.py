#!/usr/bin/env python3
"""
Bootstrap/Installer for llama-cpp-python and model (Windows-friendly)

Commands
--------
create   : Install llama-cpp-python (CPU) and ensure model file exists.
clean    : Remove the venv (optionally wheelhouse/model).
doctor   : Check environment (venv, llama_cpp import, model present).
pip      : Run pip inside the venv with your arguments.
run      : Run a Python script/module using the venv's python.

Usage examples
--------------
python bootstrap_pyenv_local.py create
python bootstrap_pyenv_local.py doctor
python bootstrap_pyenv_local.py pip -- install -r requirements.txt
python bootstrap_pyenv_local.py run -- .\run_pyenv_pipeline.py
python bootstrap_pyenv_local.py clean --also-wheelhouse --also-model
"""

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Tuple

CPU_EXTRA_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
PACKAGE_NAME    = "llama-cpp-python"

# Model config
DEFAULT_MODEL_REPO   = "Qwen/Qwen2.5-14B-Instruct-GGUF"
REMOTE_MODEL_FILE    = "qwen2.5-14b-instruct-q4_k_m.gguf"
STRICT_MODEL_NAME    = "Qwen2.5-14B-Instruct-Q4_K_M.gguf"


def run(cmd, check=False, env=None) -> Tuple[int, str, str]:
    """Run a shell command, returning (code, stdout, stderr)."""
    print(f"[RUN] {cmd}")
    proc = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc.returncode, proc.stdout, proc.stderr


def venv_paths(venv_dir: Path) -> Tuple[Path, Path]:
    """Return paths to (python_exe, pip_exe) inside the venv (does not create)."""
    py = venv_dir / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    pip = venv_dir / ("Scripts" if os.name == "nt" else "bin") / ("pip.exe" if os.name == "nt" else "pip")
    return py, pip


def ensure_pip(py: Path):
    """Ensure pip is available in the given venv python and up-to-date."""
    code, _, _ = run(f"\"{py}\" -m pip --version")
    if code != 0:
        run(f"\"{py}\" -m ensurepip --upgrade", check=True)
    run(f"\"{py}\" -m pip install --upgrade pip")


def create_venv(venv_dir: Path) -> Tuple[Path, Path]:
    """Create venv if needed and return (python_exe, pip_exe)."""
    if not venv_dir.exists():
        print(f"[INFO] Creating venv at: {venv_dir}")
        venv.create(venv_dir, with_pip=True)
    py, pip = venv_paths(venv_dir)
    if not py.exists():
        raise RuntimeError(f"Python not found in venv: {py}")
    ensure_pip(py)
    return py, pip


def option1_install(venv_dir: Path, extra_index: str, package: str) -> bool:
    """Install from extra index into venv. True on success."""
    try:
        py, pip = create_venv(venv_dir)
        print("[INFO] Installing prebuilt CPU wheel from extra index (Option 1).")
        cmd = f"\"{pip}\" install --only-binary=:all: --extra-index-url {extra_index} {package}"
        code, _, _ = run(cmd)
        if code == 0:
            print("[OK] Option 1 succeeded: package installed into venv.")
            run(f"\"{py}\" -c \"import llama_cpp,sys;print('llama_cpp OK, py=',sys.version)\"")
            return True
        print("[WARN] Option 1 pip install returned non-zero exit code.")
        return False
    except Exception as e:
        print(f"[WARN] Option 1 failed with exception: {e}")
        return False


def option2_download_wheelhouse(wheelhouse: Path, extra_index: str, package: str) -> bool:
    """Download wheels to local wheelhouse."""
    wheelhouse.mkdir(parents=True, exist_ok=True)
    print("[INFO] Downloading wheels to local wheelhouse (Option 2).")
    cmd = (
        f"\"{sys.executable}\" -m pip download --only-binary=:all: "
        f"--index-url https://pypi.org/simple "
        f"--extra-index-url {extra_index} "
        f"-d \"{wheelhouse}\" {package}"
    )
    code, _, _ = run(cmd)
    if code == 0:
        print("[OK] Wheelhouse prepared at:", wheelhouse.resolve())
        return True
    print("[ERROR] pip download failed.")
    return False


def install_from_wheelhouse(venv_dir: Path, wheelhouse: Path, package: str) -> bool:
    """Install into venv strictly from local wheelhouse."""
    try:
        py, pip = create_venv(venv_dir)
        print("[INFO] Installing from local wheelhouse (offline style).")
        cmd = f"\"{pip}\" install --no-index --find-links \"{wheelhouse}\" {package}"
        code, _, _ = run(cmd)
        if code == 0:
            print("[OK] Installed from wheelhouse.")
            run(f"\"{py}\" -c \"import llama_cpp,sys;print('llama_cpp OK (wheelhouse), py=',sys.version)\"")
            return True
        print("[ERROR] wheelhouse install failed.")
        return False
    except Exception as e:
        print(f"[ERROR] wheelhouse install exception: {e}")
        return False


def ensure_model(models_dir: Path, repo: str, remote_name: str, strict_name: str) -> bool:
    """
    Ensure models/STRICT_NAME exists. If missing, download from HF public repo.
    Returns True if model present or downloaded successfully.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / strict_name
    if target.exists() and target.stat().st_size > 0:
        print(f"[OK] Model already present: {target}")
        return True

    url = f"https://huggingface.co/{repo}/resolve/main/{remote_name}?download=true"
    tmp = models_dir / remote_name
    print(f"[INFO] Downloading model:\n  {url}\n  -> {tmp}")
    try:
        from urllib.request import urlopen, Request
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as resp, open(tmp, "wb") as f:
            import shutil as _sh
            _sh.copyfileobj(resp, f)
        if target.exists():
            target.unlink()
        tmp.rename(target)
        print(f"[OK] Saved model to: {target}")
        return True
    except Exception as e:
        print(f"[ERROR] Model download failed: {e}")
        print("       Please download manually and place at:", target)
        return False


# --------------------- Subcommand implementations ---------------------

def cmd_create(args) -> int:
    venv_dir  = Path(args.venv)
    wheelhouse = Path(args.wheelhouse)
    models_dir = Path(args.models_dir)

    # Safe defaults if older copies of this script lacked these args
    model_repo        = getattr(args, "model_repo", DEFAULT_MODEL_REPO)
    model_strict_name = getattr(args, "model_strict_name", STRICT_MODEL_NAME)

    # Step 1: Install llama-cpp-python
    ok = option1_install(venv_dir, args.extra_index_url, args.package)
    if not ok:
        print("[INFO] Falling back to Option 2: download wheelhouse and install from it.")
        if option2_download_wheelhouse(wheelhouse, args.extra_index_url, args.package):
            ok = install_from_wheelhouse(venv_dir, wheelhouse, args.package)

    if not ok:
        print("[FATAL] Could not install llama-cpp-python via Option 1 or Option 2.")
        print("        If you are fully offline, run this on an online machine first to populate the wheelhouse,")
        print("        then copy the wheelhouse to the offline machine and re-run 'create'.")
        return 2

    # Step 2: Ensure model file exists (skip download if already there)
    if not ensure_model(models_dir, model_repo, REMOTE_MODEL_FILE, model_strict_name):
        print("[WARN] Proceeding without model download; install step completed.")
        return 1

    print("[DONE] Environment ready:")
    print(f"  Venv : {venv_dir}")
    print(f"  Model: {models_dir / model_strict_name}")
    return 0


def cmd_clean(args) -> int:
    venv_dir  = Path(args.venv)
    wheelhouse = Path(args.wheelhouse)
    models_dir = Path(args.models_dir)

    rc = 0
    if venv_dir.exists():
        print(f"[CLEAN] Removing venv: {venv_dir}")
        try:
            shutil.rmtree(venv_dir)
        except Exception as e:
            print(f"[WARN] Failed to remove venv: {e}")
            rc = 1
    else:
        print(f"[CLEAN] No venv at: {venv_dir}")

    if args.also_wheelhouse and wheelhouse.exists():
        print(f"[CLEAN] Removing wheelhouse: {wheelhouse}")
        try:
            shutil.rmtree(wheelhouse)
        except Exception as e:
            print(f"[WARN] Failed to remove wheelhouse: {e}")
            rc = 1
    elif args.also_wheelhouse:
        print(f"[CLEAN] No wheelhouse at: {wheelhouse}")

    if args.also_model:
        target = models_dir / STRICT_MODEL_NAME
        if target.exists():
            try:
                print(f"[CLEAN] Removing model file: {target}")
                target.unlink()
            except Exception as e:
                print(f"[WARN] Failed to remove model: {e}")
                rc = 1
        else:
            print(f"[CLEAN] No model at: {target}")

    print("[CLEAN] Done.")
    return rc


def cmd_doctor(args) -> int:
    import struct

    ok = True
    pyver = sys.version.split()[0]
    bits  = 8 * struct.calcsize('P')
    print(f"[DOCTOR] System Python: {pyver}  ({bits}-bit)")

    major, minor = sys.version_info.major, sys.version_info.minor
    if not (major == 3 and 10 <= minor <= 12):
        print("[DOCTOR][WARN] Recommended Python 3.10-3.12 for best wheel availability.")
        ok = False

    # venv presence
    venv_dir = Path(args.venv)
    py, pip = venv_paths(venv_dir)
    if py.exists():
        print(f"[DOCTOR] Venv python: {py}")
        code, out, _ = run(f"\"{py}\" -m pip --version")
        if code != 0:
            print("[DOCTOR][WARN] pip not available in venv.")
            ok = False
        code, out, err = run(f"\"{py}\" -c \"import llama_cpp,sys; print('llama_cpp version:', getattr(llama_cpp,'__version__','unknown'))\"")
        if code != 0:
            print("[DOCTOR][WARN] Cannot import llama_cpp inside venv.")
            ok = False
    else:
        print(f"[DOCTOR][WARN] Venv not found at {venv_dir}. Run 'create' first.")
        ok = False

    # model check
    model_path = Path(args.models_dir) / STRICT_MODEL_NAME
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024*1024)
        print(f"[DOCTOR] Model present: {model_path}  ({size_mb:.1f} MB)")
        if size_mb < 100:
            print("[DOCTOR][WARN] Model file seems too small; may be incomplete.")
            ok = False
    else:
        print(f"[DOCTOR][WARN] Model missing: {model_path}")
        ok = False

    if ok:
        print("[DOCTOR] All checks passed.")
        return 0
    else:
        print("[DOCTOR] Issues detected. See warnings above.")
        return 1


def cmd_pip(args) -> int:
    venv_dir = Path(args.venv)
    py, pip = venv_paths(venv_dir)
    if not pip.exists():
        print(f"[PIP][ERROR] Venv not found at {venv_dir}. Run 'create' first.")
        return 2

    pip_args = args.pip_args
    if pip_args and pip_args[0] == "--":
        pip_args = pip_args[1:]

    cmd = f"\"{pip}\" " + " ".join(pip_args)
    code, _, _ = run(cmd)
    return code


def cmd_run(args) -> int:
    venv_dir = Path(args.venv)
    py, pip = venv_paths(venv_dir)
    if not py.exists():
        print(f"[RUN][ERROR] Venv not found at {venv_dir}. Run 'create' first.")
        return 2

    run_args = args.run_args
    if run_args and run_args[0] == "--":
        run_args = run_args[1:]

    # Use subprocess.run with list of arguments to preserve quoting
    cmd_list = [str(py)] + run_args
    cmd_display = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd_list)
    print(f"[RUN] {cmd_display}")
    
    proc = subprocess.run(cmd_list, env=os.environ)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Bootstrap llama-cpp-python (CPU) and model file.")
    ap.add_argument("--venv", default=".venv", help="Path to create/use Python venv for installation.")
    ap.add_argument("--wheelhouse", default="wheelhouse", help="Folder to place downloaded wheels (fallback).")
    ap.add_argument("--models-dir", default="models", help="Directory to store the model GGUF.")
    ap.add_argument("--model-repo", default=DEFAULT_MODEL_REPO, help="HF repo with GGUF.")
    ap.add_argument("--model-strict-name", default=STRICT_MODEL_NAME, help="Strict model filename to save as.")
    ap.add_argument("--extra-index-url", default=CPU_EXTRA_INDEX, help="Extra index for prebuilt CPU wheels.")
    ap.add_argument("--package", default=PACKAGE_NAME, help="Package name to install/download.")

    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("create", help="Install llama-cpp-python and ensure model file is present.")
    sp.set_defaults(func=cmd_create)

    sp = sub.add_parser("clean", help="Remove venv (and optionally wheelhouse/model).")
    sp.add_argument("--also-wheelhouse", action="store_true", dest="also_wheelhouse", help="Also remove wheelhouse dir.")
    sp.add_argument("--also-model", action="store_true", dest="also_model", help="Also remove model file.")
    sp.set_defaults(func=cmd_clean)

    sp = sub.add_parser("doctor", help="Check environment health.")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("pip", help="Run pip inside the venv with your arguments.")
    sp.add_argument("pip_args", nargs=argparse.REMAINDER, help="Arguments to pass to pip (prefix with --).")
    sp.set_defaults(func=cmd_pip)

    sp = sub.add_parser("run", help="Run a Python script/module using the venv's python.")
    sp.add_argument("run_args", nargs=argparse.REMAINDER, help="Arguments to pass to python (prefix with --).")
    sp.set_defaults(func=cmd_run)

    args = ap.parse_args()
    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()

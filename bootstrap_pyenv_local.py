#!/usr/bin/env python3
"""
bootstrap_pyenv_local.py — one-file, Python-only bootstrap for a local venv in CWD (no conda).

Env path: ./.venv/llcpp
Optional local CPython (Windows): ./.pyenv/python311

Commands:
  python bootstrap_pyenv_local.py create [--pyexe "C:\\Path\\To\\python.exe"] [--recreate]
  python bootstrap_pyenv_local.py run -- <command...>          # e.g. -m pkg.module or script.py
  python bootstrap_pyenv_local.py pip -- <pip args...>         # e.g. install -r requirements.txt
  python bootstrap_pyenv_local.py freeze --full requirements.txt
  python bootstrap_pyenv_local.py freeze --top requirements.top.txt --lock requirements.lock.txt
  python bootstrap_pyenv_local.py doctor
  python bootstrap_pyenv_local.py shell
  python bootstrap_pyenv_local.py clean
"""
import os, sys, platform, stat, shutil, subprocess, urllib.request, textwrap
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path.cwd()
PYENV_HOME   = PROJECT_ROOT / ".pyenv" / "python311"     # local CPython install (Windows)
VENV_DIR     = PROJECT_ROOT / ".venv" / "llcpp"          # project venv
PY_VER_DOT   = "3.11.9"                                  # CPython to install locally if needed (Windows)
PY_WIN_EXE   = PROJECT_ROOT / f"python-{PY_VER_DOT}-amd64.exe"

def log(msg):  print(f"[+] {msg}")
def warn(msg): print(f"[!] {msg}")

def run(cmd, env=None, check=True, cwd=None):
    print("→", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, env=env, check=check, cwd=cwd)

def force_rmtree(p: Path):
    if not p.exists(): return
    def onerror(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass
    shutil.rmtree(p, onerror=onerror)

# ---------- Python/venv helpers ----------
def venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python3")

def pyenv_python_exe() -> Optional[Path]:
    exe = PYENV_HOME / ("python.exe" if os.name == "nt" else "bin/python3")
    return exe if exe.exists() else None

def resolve_local_python(explicit: Optional[str] = None) -> Path:
    """Pick a Python executable to create the venv (explicit > embedded > current > PATH)."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        warn(f"--pyexe not found: {explicit}")
    p2 = pyenv_python_exe()
    if p2:
        return p2
    if Path(sys.executable).exists():
        return Path(sys.executable)
    for name in ("python.exe", "python3.exe", "python", "python3"):
        w = shutil.which(name)
        if w:
            return Path(w)
    raise RuntimeError("No usable Python found to create venv.")

def install_cpython_local() -> bool:
    """Install a project-local CPython into ./.pyenv/python311 (Windows only)."""
    if pyenv_python_exe():
        log(f"Local CPython present at {PYENV_HOME}")
        return True
    if platform.system().lower() != "windows":
        warn("Local CPython installer is only automated on Windows; using system Python instead.")
        return False
    url = f"https://www.python.org/ftp/python/{PY_VER_DOT}/python-{PY_VER_DOT}-amd64.exe"
    log(f"Downloading CPython {PY_VER_DOT} -> {PY_WIN_EXE}")
    urllib.request.urlretrieve(url, PY_WIN_EXE)
    try:
        run([str(PY_WIN_EXE),
             "/quiet",
             "InstallAllUsers=0",
             "PrependPath=0",
             "Include_test=0",
             "Shortcuts=0",
             "SimpleInstall=1",
             f"TargetDir={PYENV_HOME}"])
    finally:
        PY_WIN_EXE.unlink(missing_ok=True)
    ok = bool(pyenv_python_exe())
    if ok:
        log(f"Installed local CPython at {PYENV_HOME}")
    else:
        warn("Local CPython install did not produce a usable python.exe")
    return ok

def ensure_venv(recreate: bool = False, pyexe: Optional[str] = None):
    """Create/refresh .venv/llcpp with chosen Python; upgrade pip/setuptools/wheel."""
    if recreate and VENV_DIR.exists():
        force_rmtree(VENV_DIR)
    if not VENV_DIR.exists():
        VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        # Prefer explicit Python, else embedded, else try to install embedded (Windows), else fallback to current/python on PATH
        python = None
        if pyexe:
            python = Path(pyexe) if Path(pyexe).exists() else None
            if not python:
                warn(f"--pyexe path not found: {pyexe}")
        if python is None:
            python = pyenv_python_exe()
        if python is None:
            python = resolve_local_python(None)

        if python is None and platform.system().lower() == "windows":
            if install_cpython_local():
                python = pyenv_python_exe()

        run([str(python), "-m", "venv", str(VENV_DIR)])
        log(f"Venv created at {VENV_DIR}")

    # Always ensure base tooling is up to date
    py = str(venv_python(VENV_DIR))
    run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

def run_in_env(args):
    """Run a command inside the venv. Usage: run -- <command...>"""
    if not args or args[0] != "--":
        print("Usage: run -- <command...>"); sys.exit(2)
    cmd = args[1:]
    # If caller passed a leading "python", drop it (we're already calling venv's python)
    if cmd and str(cmd[0]).lower().startswith("python"):
        cmd = cmd[1:]
    full_cmd = [str(venv_python(VENV_DIR)), *cmd]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    run(full_cmd, env=env, cwd=PROJECT_ROOT)

def pip_in_env(args):
    """Run pip inside the venv. Usage: pip -- <pip args>"""
    if not args or args[0] != "--":
        print("Usage: pip -- <pip args>"); sys.exit(2)
    pip_args = args[1:]
    run([str(venv_python(VENV_DIR)), "-m", "pip", *pip_args], cwd=PROJECT_ROOT)

def freeze_from_env(full_out: Optional[str] = None, top_out: Optional[str] = None, lock_out: Optional[str] = None):
    """Write requirements files from the venv using importlib.metadata (pip-style)."""
    code = r'''
import re
from importlib import metadata
def canonicalize(name: str) -> str: return re.sub(r"[-_.]+", "-", name).lower()
def parse_req_name(req: str) -> str:
    s=req.strip(); s=s.split(";")[0].strip()
    s=re.split(r"[<>=!~]",s,1)[0].strip(); s=s.split("[",1)[0].strip()
    return canonicalize(s)
dists=list(metadata.distributions()); name_to_ver={}; all_required=set()
for d in dists:
    try: nm=d.metadata.get("Name") or d.name
    except Exception: nm=d.name
    name_to_ver[canonicalize(nm)]=(nm,d.version)
    for req in (d.requires or []):
        try: all_required.add(parse_req_name(req))
        except Exception: pass
ignore={"pip","setuptools","wheel"}
top=[c for c in name_to_ver if c not in all_required and c not in ignore]
def emit_all(path): open(path,"w",encoding="utf-8").write("".join(f"{n}=={v}\n" for n,v in sorted(name_to_ver.values(), key=lambda x:x[0].lower())))
def emit_top(path): open(path,"w",encoding="utf-8").write("".join(f"{name_to_ver[c][0]}=={name_to_ver[c][1]}\n" for c in sorted(top)))
'''
    py = str(venv_python(VENV_DIR))
    if full_out:
        run([py, "-c", code + f'\nemit_all(r"{full_out}")\nprint("Wrote {full_out}")\n'])
    if top_out and lock_out:
        run([py, "-c", code + f'\nemit_top(r"{top_out}")\nemit_all(r"{lock_out}")\nprint("Wrote {top_out} and {lock_out}")\n'])

def doctor():
    print("== Doctor ==")
    print("CWD:", PROJECT_ROOT)
    print("OS:", platform.platform())
    print("Host Python:", sys.version)
    print("Embedded CPython:", pyenv_python_exe())
    print("Venv path:", VENV_DIR)
    print("Venv exists:", VENV_DIR.exists())
    if VENV_DIR.exists():
        try:
            vp = venv_python(VENV_DIR)
            print("Venv python:", vp, "exists:", vp.exists())
            print("Venv sample:", list(VENV_DIR.iterdir())[:5])
        except Exception: pass

def shell():
    """Open an interactive shell with the venv activated."""
    if os.name == "nt":
        act = VENV_DIR / "Scripts" / "activate.bat"
        os.system(f'start cmd /k "{act}"')
    else:
        os.system(f"bash -lc \"source '{VENV_DIR}/bin/activate' && exec bash\"")

def clean():
    targets = [PROJECT_ROOT / ".venv", PROJECT_ROOT / ".pyenv", PY_WIN_EXE]
    for t in targets:
        try:
            if t.is_dir():
                print("🧹 dir", t); force_rmtree(t)
            elif t.exists():
                print("🧹 file", t); t.unlink(missing_ok=True)
        except Exception as e:
            warn(f"Could not remove {t}: {e}")
    log("Cleanup complete.")

# ---------- CLI ----------
def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    cmd = sys.argv[1].lower()
    args = sys.argv[2:]

    if cmd == "create":
        # parse optional flags: --pyexe <path>, --recreate
        pyexe = None
        recreate = False
        if "--pyexe" in args:
            i = args.index("--pyexe")
            pyexe = args[i+1] if i + 1 < len(args) else None
            args = args[:i] + args[i+2:]
        if "--recreate" in args:
            recreate = True
            args.remove("--recreate")

        ensure_venv(recreate=recreate, pyexe=pyexe)
        vp = venv_python(VENV_DIR)
        log(f'Use without activation:\n  "{vp}" -V')

    elif cmd == "run":
        if not VENV_DIR.exists():
            ensure_venv(recreate=False)
        run_in_env(args)

    elif cmd == "pip":
        if not VENV_DIR.exists():
            ensure_venv(recreate=False)
        pip_in_env(args)

    elif cmd == "freeze":
        if "--full" in args:
            i = args.index("--full")
            try:
                out = args[i+1]
            except Exception:
                print("freeze --full requirements.txt"); sys.exit(2)
            freeze_from_env(full_out=out)
        elif "--top" in args and "--lock" in args:
            it = args.index("--top"); il = args.index("--lock")
            try:
                top = args[it+1]; lock = args[il+1]
            except Exception:
                print("freeze --top <top.txt> --lock <lock.txt>"); sys.exit(2)
            freeze_from_env(top_out=top, lock_out=lock)
        else:
            print(textwrap.dedent("""\
              Usage:
                freeze --full requirements.txt
                freeze --top requirements.top.txt --lock requirements.lock.txt
            """)); sys.exit(2)

    elif cmd == "doctor":
        doctor()

    elif cmd == "shell":
        if not VENV_DIR.exists():
            ensure_venv(recreate=False)
        shell()

    elif cmd == "clean":
        clean()

    else:
        print("Unknown command:", cmd); print(__doc__); sys.exit(2)

if __name__ == "__main__":
    main()

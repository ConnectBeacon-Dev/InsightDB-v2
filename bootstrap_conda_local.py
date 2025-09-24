#!/usr/bin/env python3
"""
bootstrap_conda_local.py  —  one-file, Python-only bootstrap for a local conda env in CWD.

Commands:
  python bootstrap_conda_local.py create        # install conda (Miniforge) or micromamba, create ./.conda/llcpp
  python bootstrap_conda_local.py run -- python -V
  python bootstrap_conda_local.py pip -- install -r requirements.txt
  python bootstrap_conda_local.py freeze --full requirements.txt
  python bootstrap_conda_local.py freeze --top requirements.top.txt --lock requirements.lock.txt
  python bootstrap_conda_local.py doctor
  python bootstrap_conda_local.py clean
"""
import os, sys, platform, stat, shutil, subprocess, urllib.request, time, textwrap
from pathlib import Path

ROOT       = Path.cwd()
FORGE_DIR  = ROOT / ".miniforge"
MAMBA_DIR  = ROOT / ".micromamba"
ENV_DIR    = ROOT / ".conda" / "llcpp"
ENVS_DIR   = ROOT / ".conda" / "envs"
PKGS_DIR   = ROOT / ".conda" / "pkgs"
CONDARC    = ROOT / ".condarc"

WIN_FORGE_EXE = ROOT / ".miniforge_installer.exe"
LIN_FORGE_SH  = ROOT / ".miniforge_installer.sh"
WIN_MAMBA_EXE = MAMBA_DIR / "micromamba.exe"
LIN_MAMBA    = MAMBA_DIR / "micromamba"

def log(msg): print(f"[+] {msg}")
def warn(msg): print(f"[!] {msg}")
def run(cmd, env=None, check=True):
    print("→", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, env=env, check=check)

def force_rmtree(p: Path):
    if not p.exists(): return
    def onerror(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE); func(path)
        except Exception: pass
    shutil.rmtree(p, onerror=onerror)

def ensure_dirs():
    (ROOT / ".conda").mkdir(parents=True, exist_ok=True)
    ENVS_DIR.mkdir(parents=True, exist_ok=True)
    PKGS_DIR.mkdir(parents=True, exist_ok=True)
    MAMBA_DIR.mkdir(parents=True, exist_ok=True)

def write_condarc():
    text = f"""channels:
  - conda-forge
channel_priority: flexible
auto_activate_base: false
envs_dirs:
  - {ENVS_DIR.as_posix()}
pkgs_dirs:
  - {PKGS_DIR.as_posix()}
solver: libmamba
"""
    CONDARC.write_text(text, encoding="utf-8")

def env_overrides():
    e = os.environ.copy()
    e["CONDARC"] = str(CONDARC)
    e["CONDA_ENVS_DIRS"] = str(ENVS_DIR)
    e["CONDA_PKGS_DIRS"] = str(PKGS_DIR)
    e["CONDA_CHANNELS"]  = "conda-forge"
    return e

def download(url: str, out: Path):
    log(f"Downloading {url} -> {out}")
    urllib.request.urlretrieve(url, out)

def conda_path() -> Path | None:
    if platform.system().lower() == "windows":
        cands = [FORGE_DIR / "Scripts" / "conda.exe", FORGE_DIR / "condabin" / "conda.bat"]
    else:
        cands = [FORGE_DIR / "bin" / "conda"]
    for c in cands:
        if c.exists(): return c
    return None

def micromamba_path() -> Path | None:
    if platform.system().lower() == "windows":
        return WIN_MAMBA_EXE if WIN_MAMBA_EXE.exists() else None
    else:
        return LIN_MAMBA if LIN_MAMBA.exists() else None

def install_miniforge():
    if conda_path(): 
        log(f"Miniforge present at {FORGE_DIR}"); 
        return True
    log(f"Installing Miniforge into {FORGE_DIR} ...")
    if platform.system().lower() == "windows":
        url = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
        download(url, WIN_FORGE_EXE)
        try:
            run([str(WIN_FORGE_EXE), "/S","/InstallationType=JustMe","/AddToPath=0","/RegisterPython=0", f"/D={FORGE_DIR}"])
        finally:
            WIN_FORGE_EXE.unlink(missing_ok=True)
        # give the installer a moment to finalize
        for _ in range(20):
            if conda_path(): return True
            time.sleep(0.3)
        warn("Miniforge installer finished but conda.exe not found.")
        return False
    else:
        url = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
        download(url, LIN_FORGE_SH)
        try:
            run(["bash", str(LIN_FORGE_SH), "-b", "-p", str(FORGE_DIR)])
        finally:
            LIN_FORGE_SH.unlink(missing_ok=True)
        return bool(conda_path())

def install_micromamba():
    if micromamba_path():
        log(f"micromamba present at {micromamba_path()}")
        return True
    log(f"Installing micromamba into {MAMBA_DIR} ...")
    if platform.system().lower() == "windows":
        url = "https://micro.mamba.pm/api/micromamba/win-64/latest"
        download(url, WIN_MAMBA_EXE)
        return WIN_MAMBA_EXE.exists()
    else:
        url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
        download(url, LIN_MAMBA)
        os.chmod(LIN_MAMBA, 0o755)
        return LIN_MAMBA.exists()

def ensure_pkg_mgr():
    """Return ('conda', path) or ('micromamba', path). Try conda first, fallback to micromamba."""
    if install_miniforge():
        c = conda_path()
        if c: return ("conda", c)
    warn("Falling back to micromamba.")
    if install_micromamba():
        m = micromamba_path()
        if m: return ("micromamba", m)
    raise RuntimeError("Could not install Miniforge or micromamba (no internet or blocked by policy).")

def create_env():
    ensure_dirs(); write_condarc()
    mgr, exe = ensure_pkg_mgr()
    e = env_overrides()
    force_rmtree(ENV_DIR)
    ENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    log(f"Creating env at {ENV_DIR} using {mgr} ...")
    if mgr == "conda":
        run([str(exe), "create", "--yes", "--prefix", str(ENV_DIR),
             "--override-channels", "-c", "conda-forge",
             "python=3.11", "pip", "setuptools", "wheel"], env=e)
    else:
        # micromamba is drop-in for conda commands
        run([str(exe), "create", "-y", "-p", str(ENV_DIR),
             "-c", "conda-forge", "python=3.11", "pip", "setuptools", "wheel"], env=e)
    log("Env created.")

def run_in_env(args):
    mgr, exe = ensure_pkg_mgr()
    e = env_overrides()
    if not args or args[0] != "--":
        print("Usage: run -- <command...>"); sys.exit(2)
    cmd = args[1:]
    if mgr == "conda":
        run([str(exe), "run", "-p", str(ENV_DIR), *cmd], env=e)
    else:
        run([str(exe), "run", "-p", str(ENV_DIR), *cmd], env=e)

def pip_in_env(args):
    mgr, exe = ensure_pkg_mgr()
    e = env_overrides()
    if not args or args[0] != "--":
        print("Usage: pip -- <pip args>"); sys.exit(2)
    pip_args = args[1:]
    run([str(exe), "run", "-p", str(ENV_DIR), "python", "-m", "pip", *pip_args], env=e)

def freeze_from_env(full_out=None, top_out=None, lock_out=None):
    mgr, exe = ensure_pkg_mgr()
    e = env_overrides()
    code = r'''
import re
from importlib import metadata
def canonicalize(name: str) -> str: return re.sub(r"[-_.]+", "-", name).lower()
def parse_req_name(req: str) -> str:
    s=req.strip(); s=s.split(";")[0].strip(); s=re.split(r"[<>=!~]",s,1)[0].strip(); s=s.split("[",1)[0].strip(); return canonicalize(s)
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
def emit_all(p): open(p,"w",encoding="utf-8").write("".join(f"{n}=={v}\n" for n,v in sorted(name_to_ver.values(), key=lambda x:x[0].lower())))
def emit_top(p): open(p,"w",encoding="utf-8").write("".join(f"{name_to_ver[c][0]}=={name_to_ver[c][1]}\n" for c in sorted(top)))
'''
    py = [str(exe), "run", "-p", str(ENV_DIR), "python", "-c"]
    if full_out:
        run(py + [code + f'\nemit_all(r"{full_out}")\nprint("Wrote {full_out}")\n'], env=e)
    if top_out and lock_out:
        run(py + [code + f'\nemit_top(r"{top_out}")\nemit_all(r"{lock_out}")\nprint("Wrote {top_out} and {lock_out}")\n'], env=e)

def doctor():
    print("== Doctor ==")
    print("CWD:", ROOT)
    print("OS:", platform.platform())
    print("Python:", sys.version)
    print("Miniforge conda:", conda_path())
    print("Micromamba:", micromamba_path())
    print(".conda exists:", (ROOT / ".conda").exists())
    print("ENV_DIR exists:", ENV_DIR.exists())
    if ENV_DIR.exists():
        print("Contents of env dir:", list(ENV_DIR.iterdir())[:5])

def clean():
    targets = [FORGE_DIR, MAMBA_DIR, ROOT / ".conda", CONDARC, WIN_FORGE_EXE, LIN_FORGE_SH, WIN_MAMBA_EXE, LIN_MAMBA]
    for t in targets:
        try:
            if t.is_dir():
                print("🧹 dir", t); force_rmtree(t)
            elif t.exists():
                print("🧹 file", t); t.unlink(missing_ok=True)
        except Exception as e:
            warn(f"Could not remove {t}: {e}")
    log("Cleanup complete.")

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    cmd = sys.argv[1].lower()
    args = sys.argv[2:]
    if cmd == "create":
        ensure_dirs(); write_condarc(); create_env(); 
        mgr, exe = ensure_pkg_mgr()
        log(f'Use without activation:\n  "{exe}" run -p "{ENV_DIR}" python -V')
    elif cmd == "run":
        run_in_env(args)
    elif cmd == "pip":
        pip_in_env(args)
    elif cmd == "freeze":
        if "--full" in args:
            i = args.index("--full"); out = args[i+1] if i+1 < len(args) else None
            if not out: print("freeze --full requirements.txt"); sys.exit(2)
            freeze_from_env(full_out=out)
        elif "--top" in args and "--lock" in args:
            it = args.index("--top"); il = args.index("--lock")
            top = args[it+1] if it+1 < len(args) else None
            lock= args[il+1] if il+1 < len(args) else None
            if not (top and lock): print("freeze --top <top.txt> --lock <lock.txt>"); sys.exit(2)
            freeze_from_env(top_out=top, lock_out=lock)
        else:
            print(textwrap.dedent("""\
              Usage:
                freeze --full requirements.txt
                freeze --top requirements.top.txt --lock requirements.lock.txt
            """)); sys.exit(2)
    elif cmd == "doctor":
        doctor()
    elif cmd == "clean":
        clean()
    else:
        print("Unknown command:", cmd); print(__doc__); sys.exit(2)

if __name__ == "__main__":
    main()

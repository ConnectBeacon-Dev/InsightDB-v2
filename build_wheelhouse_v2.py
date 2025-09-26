
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_wheelhouse_v2.py

Build an offline wheelhouse. Adds support for:
  --extra-index-url <URL>        (applies to ALL downloads)
  --llama-extra-index <URL>      (applies ONLY to llama-cpp-python)

Example (CPU prebuilt wheels from official extra index):
  python build_wheelhouse_v2.py --dest wheels --requirements requirements.txt \
    --llama "llama-cpp-python==0.3.16" \
    --llama-extra-index https://abetlen.github.io/llama-cpp-python/whl/cpu

Example (CUDA 12.4 prebuilt wheels):
  python build_wheelhouse_v2.py --dest wheels --requirements requirements.txt \
    --llama "llama-cpp-python==0.3.16" \
    --llama-extra-index https://abetlen.github.io/llama-cpp-python/whl/cu124

Then on offline machine:
  PIP_NO_INDEX=1 PIP_FIND_LINKS=./wheels pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

def p(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}")

def repo_root() -> Path:
    return Path(__file__).resolve().parent

def ensure_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d

def read_text_any(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe"):
        return data.decode("utf-16-le")
    if data.startswith(b"\xfe\xff"):
        return data.decode("utf-16-be")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    for enc in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")

def filter_requirements_excluding_llama(req_path: Path) -> Path:
    is_empty_or_comment = re.compile(r"^\s*(?:#.*)?$")
    is_llama = re.compile(r"^\s*llama[-_]?cpp[-_]?python", re.IGNORECASE)
    tmp = repo_root() / ".requirements.no-llama.txt"
    lines_out: List[str] = []
    for line in read_text_any(req_path).splitlines():
        if is_empty_or_comment.match(line):
            lines_out.append(line)
            continue
        if is_llama.match(line):
            p("INFO", f"Pruned from requirements: {line.strip()}")
            continue
        lines_out.append(line)
    tmp.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return tmp

def pip_download(dest: Path, items: List[str], *, only_binary: str = ":all:",
                 platform: Optional[str] = None, python_version: Optional[str] = None,
                 implementation: Optional[str] = None, abi: Optional[str] = None,
                 no_deps: bool = False, extra_index_url: Optional[str] = None) -> None:
    if not items:
        return
    ensure_dir(dest)
    cmd = [sys.executable, "-m", "pip", "download", "-d", str(dest), "--only-binary", only_binary]
    if extra_index_url:
        cmd += ["--extra-index-url", extra_index_url]
    if platform:
        cmd += ["--platform", platform]
    if python_version:
        cmd += ["--python-version", python_version]
    if implementation:
        cmd += ["--implementation", implementation]
    if abi:
        cmd += ["--abi", abi]
    if no_deps:
        cmd += ["--no-deps"]
    cmd += items
    p("RUN", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)
    p("OK", f"Downloaded {len(items)} item(s) to {dest}")

def copy_wheel(src: Path, dest: Path) -> None:
    ensure_dir(dest)
    tgt = dest / src.name
    shutil.copy2(src, tgt)
    p("OK", f"Copied wheel -> {tgt}")

def find_local_llama_wheel(roots: List[Path]) -> Optional[Path]:
    pats = ("llama_cpp_python-*.whl", "llama-cpp-python-*.whl")
    for r in roots:
        if not r.exists():
            continue
        for pat in pats:
            matches = sorted(r.glob(pat), reverse=True)
            if matches:
                return matches[0]
    return None

def build_wheelhouse(args: argparse.Namespace) -> int:
    dest = Path(args.dest)
    ensure_dir(dest)

    # 1) requirements (minus llama-cpp-python) unless skipped
    if args.include_reqs:
        req = Path(args.requirements)
        if not req.exists():
            p("WARN", f"requirements file not found: {req} (skipping)")
        else:
            filt = filter_requirements_excluding_llama(req) if args.prune_llama else req
            pip_download(dest, ["-r", str(filt)], only_binary=args.only_binary,
                         platform=args.platform, python_version=args.python_version,
                         implementation=args.implementation, abi=args.abi,
                         no_deps=args.no_deps, extra_index_url=args.extra_index_url)

    # 2) llama-cpp-python
    llama = (args.llama or "auto").strip()
    if llama.lower() != "skip":
        wheel_path = Path(llama)
        if wheel_path.suffix.lower() == ".whl" and wheel_path.exists():
            copy_wheel(wheel_path, dest)
        else:
            if llama.lower() == "auto":
                env_spec = os.getenv("LLAMA_CPP_SPEC", "").strip()
                if env_spec:
                    llama = env_spec
                    p("INFO", f"Using LLAMA_CPP_SPEC: {llama}")
                else:
                    wheel = find_local_llama_wheel([repo_root()/ "wheels", repo_root()/ "vendor", repo_root()])
                    if wheel:
                        copy_wheel(wheel, dest)
                        llama = ""  # done
                    else:
                        llama = "llama-cpp-python"
                        p("WARN", "No local llama wheel found; downloading unpinned 'llama-cpp-python'. "
                                  "Prefer --llama 'llama-cpp-python==<ver>' for reproducibility.")
            if llama:
                pip_download(dest, [llama], only_binary=args.only_binary,
                             platform=args.platform, python_version=args.python_version,
                             implementation=args.implementation, abi=args.abi,
                             no_deps=args.no_deps, extra_index_url=args.llama_extra_index or args.extra_index_url)

    # 3) extras
    if args.extra:
        pip_download(dest, args.extra, only_binary=args.only_binary,
                     platform=args.platform, python_version=args.python_version,
                     implementation=args.implementation, abi=args.abi,
                     no_deps=args.no_deps, extra_index_url=args.extra_index_url)

    wheels = sorted([p.name for p in dest.glob("*.whl")])
    p("OK", f"Wheelhouse ready: {dest} ({len(wheels)} wheels)")
    for w in wheels[:20]:
        print("  -", w)
    if len(wheels) > 20:
        print(f"  ... and {len(wheels)-20} more")

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "python": sys.version,
        "dest": str(dest.resolve()),
        "requirements": str(Path(args.requirements).resolve()),
        "include_reqs": args.include_reqs,
        "prune_llama": args.prune_llama,
        "llama": args.llama,
        "extra": args.extra,
        "cross": {
            "platform": args.platform,
            "python_version": args.python_version,
            "implementation": args.implementation,
            "abi": args.abi,
            "only_binary": args.only_binary,
            "no_deps": args.no_deps,
            "extra_index_url": args.extra_index_url,
            "llama_extra_index": args.llama_extra_index,
        },
        "wheels": wheels,
    }
    (dest / "wheelhouse.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    p("OK", f"Wrote manifest -> {dest / 'wheelhouse.manifest.json'}")
    return 0

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an offline wheelhouse (keeps llama-cpp-python outside requirements.txt).")
    p.add_argument("--dest", default="wheels", help="Destination directory for wheels (default: wheels)")
    p.add_argument("--requirements", default="requirements.txt", help="Path to requirements.txt")
    p.add_argument("--include-reqs", dest="include_reqs", action="store_true", default=True, help="Download requirements (default: true)")
    p.add_argument("--no-reqs", dest="include_reqs", action="store_false", help="Skip downloading requirements")
    p.add_argument("--prune-llama", action="store_true", default=True, help="Exclude llama-cpp-python lines from requirements (default: true)")
    p.add_argument("--no-prune-llama", dest="prune_llama", action="store_false", help="Do not prune llama from requirements")
    p.add_argument("--llama", default="auto",
                   help="llama-cpp-python spec or path to local wheel. "
                        "Use 'auto' to prefer LLAMA_CPP_SPEC/local wheel, else download 'llama-cpp-python'. "
                        "Use 'skip' to omit.")
    p.add_argument("--extra", nargs="*", default=None, help="Extra pip specs to include in the wheelhouse")
    p.add_argument("--only-binary", default=":all:", help="pip --only-binary value (default: :all:)")
    p.add_argument("--extra-index-url", default=None, help="Extra package index URL for ALL downloads")
    p.add_argument("--llama-extra-index", default=None, help="Extra index URL ONLY for llama-cpp-python")
    p.add_argument("--platform", default=None, help="Target platform tag (e.g., manylinux2014_x86_64, win_amd64)")
    p.add_argument("--python-version", default=None, help="Target Python version tag (e.g., 3.12, 3.11)")
    p.add_argument("--implementation", default=None, help="Target implementation tag (e.g., cp)")
    p.add_argument("--abi", default=None, help="Target ABI tag (e.g., cp312)")
    p.add_argument("--no-deps", action="store_true", help="Do not download dependencies (advanced)")
    return p.parse_args()

def main() -> int:
    try:
        return build_wheelhouse(parse_args())
    except subprocess.CalledProcessError as e:
        p("ERROR", f"pip exited with code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        p("ERROR", "Interrupted by user")
        return 130
    except Exception as e:
        p("ERROR", f"{e.__class__.__name__}: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

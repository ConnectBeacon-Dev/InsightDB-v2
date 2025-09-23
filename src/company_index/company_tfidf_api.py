#!/usr/bin/env python3
"""
Build TF-IDF artifacts for integrated_company_search.json

Creates:
  tfidf_vectorizer.joblib
  tfidf_matrix.npz
  tfidf_doc_index.json

Resolves paths from config['company_mapped_data']:
  - processed_data_store
  - tfidf_search_store (optional; defaults to <processed>/tfidf_search)
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Dict, List, Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.load_config import load_config


def _resolve_path(p: str | None, base: Path | None) -> Path:
    """Resolve a possibly-relative path against `base` (repo root)."""
    if not p:
        raise ValueError("Path not provided in config.")
    q = Path(p)
    if q.is_absolute():
        return q
    return (base / q) if base else (Path.cwd() / q)


def _repo_root_from_file() -> Path:
    """
    Best-effort repo root: assume this file is under src/<...>/company_tfidf_api.py.
    parents[2] usually points to project root (…/InsightDB-v2).
    """
    return Path(__file__).resolve().parents[2]


def _load_companies(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("companies"), list):
        return data["companies"]
    if isinstance(data, list):
        return data
    return []


def _make_card(c: dict) -> str:
    fields = [
        str(c.get("company_name", "")),
        str(c.get("llm_summary", "")),
        str(c.get("core_expertise", "")),
        str(c.get("industry_domain", "")),
        str(c.get("certifications", "")),
        str(c.get("address", "")),
        str(c.get("city", "")),
        str(c.get("state", "")),
        str(c.get("country", "")),
    ]
    return " | ".join([x for x in fields if x]).strip()


def company_tfidf_api():
    # Load project config and logger
    (config, logger) = load_config()
    repo_root = _repo_root_from_file()

    # Read company_mapped_data from config (dict or string fallback)
    cmd = config.get("company_mapped_data")
    if isinstance(cmd, dict):
        processed_rel = cmd.get("processed_data_store")
        tfidf_rel = cmd.get("tfidf_search_store")  # optional
    else:
        # If someone set company_mapped_data as a string path, accept it
        processed_rel = cmd
        tfidf_rel = None

    # Resolve absolute paths
    DATA_DIR = _resolve_path(processed_rel, repo_root)
    OUT_DIR = _resolve_path(tfidf_rel, repo_root) if tfidf_rel else (DATA_DIR / "tfidf_search")

    # Locate integrated JSON (primary + a common alternative)
    INTEGRATED = DATA_DIR / "integrated_company_search.json"
    if not INTEGRATED.exists():
        alt = DATA_DIR / "company_mapped_store" / "integrated_company_search.json"
        if alt.exists():
            INTEGRATED = alt

    # Create output dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    VECTORIZER_PATH = OUT_DIR / "tfidf_vectorizer.joblib"
    MATRIX_PATH = OUT_DIR / "tfidf_matrix.npz"
    INDEX_PATH = OUT_DIR / "tfidf_doc_index.json"

    logger.info(f"[TFIDF] DATA_DIR={DATA_DIR}")
    logger.info(f"[TFIDF] OUT_DIR={OUT_DIR}")
    logger.info(f"[TFIDF] INTEGRATED={INTEGRATED} | Exists={INTEGRATED.exists()}")

    if not INTEGRATED.exists():
        raise FileNotFoundError(
            f"integrated_company_search.json not found. Looked at:\n  {INTEGRATED}\n"
            f"Check config['company_mapped_data']['processed_data_store'] or pass absolute paths."
        )

    companies = _load_companies(INTEGRATED)
    n = len(companies)
    logger.info(f"[TFIDF] Loaded companies: {n}")
    if n == 0:
        raise RuntimeError("No companies found in integrated_company_search.json")

    cards = [_make_card(c) for c in companies]

    # TF-IDF config: good defaults for short 'cards'
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        min_df=1,
        max_df=0.95,
        strip_accents="unicode",
    )

    try:
        X = vec.fit_transform(cards)  # l2-normalized rows → cosine = dot
    except ValueError as e:
        # Common cause: empty vocabulary (all cards empty/stopwords or wrong file)
        msg = (
            "TF-IDF failed with 'empty vocabulary'. "
            "Ensure cards contain tokens and the integrated JSON is correct."
        )
        logger.error(f"[TFIDF] {msg}")
        raise

    # Persist artifacts
    joblib.dump(vec, VECTORIZER_PATH)
    sparse.save_npz(MATRIX_PATH, X)

    # Lightweight meta to map row -> display fields
    meta = []
    for i, c in enumerate(companies):
        meta.append({
            "row": i,
            "company_ref_no": c.get("company_ref_no"),
            "company_name": c.get("company_name"),
            "core_expertise": c.get("core_expertise"),
            "industry_domain": c.get("industry_domain"),
            "address": c.get("address"),
            "city": c.get("city"),
            "state": c.get("state"),
            "country": c.get("country"),
            "email": c.get("email") or c.get("poc_email"),
            "website": c.get("website"),
            "phone": c.get("phone"),
        })
    INDEX_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"[TFIDF] Saved:\n  {VECTORIZER_PATH}\n  {MATRIX_PATH}\n  {INDEX_PATH}")
    print(f"OK: TF-IDF built for {n} companies → {OUT_DIR}")


if __name__ == "__main__":
    try:
        company_tfidf_api()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

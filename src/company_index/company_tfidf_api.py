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

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional, Tuple, Union

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.load_config import load_config


# ---------------- Path helpers ----------------

def _resolve_path(p: Optional[Union[str, Path]], base: Optional[Path]) -> Path:
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


# ---------------- Data helpers ----------------

def _load_companies(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("companies"), list):
        return data["companies"]
    if isinstance(data, list):
        return data
    return []


def _make_card(c: dict) -> str:
    """Generate a searchable text card from company data with proper data structure handling."""
    company_details = c.get("CompanyDetails", {}) or {}
    products = (c.get("ProductsAndServices", {}) or {}).get("ProductList", []) or []
    qc = c.get("QualityAndCompliance", {}) or {}
    certifications = qc.get("CertificationsList", []) or []
    testing = qc.get("TestingCapabilitiesList", []) or []
    rd = (c.get("ResearchAndDevelopment", {}) or {}).get("RDCapabilitiesList", []) or []

    fields: List[str] = [
        str(company_details.get("company_name", "")),
        str(company_details.get("core_expertise", "")),
        str(company_details.get("industry_domain", "")),
        str(company_details.get("industry_subdomain", "")),
        str(company_details.get("address", "")),
        str(company_details.get("city", "")),
        str(company_details.get("state", "")),
        str(company_details.get("country", "")),
        str(company_details.get("company_scale", "")),
        str(company_details.get("organization_type", "")),
    ]

    for product in products:
        fields.extend([
            str(product.get("product_name", "")),
            str(product.get("product_description", "")),
            str(product.get("product_type", "")),
            str(product.get("salient_features", "")),
        ])

    for cert in certifications:
        fields.extend([
            str(cert.get("certification_detail", "")),
            str(cert.get("certification_type_master", "")),
        ])

    for test in testing:
        fields.extend([
            str(test.get("test_details", "")),
            str(test.get("test_category", "")),
            str(test.get("test_subcategory", "")),
        ])

    for rd_item in rd:
        fields.extend([
            str(rd_item.get("rd_details", "")),
            str(rd_item.get("rd_category", "")),
            str(rd_item.get("rd_subcategory", "")),
        ])

    # Filter out blanks and NaNs
    meaningful = [x.strip() for x in fields if x and isinstance(x, str) and x.strip() and x.strip().lower() != "nan"]
    return " | ".join(meaningful)


# ---------------- Core API ----------------

def _resolve_io_paths(
    input_json_file: Optional[Union[str, Path]]
) -> Tuple[Path, Path, Any]:
    """
    Resolve:
      - DATA_DIR (processed_data_store)
      - OUT_DIR (tfidf_search_store or default)
      - INTEGRATED file (explicit, default, or common alternative)
    Returns: (INTEGRATED, OUT_DIR, logger)
    Raises: FileNotFoundError/ValueError on issues.
    """
    (config, logger) = load_config()
    repo_root = _repo_root_from_file()

    cmd = config.get("company_mapped_data")
    if isinstance(cmd, dict):
        processed_rel = cmd.get("processed_data_store")
        tfidf_rel = cmd.get("tfidf_search_store")  # optional
    elif isinstance(cmd, (str, Path)):
        processed_rel = cmd
        tfidf_rel = None
    else:
        raise ValueError(
            "config['company_mapped_data'] must be a dict with keys "
            "('processed_data_store', optional 'tfidf_search_store') or a string path."
        )

    if not processed_rel:
        raise ValueError("Missing 'processed_data_store' in config['company_mapped_data'].")

    DATA_DIR = _resolve_path(processed_rel, repo_root)
    if DATA_DIR.is_file():
        DATA_DIR = DATA_DIR.parent
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Processed data directory not found: {DATA_DIR}")

    OUT_DIR = _resolve_path(tfidf_rel, repo_root) if tfidf_rel else (DATA_DIR / "tfidf_search")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates: List[Path] = []
    if input_json_file:
        candidates.append(Path(input_json_file))
    else:
        candidates.append(DATA_DIR / "integrated_company_search.json")
        candidates.append(DATA_DIR / "company_mapped_store" / "integrated_company_search.json")

    INTEGRATED = next((p for p in candidates if p.exists()), None)
    if not INTEGRATED:
        raise FileNotFoundError(
            "integrated_company_search.json not found. Tried:\n  " + "\n  ".join(str(p) for p in candidates)
        )

    logger.info(f"[TFIDF] DATA_DIR={DATA_DIR}")
    logger.info(f"[TFIDF] OUT_DIR={OUT_DIR}")
    logger.info(f"[TFIDF] INTEGRATED={INTEGRATED} | Exists={INTEGRATED.exists()}")

    return INTEGRATED, OUT_DIR, logger


def company_tfidf_api(input_json_file: Optional[Union[str, Path]] = None) -> None:
    INTEGRATED, OUT_DIR, logger = _resolve_io_paths(input_json_file)

    VECTORIZER_PATH = OUT_DIR / "tfidf_vectorizer.joblib"
    MATRIX_PATH = OUT_DIR / "tfidf_matrix.npz"
    INDEX_PATH = OUT_DIR / "tfidf_doc_index.json"

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
        X = vec.fit_transform(cards)  # l2-normalized rows -> cosine = dot
    except ValueError as e:
        # Common cause: empty vocabulary (all cards empty/stopwords or wrong file)
        logger.error("[TFIDF] empty vocabulary — check integrated JSON contents.")
        raise

    # Optionally shrink to float32 to save disk/mem
    X = X.astype(np.float32)

    # Persist artifacts
    joblib.dump(vec, VECTORIZER_PATH)
    sparse.save_npz(MATRIX_PATH, X)

    # Lightweight meta to map row -> display fields
    meta = []
    for i, c in enumerate(companies):
        d = c.get("CompanyDetails", {}) or {}
        meta.append({
            "row": i,
            "company_ref_no": d.get("company_ref_no"),
            "company_name": d.get("company_name"),
            "core_expertise": d.get("core_expertise"),
            "industry_domain": d.get("industry_domain"),
            "address": d.get("address"),
            "city": d.get("city"),
            "state": d.get("state"),
            "country": d.get("country"),
            "email": d.get("email") or d.get("poc_email"),
            "website": d.get("website"),
            "phone": d.get("phone"),
        })
    INDEX_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"[TFIDF] Matrix shape: {X.shape}, nnz={X.nnz}")
    logger.info(f"[TFIDF] Saved:\n  {VECTORIZER_PATH}\n  {MATRIX_PATH}\n  {INDEX_PATH}")
    print(f"OK: TF-IDF built for {n} companies -> {OUT_DIR}")


# ---------------- CLI ----------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TF-IDF artifacts for integrated company search JSON")
    p.add_argument(
        "-i", "--input-file",
        default=None,
        help="Optional path to integrated_company_search.json. "
             "If omitted, will search under processed_data_store."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        company_tfidf_api(args.input_file)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

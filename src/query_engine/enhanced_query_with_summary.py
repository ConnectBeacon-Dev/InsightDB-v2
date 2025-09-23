#!/usr/bin/env python3
"""
Enhanced Query Execution with Integrated Summarization - Single File Edition
----------------------------------------------------------------------------
Self-contained module implementing a hybrid search over an integrated companies
dataset with optional embedding re-ranker, TTL+mtime-cached config/data, O(1)
indexes, intent detection, and summarized results.

- Optional Embeddings: uses sentence-transformers if available; falls back to lexical.
- No hard dependency on your project modules; will try to import load_config/logger if present.
- Works directly with a JSON file at: {company_mapped_data}/integrated_company_search.json
  where company_mapped_data is provided via config["company_mapped_data"] or defaults to ./processed_data_store

Author: ChatGPT
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------- Optional pandas ------------------------------
try:
    import pandas as pd  # noqa: F401
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

# ---------------------- Optional embedding dependencies ---------------------
_EMBED_OK = True
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except Exception:
    _EMBED_OK = False
    np = None
    SentenceTransformer = None

# ----------------------------- Logging helpers ------------------------------
def _make_default_logger() -> logging.Logger:
    logger = logging.getLogger("enhanced_query_single")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(ch)
    return logger


# =============================== Data Models ================================

class ProcessingStrategy(str, Enum):
    EMBEDDING_PRIMARY = "embedding_primary"
    INTENT_ENHANCED = "intent_enhanced"
    INTENT_FALLBACK = "intent_fallback"
    HYBRID = "hybrid"
    ERROR_FALLBACK = "error_fallback"


@dataclass
class ContactInfo:
    email: str = ""
    website: str = ""
    phone: str = ""
    poc_email: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "email": self.email or self.poc_email,
            "website": self.website,
            "phone": self.phone,
        }


@dataclass
class Company:
    ref_no: str
    name: str
    contact_info: ContactInfo = field(default_factory=ContactInfo)
    # Domain / expertise fields
    domain: str = ""                  # core_expertise (primary)
    industry_domain: str = ""         # optional, if present
    # Location / address
    address: str = ""                 # full address line if present
    pincode: str = ""
    city: str = ""
    state: str = ""
    country: str = ""
    # Other facets
    certifications: List[str] = field(default_factory=list)
    rd_categories: List[str] = field(default_factory=list)
    testing_categories: List[str] = field(default_factory=list)
    # Ranking
    score: float = 0.0
    category: str = ""
    status: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_ref_no": self.ref_no,
            "company_name": self.name,
            "domain": self.domain,
            "industry_domain": self.industry_domain,
            "address": self.address,
            "pincode": self.pincode,
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "certifications": self.certifications,
            "rd_categories": self.rd_categories,
            "testing_categories": self.testing_categories,
            "score": self.score,
            **self.contact_info.to_dict(),
        }

@dataclass
class QueryMetadata:
    strategy: ProcessingStrategy
    confidence: float
    llm_validation_used: bool = False
    companies_count: int = 0
    products_count: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    fallback_used: bool = False
    intent_enhancement_used: bool = False
    timings_ms: Dict[str, float] = field(default_factory=dict)


@dataclass
class QueryResult:
    companies: List[Company]
    metadata: QueryMetadata
    raw_results: Dict = field(default_factory=dict)
    enhanced_summary: str = ""
    intent_info: Optional[Dict] = None


# ================================ Utilities =================================

_TOKEN = re.compile(r"[A-Za-z0-9_]{3,}")
def tokenize(s: str) -> List[str]:
    return _TOKEN.findall((s or "").lower())

def get_maybe_dotted(d: Dict, dotted: str, default=""):
    if not isinstance(d, dict): return default
    if dotted in d: return d.get(dotted, default)
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def as_iter(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return x
    return [x]


# ====================== Cached Configuration / Data =========================

class CachedConfigurationManager:
    """Thread-safe singleton-like config/data cache with TTL + mtime invalidation."""
    _instance: Optional["CachedConfigurationManager"] = None
    _class_lock: Lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Dict[str, Any], cache_ttl_seconds: int = 300, logger: Optional[logging.Logger] = None):
        if getattr(self, "_initialized", False):
            return
        self._lock = Lock()
        self._cache_lock = Lock()
        self.config = dict(config or {})
        self.cache_ttl = int(cache_ttl_seconds)
        self.logger = logger or _make_default_logger()

        self._intent_patterns_cache: Optional[Dict[str, List[str]]] = None
        self._integrated_search_cache: Optional[Dict[str, Any]] = None

        self._cache_timestamps: Dict[str, float] = {}
        self._file_mtimes: Dict[str, float] = {}
        self._initialized = True

    # ---- Paths ----
    def get_path(self, key: str) -> Optional[Path]:
        if key == "intent_mapping":
            v = self.config.get("intent_mapping")
            return Path(v) if v else None
        if key == "company_mapped_data":
            v = self.config.get("company_mapped_data", "./processed_data_store")
            return Path(v) if v else None
        if key == "domain_mapping":
            v = self.config.get("domain_mapping")
            return Path(v) if v else None
        if key == "config_dir":
            v = self.config.get("config_dir")
            return Path(v) if v else None
        return None

    def path_exists(self, key: str) -> bool:
        p = self.get_path(key)
        return bool(p and p.exists())

    # ---- Cache helpers ----
    def _mtime(self, p: Optional[Path]) -> float:
        try:
            return p.stat().st_mtime if p and p.exists() else -1.0
        except Exception:
            return -1.0

    def _is_valid(self, cache_key: str, watched: List[Tuple[str, Optional[Path]]]) -> bool:
        ts = self._cache_timestamps.get(cache_key)
        if ts is None or (time.time() - ts) >= self.cache_ttl:
            return False
        for label, path in watched:
            mt = self._mtime(path)
            if self._file_mtimes.get(label) != mt:
                return False
        return True

    def _mark_fresh(self, cache_key: str, watched: List[Tuple[str, Optional[Path]]]) -> None:
        self._cache_timestamps[cache_key] = time.time()
        for label, path in watched:
            self._file_mtimes[label] = self._mtime(path)

    # ---- Intent patterns ----
    def get_intent_patterns(self) -> Dict[str, List[str]]:
        with self._cache_lock:
            ipath = self.get_path("intent_mapping")
            watched = [("intent_mapping", ipath)]
            if self._intent_patterns_cache is not None and self._is_valid("intent_patterns", watched):
                return self._intent_patterns_cache
            try:
                if ipath and ipath.exists():
                    data = json.loads(ipath.read_text(encoding="utf-8"))
                    patterns: Dict[str, List[str]] = {}
                    for intent, spec in data.items():
                        if intent in ("intent_resolution_rules", "query_preprocessing"):
                            continue
                        bucket: List[str] = []
                        for _, kws in (spec.get("keywords") or {}).items():
                            bucket.extend(kws or [])
                        bucket.extend(spec.get("synonyms") or [])
                        bucket.extend(spec.get("confidence_boosters") or [])
                        patterns[intent] = sorted({x.strip().lower() for x in bucket if x})
                else:
                    patterns = self._fallback_patterns()
                self._intent_patterns_cache = patterns
            except Exception as e:
                self.logger.warning(f"[Config] Intent patterns load failed: {e}")
                self._intent_patterns_cache = self._fallback_patterns()
            self._mark_fresh("intent_patterns", watched)
            return self._intent_patterns_cache

    # ---- Integrated data ----
    def get_integrated_search_data(self) -> Dict[str, Any]:
        with self._cache_lock:
            cdir = self.get_path("company_mapped_data")
            ipath = (cdir / "integrated_company_search.json") if cdir else None
            watched = [("integrated", ipath)]
            if self._integrated_search_cache is not None and self._is_valid("integrated_search", watched):
                return self._integrated_search_cache
            try:
                if ipath and ipath.exists():
                    self._integrated_search_cache = json.loads(ipath.read_text(encoding="utf-8"))
                else:
                    self._integrated_search_cache = {"companies": []}
            except Exception as e:
                self.logger.warning(f"[Config] Integrated data load failed: {e}")
                self._integrated_search_cache = {"companies": []}
            self._mark_fresh("integrated_search", watched)
            return self._integrated_search_cache

    def get_integrated_companies(self) -> List[Dict[str, Any]]:
        data = self.get_integrated_search_data()
        if isinstance(data, dict) and isinstance(data.get("companies"), list):
            return data["companies"]
        if isinstance(data, list):
            return data
        return []

    @staticmethod
    def _fallback_patterns() -> Dict[str, List[str]]:
        return {
            "basic_info": ["company", "name", "registration"],
            "products_services": ["product", "service", "manufacture", "supplier"],
            "location": ["where", "located", "based", "in", "near"],
            "contact": ["contact", "email", "phone", "website"],
            "certifications": ["iso", "certified", "certification", "nabl"],
            "rd_capabilities": ["research", "r&d", "development"],
            "testing_capabilities": ["testing", "test", "lab"],
            "business_domain": ["expertise", "domain", "industry"],
        }


# ============================== Contact Extract =============================

class ContactExtractor:
    CONTACT_FIELD_MAPPING = {
        "email": ["CompanyMaster.EmailId", "CompanyMaster.POC_Email", "EmailId", "POC_Email", "email"],
        "website": ["CompanyMaster.Website", "Website", "website"],
        "phone": ["CompanyMaster.Phone", "Phone", "phone"],
        "poc_email": ["CompanyMaster.POC_Email", "POC_Email", "poc_email"],
    }

    def extract_contact_info(self, company_info: Dict) -> ContactInfo:
        ci = ContactInfo()
        if not company_info:
            return ci
        complete = company_info.get("complete_data", {})
        data_sources = [
            get_maybe_dotted(complete, "CompanyProfile.BasicInfo.data", []),
            get_maybe_dotted(complete, "CompanyProfile.ContactInfo.data", []),
            [company_info],
        ]
        for src in data_sources:
            if not src:
                continue
            item = src[0] if isinstance(src, list) else src
            if not isinstance(item, dict):
                continue
            for field, keys in self.CONTACT_FIELD_MAPPING.items():
                if getattr(ci, field):
                    continue
                for k in keys:
                    v = get_maybe_dotted(item, k, "")
                    if v:
                        setattr(ci, field, str(v))
                        break
        return ci


# =============================== Search Engine ==============================

class OptimizedSearchEngine:
    """In-memory O(1) indexes + optional embedding vectors (lazy)."""

    def __init__(self, cm: CachedConfigurationManager, logger: logging.Logger):
        self.cm = cm
        self.logger = logger
        self._built = False
        self._companies: List[Dict[str, Any]] = []

        # Indexes
        self.idx_loc: Dict[str, List[int]] = {}
        self.idx_dom: Dict[str, List[int]] = {}
        self.idx_cert: Dict[str, List[int]] = {}
        self.idx_kw: Dict[str, List[int]] = {}

        # Embeddings
        self._embed_enabled = _EMBED_OK
        self._model: Optional[SentenceTransformer] = None
        self._emb_matrix: Optional[Any] = None  # np.ndarray
        self._card_cache: List[str] = []

    # --------- Build ---------
    def ensure_built(self):
        if self._built:
            return
        self._companies = self.cm.get_integrated_companies()
        N = len(self._companies)
        self.idx_loc = {}
        self.idx_dom = {}
        self.idx_cert = {}
        self.idx_kw = {}
        for i, c in enumerate(self._companies):
            # Location
            for f in ("country", "state", "city", "district"):
                v = str(c.get(f, "")).strip().lower()
                if v:
                    self.idx_loc.setdefault(v, []).append(i)
            # Domain
            for f in ("core_expertise", "industry_domain", "business_domain"):
                for v in as_iter(c.get(f)):
                    vs = str(v).strip().lower()
                    if vs:
                        self.idx_dom.setdefault(vs, []).append(i)
            # Certifications
            for v in as_iter(c.get("certifications") or c.get("certification_type")):
                vs = str(v).strip().lower()
                if vs:
                    self.idx_cert.setdefault(vs, []).append(i)
            # Keywords
            bag = " ".join([
                str(c.get("company_name","")), str(c.get("llm_summary","")),
                str(c.get("core_expertise","")), str(c.get("industry_domain","")),
                str(c.get("address","")), str(c.get("city","")), str(c.get("state","")), str(c.get("country",""))
            ])
            for tok in set(tokenize(bag)):
                self.idx_kw.setdefault(tok, []).append(i)
        self._built = True
        self.logger.info(f"[Index] Built over {N} companies")

    # --------- Optional embeddings ---------
    def _ensure_embeddings(self):
        if not self._embed_enabled or self._emb_matrix is not None:
            return
        if SentenceTransformer is None or np is None:
            self._embed_enabled = False
            return
        # Build compact text cards
        self._card_cache = []
        for c in self._companies:
            card = " | ".join([
                str(c.get("company_name","")),
                str(c.get("llm_summary","")),
                str(c.get("core_expertise","")),
                str(c.get("industry_domain","")),
                str(c.get("city","")), str(c.get("state","")), str(c.get("country",""))
            ]).strip()
            self._card_cache.append(card)
        try:
            self._model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            X = self._model.encode(self._card_cache, normalize_embeddings=True, show_progress_bar=False)
            self._emb_matrix = np.asarray(X, dtype=np.float32)
            self.logger.info(f"[Embeddings] Precomputed matrix shape={self._emb_matrix.shape}")
        except Exception as e:
            self.logger.warning(f"[Embeddings] Disabled ({e})")
            self._embed_enabled = False
            self._model = None
            self._emb_matrix = None

    # --------- Lookups ---------
    def search_by_location(self, token: str) -> List[int]:
        self.ensure_built()
        return list(self.idx_loc.get(str(token).strip().lower(), []))

    def search_by_domain_terms(self, terms: Dict[str, List[str]]) -> List[int]:
        self.ensure_built()
        hits = []
        for _, arr in terms.items():
            for t in arr:
                hits.extend(self.idx_dom.get(t.strip().lower(), []))
                hits.extend(self.idx_cert.get(t.strip().lower(), []))
        # uniq keep order
        out = list(dict.fromkeys(hits))
        return out

    def search_by_keywords(self, tokens: List[str]) -> List[int]:
        self.ensure_built()
        hits = []
        for t in tokens:
            hits.extend(self.idx_kw.get(t.strip().lower(), []))
        return list(dict.fromkeys(hits))

    def rerank_with_embeddings(self, query: str, candidates: List[int], topk: int = 20) -> List[int]:
        """Cosine-sim rerank over candidates; falls back to identity if embeddings disabled."""
        self.ensure_built()
        self._ensure_embeddings()
        if not self._embed_enabled or not candidates:
            return candidates[:topk]
        try:
            qv = self._model.encode([query], normalize_embeddings=True)
            sub = np.asarray([self._emb_matrix[i] for i in candidates], dtype=np.float32)
            sims = (sub @ np.asarray(qv, dtype=np.float32).T).ravel()
            order = np.argsort(-sims)[:topk]
            return [candidates[i] for i in order]
        except Exception as e:
            self.logger.warning(f"[Embeddings] Rerank failed, fallback ({e})")
            return candidates[:topk]


# =============================== Intent Logic ===============================

class IntentProcessor:
    def __init__(self, cm: CachedConfigurationManager, logger: logging.Logger):
        self.cm = cm
        self.logger = logger
        self._cache: Dict[str, Tuple[str, ...]] = {}
        # Priority if present in mapping
        self._priority: List[str] = self._load_priority_order()

    def _load_priority_order(self) -> List[str]:
        try:
            ipath = self.cm.get_path("intent_mapping")
            if not ipath or not ipath.exists():
                return ["location", "company", "contact", "products_services", "business_domain", "certifications"]
            data = json.loads(ipath.read_text(encoding="utf-8"))
            return (data.get("intent_resolution_rules", {})
                        .get("multi_intent_handling", {})
                        .get("priority_order", [])) or []
        except Exception:
            return []

    def detect(self, query: str) -> Tuple[str, ...]:
        q = query.lower().strip()
        if q in self._cache:
            return self._cache[q]
        patterns = self.cm.get_intent_patterns()
        scores: Dict[str, int] = {}
        for intent, keys in patterns.items():
            s = sum(1 for k in keys if k in q)
            if s > 0:
                scores[intent] = s
        if not scores:
            out = ("basic_info", "products_services")
        else:
            ordered = sorted(scores.items(), key=lambda kv: (
                self._priority.index(kv[0]) if kv[0] in self._priority else 999, -kv[1]
            ))
            out = tuple([k for k, _ in ordered[:3]])
        self._cache[q] = out
        return out


# ============================== Result Building =============================

class ResultProcessor:
    def __init__(self, cm: CachedConfigurationManager, logger: logging.Logger):
        self.cm = cm
        self.logger = logger
        self.contact = ContactExtractor()

    def _ref_from_company(self, d: Dict, idx: int) -> str:
        if d.get("company_ref_no"): return str(d["company_ref_no"])
        name = d.get("company_name") or d.get("name") or f"Unknown_{idx+1}"
        m = re.search(r"Company_(\d+)", str(name))
        if m: return f"CMP{m.group(1).zfill(3)}"
        return f"UNK{idx+1:03d}"

    def from_integrated(self, d: Dict, idx: int) -> Company:
        ref_no = self._ref_from_company(d, idx)
        ci = ContactInfo(
            email=str(d.get("email","") or d.get("poc_email","")),
            website=str(d.get("website","")),
            phone=str(d.get("phone","")),
            poc_email=str(d.get("poc_email","")),
        )
        # Support strings or lists
        def _as_list(x):
            if x is None: return []
            if isinstance(x, (list, tuple, set)): return [str(v) for v in x]
            return [str(x)]

        return Company(
            ref_no=ref_no,
            name=str(d.get("company_name","Unknown")),
            contact_info=ci,
            domain=str(d.get("core_expertise","")),
            industry_domain=str(d.get("industry_domain","")),
            address=str(d.get("address","")),
            pincode=str(d.get("pincode","")),
            city=str(d.get("city","")),
            state=str(d.get("state","")),
            country=str(d.get("country","")),
            certifications=_as_list(d.get("certifications") or d.get("certification_type")),
            rd_categories=_as_list(d.get("rd_category")),
            testing_categories=_as_list(d.get("test_category")),
            score=float(d.get("score", d.get("tfidf_score", 1.0)) or 1.0),
        )


# ============================= Strategy Helpers =============================

def compute_confidence(prefilter_hits: int, embed_used: bool) -> float:
    # Simple bounded heuristic
    base = min(1.0, prefilter_hits / 1000.0)  # lexical evidence
    if embed_used:
        return min(1.0, 0.6 * 0.75 + 0.4 * base)  # pretend avg embed score ~0.75
    return min(1.0, 0.4 * base)

def choose_strategy(intents: Tuple[str, ...], confidence: float, has_loc_or_company: bool) -> ProcessingStrategy:
    if confidence >= 0.5:
        return ProcessingStrategy.HYBRID if has_loc_or_company else ProcessingStrategy.EMBEDDING_PRIMARY
    if 0.3 <= confidence < 0.5:
        return ProcessingStrategy.INTENT_ENHANCED
    return ProcessingStrategy.INTENT_FALLBACK


# ============================ Main Orchestrator =============================

class EnhancedQueryProcessor:
    CONFIDENCE_THRESHOLD = 0.5
    ENHANCEMENT_THRESHOLD = 0.3

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or _make_default_logger()
        self.cm = CachedConfigurationManager(config, logger=self.logger)
        self.intent = IntentProcessor(self.cm, self.logger)
        self.engine = OptimizedSearchEngine(self.cm, self.logger)
        self.builder = ResultProcessor(self.cm, self.logger)

    # ---- Query flow ----
    def process_query(self, user_query: str, topk: int = 20) -> QueryResult:
        t0 = time.time()
        timings = {}

        self.engine.ensure_built()
        timings["index_ready_ms"] = (time.time() - t0) * 1000

        # 1) intents
        t1 = time.time()
        intents = self.intent.detect(user_query)
        timings["intent_ms"] = (time.time() - t1) * 1000

        # 2) prefilter
        t2 = time.time()
        tokens = [t for t in tokenize(user_query) if len(t) > 2]
        loc_hits = []
        for tok in tokens:
            loc_hits += self.engine.search_by_location(tok)
        dom_hits = self.engine.search_by_domain_terms({"_": tokens})
        kw_hits = self.engine.search_by_keywords(tokens)
        pre = list(dict.fromkeys(loc_hits + dom_hits + kw_hits))
        timings["prefilter_ms"] = (time.time() - t2) * 1000

        # 3) embeddings (optional) re-rank
        t3 = time.time()
        ranked = self.engine.rerank_with_embeddings(user_query, pre, topk=topk)
        timings["rerank_ms"] = (time.time() - t3) * 1000
        embed_used = _EMBED_OK and ranked is not None

        # 4) confidence & strategy
        confidence = compute_confidence(len(pre), embed_used)
        has_loc_or_company = any(x in intents for x in ("location", "company"))
        strategy = choose_strategy(intents, confidence, has_loc_or_company)

        # 5) materialize companies
        t4 = time.time()
        if strategy == ProcessingStrategy.EMBEDDING_PRIMARY:
            ids = ranked[:topk]
        elif strategy == ProcessingStrategy.HYBRID:
            ids = ranked[:topk]
        elif strategy == ProcessingStrategy.INTENT_ENHANCED:
            ids = pre[:topk] if pre else kw_hits[:topk]
        else:
            ids = kw_hits[:topk]

        companies_raw = self.cm.get_integrated_companies()
        companies = [self.builder.from_integrated(companies_raw[i], i) for i in ids if 0 <= i < len(companies_raw)]
        # small boost for multi-hit
        seen_counts: Dict[str, int] = {}
        for tok in tokens:
            for idx in self.engine.search_by_keywords([tok]):
                if idx in ids:
                    key = companies_raw[idx].get("company_ref_no") or str(idx)
                    seen_counts[key] = seen_counts.get(key, 0) + 1
        for c in companies:
            if seen_counts.get(c.ref_no, 0) > 2:
                c.score += 0.2

        timings["materialize_ms"] = (time.time() - t4) * 1000
        total = time.time() - t0

        md = QueryMetadata(
            strategy=strategy,
            confidence=confidence,
            companies_count=len(companies),
            processing_time=total,
            intent_enhancement_used=(strategy in (ProcessingStrategy.INTENT_ENHANCED, ProcessingStrategy.HYBRID, ProcessingStrategy.INTENT_FALLBACK)),
            timings_ms=timings,
        )

        summary = self._summarize(companies, intents)
        intent_answer = self._intent_answer(companies, intents)
        return QueryResult(
            companies=companies,
            metadata=md,
            enhanced_summary=summary,
            raw_results={"prefilter": len(pre)},
            intent_info={"intents": intents, "answer": intent_answer}
        )
    # ---- Helpers ----
    def _summarize(self, companies: List[Company], intents: Tuple[str, ...]) -> str:
        if not companies:
            return "No companies matched your query."
        top = ", ".join([c.name for c in companies[:3]])
        extra = []
        if "location" in intents:
            places = []
            for c in companies[:5]:
                loc = ", ".join([x for x in [c.city, c.state, c.country] if x])
                if loc:
                    places.append(loc)
            if places:
                extra.append("Locations: " + ", ".join(sorted(set(places))[:3]))
        have_contact = sum(1 for c in companies if c.contact_info.email or c.contact_info.website)
        if have_contact:
            extra.append(f"Contact info for {have_contact} companies")
        return f"Found {len(companies)} companies. Top matches: {top}. " + " ".join(extra)

    def _intent_answer(self, companies: List[Company], intents: Tuple[str, ...], topk: int = 5) -> str:
        if not companies:
            return "No matching companies found."

        lines = []
        show_location = ("location" in intents)
        show_domain   = ("business_domain" in intents) or ("products_services" in intents) or ("basic_info" in intents)

        # Prefer fewer, clearer rows
        for c in companies[:topk]:
            parts = [f"{c.name} [{c.ref_no}]"]

            if show_location:
                addr_bits = [c.address, c.city, c.state, c.country]
                if c.pincode: addr_bits.append(c.pincode)
                addr = ", ".join([x for x in addr_bits if x])
                if addr:
                    parts.append(f"Address: {addr}")

            if show_domain:
                dom_bits = []
                if c.domain:
                    dom_bits.append(f"Core expertise: {c.domain}")
                if c.industry_domain:
                    dom_bits.append(f"Industry: {c.industry_domain}")
                if dom_bits:
                    parts.append("; ".join(dom_bits))

            # Always useful: contact
            if c.contact_info.email or c.contact_info.website or c.contact_info.phone:
                contact_bits = []
                if c.contact_info.email:   contact_bits.append(f"Email: {c.contact_info.email}")
                if c.contact_info.website: contact_bits.append(f"Website: {c.contact_info.website}")
                if c.contact_info.phone:   contact_bits.append(f"Phone: {c.contact_info.phone}")
                parts.append(" | ".join(contact_bits))

            # Optional: certs
            if "certifications" in intents and c.certifications:
                parts.append("Certifications: " + ", ".join(c.certifications))

            # Optional: RD / testing
            if "rd_capabilities" in intents and c.rd_categories:
                parts.append("R&D: " + ", ".join(c.rd_categories))
            if "testing_capabilities" in intents and c.testing_categories:
                parts.append("Testing: " + ", ".join(c.testing_categories))

            lines.append(" - " + "  ·  ".join(parts))

        header = []
        if show_location: header.append("Location details")
        if show_domain:   header.append("Core expertise")
        if not header:    header.append("Company details")

        return f"{' & '.join(header)}:\n" + "\n".join(lines)

# ============================= Public API / CLI =============================

def _try_project_load_config():
    try:
        from src.load_config import load_config as _lc, get_company_mapped_data_processed_data_store  # type: ignore
        cfg, logger = _lc()
        # Map from the project's config structure to what this module expects
        if cfg:
            # The project config has company_mapped_data as a dict, but this module expects a string path
            try:
                processed_data_store_path = get_company_mapped_data_processed_data_store(cfg)
                cfg["company_mapped_data"] = str(processed_data_store_path)
            except Exception:
                # Fallback if the function fails
                if isinstance(cfg.get("company_mapped_data"), dict):
                    cfg["company_mapped_data"] = cfg["company_mapped_data"].get("processed_data_store", "./processed_data_store/company_mapped_store")
                else:
                    cfg["company_mapped_data"] = "./processed_data_store/company_mapped_store"
        return cfg, logger
    except Exception:
        return None, None

def execute_enhanced_query_with_summary(user_query: str, config: Optional[Dict[str, Any]] = None,
                                        logger: Optional[logging.Logger] = None,
                                        topk: int = 20) -> Dict[str, Any]:
    if config is None or logger is None:
        cfg2, log2 = _try_project_load_config()
        config = config or (cfg2 or {"company_mapped_data": "./processed_data_store"})
        logger = logger or (log2 or _make_default_logger())

    proc = EnhancedQueryProcessor(config, logger)
    res = proc.process_query(user_query, topk=topk)

    return {
        "query": user_query,
        "strategy": res.metadata.strategy.value,
        "confidence": res.metadata.confidence,
        "processing_time": res.metadata.processing_time,
        "intent_enhancement_used": res.metadata.intent_enhancement_used,
        "timings_ms": res.metadata.timings_ms,
        "results": {
            "companies": [c.to_dict() for c in res.companies],
            "companies_count": res.metadata.companies_count,
            "products_count": res.metadata.products_count,
        },
        "company_contacts": [c.to_dict() for c in res.companies],
        "enhanced_summary": res.enhanced_summary,
        "intent_answer": (res.intent_info or {}).get("answer", ""),
        "raw_results": res.raw_results,
    }



def _demo_queries():
    return [
        #companies having ISO certification with testing facilities",
        #"electrical companies in Sweden",
        #"Company_036 location",
        #"small scale manufacturing companies",
        #"r&d capabilities in power systems",
        "list all companies in State5"
    ]

def main():
    # Try to use project config first, fallback to hardcoded if needed
    config, logger = _try_project_load_config()
    if config is None or logger is None:
        logger = _make_default_logger()
        config = {
            "company_mapped_data": "./processed_data_store/company_mapped_store",
            "intent_mapping": "./config/intent_mapping.json",
            "domain_mapping": "./config/domain_mapping.json",
        }
    logger.info("Enhanced Query Execution - Single File")
    for q in _demo_queries():
        out = execute_enhanced_query_with_summary(q, config=config, logger=logger)
        logger.info(f"Query: {q}")
        logger.info(f"  Strategy={out['strategy']} conf={out['confidence']:.2f} time={out['processing_time']:.3f}s")
        logger.info(f"  Summary: {out['enhanced_summary']}")
        logger.info(f"  Companies: {len(out['results']['companies'])}")
        logger.info("-"*60)

if __name__ == "__main__":
    main()

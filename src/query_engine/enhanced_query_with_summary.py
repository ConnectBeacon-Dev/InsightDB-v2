#!/usr/bin/env python3
"""
Enhanced Query Execution with Integrated Summarization - Single File (TF-IDF + Embeddings + RRF)
-------------------------------------------------------------------------------------------------
- Per-company in-memory indexes (location/domain/keywords)
- Optional Embeddings (sentence-transformers)
- Optional TF-IDF (scikit-learn), fused with embeddings via Reciprocal Rank Fusion (RRF)
- Intent-aware output (location -> address; domain -> core expertise; plus contacts/certs/R&D/testing)

Configure with:
  config = {
      "company_data": "<DIR containing integrated_company_search.json>",
      "intent_mapping": "<optional path>",
      "domain_mapping": "<optional path>"
  }

The integrated JSON can be either:
  {"companies": [ ... company dicts ... ]}
or a plain list: [ ... ]

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

# ----------------------------- Optional TF-IDF ------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _TFIDF_OK = True
except Exception:
    _TFIDF_OK = False
    TfidfVectorizer = None

try:
    import joblib
    from scipy import sparse
    _SPARSE_OK = True
except Exception:
    joblib = None
    sparse = None
    _SPARSE_OK = False

# ----------------------------- Optional LLM Enhancement ---------------------
try:
    from .llm_query_enhancer import LocalLLMQueryEnhancer
    _LLM_ENHANCER_OK = True
except Exception:
    _LLM_ENHANCER_OK = False
    LocalLLMQueryEnhancer = None
    
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
    products: List[str] = field(default_factory=list)  # Product list
    # Ranking
    score: float = 0.0
    category: str = ""
    status: str = ""

    def to_dict(self, include_location: bool = False, include_domain: bool = False, include_certifications: bool = False, include_products: bool = False) -> Dict[str, Any]:
        # Build result dict with essential company information
        result = {
            "company_name": self.name,
            "company_ref_no": self.ref_no,
        }
        
        # Add contact info only if fields have values
        contact_dict = self.contact_info.to_dict()
        
        # Only include website, email, and phone if they have values
        if contact_dict.get("website") and str(contact_dict["website"]).strip():
            result["website"] = contact_dict["website"]
        if contact_dict.get("email") and str(contact_dict["email"]).strip():
            result["email"] = contact_dict["email"]
        if contact_dict.get("phone") and str(contact_dict["phone"]).strip():
            result["phone"] = contact_dict["phone"]
        
        # Add location information if requested
        if include_location:
            location_parts = []
            if self.address and str(self.address).strip():
                location_parts.append(str(self.address).strip())
            if self.city and str(self.city).strip():
                location_parts.append(str(self.city).strip())
            if self.state and str(self.state).strip():
                location_parts.append(str(self.state).strip())
            if self.country and str(self.country).strip():
                location_parts.append(str(self.country).strip())
            if self.pincode and str(self.pincode).strip():
                location_parts.append(str(self.pincode).strip())
            
            if location_parts:
                result["location"] = ", ".join(location_parts)
        
        # Add domain information if requested
        if include_domain:
            if self.domain and str(self.domain).strip():
                result["domain"] = str(self.domain).strip()
            if self.industry_domain and str(self.industry_domain).strip():
                result["industry_domain"] = str(self.industry_domain).strip()
        
        # Add certifications if requested
        if include_certifications and self.certifications:
            result["certifications"] = self.certifications
        
        # Add products if requested
        if include_products and self.products:
            result["products"] = self.products
                
        return result


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
    if isinstance(x, (list, tuple, set)): return list(x)
    return [x]

def rrf_merge(rank_lists: List[List[int]], k: int = 60) -> List[int]:
    """Reciprocal Rank Fusion over multiple ranked lists."""
    score: Dict[int, float] = {}
    for lst in rank_lists:
        for r, doc in enumerate(lst):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + r + 1)
    return [d for d, _ in sorted(score.items(), key=lambda kv: -kv[1])]


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
        if key == "company_data":
            v = self.config.get("company_data", "./processed_data_store")
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

    def get_tfidf_store(self) -> Optional[Path]:
        """
        Resolve TF-IDF store directory.
        Priority:
        1) config['company_mapped_data']['tfidf_search_store']
        2) <company_data>/tfidf_search
        """
        try:
            cm = self.config.get("company_mapped_data", {})
            if isinstance(cm, dict) and cm.get("tfidf_search_store"):
                p = Path(cm["tfidf_search_store"])
                return p if p.exists() else p  # return even if missing; we'll check files later
        except Exception:
            pass
        cdir = self.get_path("company_data")
        if cdir:
            return cdir / "tfidf_search"
        return None

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
            cdir = self.get_path("company_data")
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
    """In-memory O(1) indexes + optional embedding vectors + TF-IDF (lazy)."""

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
        self.idx_company_names: Dict[str, List[int]] = {}  # New: exact company name index

        # Cards (shared by embeddings/TF-IDF)
        self._card_cache: List[str] = []

        # Embeddings
        self._embed_enabled = _EMBED_OK
        self._model: Optional[SentenceTransformer] = None
        self._emb_matrix: Optional[Any] = None  # np.ndarray

        # TF-IDF
        self._tfidf: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None

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
        self.idx_company_names = {}  # Initialize company name index
        for i, c in enumerate(self._companies):
            # Extract company details from nested structure
            company_details = c.get("CompanyDetails", {})
            
            # Company name indexing for exact matches
            company_name = str(company_details.get("company_name", "")).strip()
            if company_name and company_name.lower() != "nan":
                # Index exact company name (case-insensitive)
                name_key = company_name.lower()
                self.idx_company_names.setdefault(name_key, []).append(i)
                
                # Also index individual words in company name for partial matches
                name_words = company_name.lower().split()
                for word in name_words:
                    if len(word) > 2:  # Skip very short words
                        self.idx_company_names.setdefault(word, []).append(i)
            
            # Location
            for f in ("country", "state", "city", "district"):
                v = str(company_details.get(f, "")).strip().lower()
                if v:
                    self.idx_loc.setdefault(v, []).append(i)
            # Domain
            for f in ("core_expertise", "industry_domain", "business_domain"):
                for v in as_iter(company_details.get(f)):
                    vs = str(v).strip().lower()
                    if vs:
                        self.idx_dom.setdefault(vs, []).append(i)
            
            # Certifications from QualityAndCompliance section
            certifications = c.get("QualityAndCompliance", {}).get("CertificationsList", [])
            for cert in certifications:
                cert_type = cert.get("certification_type_master", "")
                cert_detail = cert.get("certification_detail", "")
                for cert_val in [cert_type, cert_detail]:
                    vs = str(cert_val).strip().lower()
                    if vs and vs != "nan":
                        self.idx_cert.setdefault(vs, []).append(i)
                        # Also index individual words for better matching
                        for word in vs.split():
                            if len(word) > 2:
                                self.idx_cert.setdefault(word, []).append(i)
            
            # Testing capabilities from QualityAndCompliance section
            testing = c.get("QualityAndCompliance", {}).get("TestingCapabilitiesList", [])
            for test in testing:
                test_cat = test.get("test_category", "")
                test_subcat = test.get("test_subcategory", "")
                test_details = test.get("test_details", "")
                for test_val in [test_cat, test_subcat, test_details]:
                    vs = str(test_val).strip().lower()
                    if vs and vs != "nan":
                        self.idx_cert.setdefault(vs, []).append(i)
                        # Also index individual words
                        for word in vs.split():
                            if len(word) > 2:
                                self.idx_cert.setdefault(word, []).append(i)
            
            # R&D capabilities from ResearchAndDevelopment section
            rd_capabilities = c.get("ResearchAndDevelopment", {}).get("RDCapabilitiesList", [])
            for rd in rd_capabilities:
                rd_cat = rd.get("rd_category", "")
                rd_subcat = rd.get("rd_subcategory", "")
                rd_details = rd.get("rd_details", "")
                for rd_val in [rd_cat, rd_subcat, rd_details]:
                    vs = str(rd_val).strip().lower()
                    if vs and vs != "nan":
                        self.idx_cert.setdefault(vs, []).append(i)
                        # Also index individual words
                        for word in vs.split():
                            if len(word) > 2:
                                self.idx_cert.setdefault(word, []).append(i)
            
            # Keywords from all relevant fields
            bag = " ".join([
                str(company_details.get("company_name","")), 
                str(company_details.get("core_expertise","")), 
                str(company_details.get("industry_domain","")),
                str(company_details.get("industry_subdomain","")),
                str(company_details.get("address","")), 
                str(company_details.get("city","")), 
                str(company_details.get("state","")), 
                str(company_details.get("country","")),
                str(company_details.get("company_scale","")),
                str(company_details.get("organization_type",""))
            ])
            for tok in set(tokenize(bag)):
                self.idx_kw.setdefault(tok, []).append(i)
            
            # Also index company scale separately for better matching
            scale = str(company_details.get("company_scale", "")).strip().lower()
            if scale and scale != "nan":
                self.idx_kw.setdefault(scale, []).append(i)
                # Add scale synonyms for better matching
                scale_synonyms = {
                    "small": ["small", "micro"],
                    "medium": ["medium", "mid"],
                    "large": ["large", "big"],
                    "enterprise": ["enterprise", "big", "large"]
                }
                for synonym_group, synonyms in scale_synonyms.items():
                    if scale in synonyms:
                        for syn in synonyms:
                            self.idx_kw.setdefault(syn, []).append(i)

        # Build compact text cards once (shared by embeddings & TF-IDF)
        self._card_cache = []
        for c in self._companies:
            company_details = c.get("CompanyDetails", {})
            products = c.get("ProductsAndServices", {}).get("ProductList", [])
            certifications = c.get("QualityAndCompliance", {}).get("CertificationsList", [])
            testing = c.get("QualityAndCompliance", {}).get("TestingCapabilitiesList", [])
            rd = c.get("ResearchAndDevelopment", {}).get("RDCapabilitiesList", [])
            
            # Build comprehensive text from all available data (same as TF-IDF API)
            fields = [
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
            
            # Add product information
            for product in products:
                fields.extend([
                    str(product.get("product_name", "")),
                    str(product.get("product_description", "")),
                    str(product.get("product_type", "")),
                    str(product.get("salient_features", "")),
                ])
            
            # Add certification information
            for cert in certifications:
                fields.extend([
                    str(cert.get("certification_detail", "")),
                    str(cert.get("certification_type_master", "")),
                ])
            
            # Add testing capabilities
            for test in testing:
                fields.extend([
                    str(test.get("test_details", "")),
                    str(test.get("test_category", "")),
                    str(test.get("test_subcategory", "")),
                ])
            
            # Add R&D capabilities
            for rd_item in rd:
                fields.extend([
                    str(rd_item.get("rd_details", "")),
                    str(rd_item.get("rd_category", "")),
                    str(rd_item.get("rd_subcategory", "")),
                ])
            
            # Filter out empty strings and "nan" values
            meaningful_fields = [x for x in fields if x and x.lower() != "nan" and x.strip()]
            card = " | ".join(meaningful_fields).strip()
            self._card_cache.append(card)

        self._built = True
        self.logger.info(f"[Index] Built over {N} companies")

    # --------- Optional embeddings ---------
    def _ensure_embeddings(self):
        if not self._embed_enabled or self._emb_matrix is not None:
            return
        if SentenceTransformer is None or np is None or not self._card_cache:
            self._embed_enabled = False
            return
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

    # --------- Optional TF-IDF ---------
    def _ensure_tfidf(self):
        """
        Ensure TF-IDF is ready.
        Try to LOAD prebuilt files from disk (vectorizer + sparse matrix).
        If missing or load fails, FIT in-memory from _card_cache.
        """
        if self._tfidf_matrix is not None:
            return

        # Try disk load first
        tfidf_dir = self.cm.get_tfidf_store()
        vec_path = mat_path = None
        if _TFIDF_OK and _SPARSE_OK and tfidf_dir:
            vec_path = tfidf_dir / "tfidf_vectorizer.joblib"
            mat_path = tfidf_dir / "tfidf_matrix.npz"
            if vec_path.exists() and mat_path.exists():
                try:
                    self._tfidf = joblib.load(vec_path)  # type: ignore[arg-type]
                    self._tfidf_matrix = sparse.load_npz(mat_path)  # type: ignore[attr-defined]
                    self.logger.debug(f"[TF-IDF] Loaded prebuilt index from {tfidf_dir}")
                    return
                except Exception as e:
                    self.logger.warning(f"[TF-IDF] Prebuilt load failed ({e}); will fit in-memory")

        # Fallback: fit in-memory
        if not _TFIDF_OK:
            self.logger.warning("[TF-IDF] scikit-learn not available; skipping")
            return

        if not self._card_cache:
            self.ensure_built()

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # local import safety
            self._tfidf = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                lowercase=True,
                min_df=1,
                max_df=0.95,
                strip_accents="unicode",
            )
            self._tfidf_matrix = self._tfidf.fit_transform(self._card_cache)
            self.logger.info("[TF-IDF] Fitted in-memory vectorizer (no prebuilt files found)")
        except Exception as e:
            self.logger.warning(f"[TF-IDF] Disabled ({e})")
            self._tfidf = None
            self._tfidf_matrix = None

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

    def search_by_company_name(self, query: str) -> List[int]:
        """
        Search for companies by exact name match or partial name match.
        This is the first strategy to try for company name queries.
        """
        self.ensure_built()
        query_lower = query.strip().lower()
        
        # Try exact match first
        exact_matches = self.idx_company_names.get(query_lower, [])
        if exact_matches:
            self.logger.debug(f"[CompanyName] Found exact match for '{query}': {len(exact_matches)} companies")
            return exact_matches
        
        # Try partial matches by searching individual words
        query_words = query_lower.split()
        if len(query_words) > 1:
            # For multi-word queries, find companies that match all words
            word_matches = []
            for word in query_words:
                if len(word) > 2:  # Skip very short words
                    matches = self.idx_company_names.get(word, [])
                    word_matches.append(set(matches))
            
            if word_matches:
                # Find intersection of all word matches
                common_matches = set.intersection(*word_matches) if len(word_matches) > 1 else word_matches[0]
                if common_matches:
                    result = list(common_matches)
                    self.logger.debug(f"[CompanyName] Found partial matches for '{query}': {len(result)} companies")
                    return result
        
        # Single word or no multi-word matches found
        single_word_matches = []
        for word in query_words:
            if len(word) > 2:
                single_word_matches.extend(self.idx_company_names.get(word, []))
        
        if single_word_matches:
            result = list(dict.fromkeys(single_word_matches))  # Remove duplicates while preserving order
            self.logger.debug(f"[CompanyName] Found single-word matches for '{query}': {len(result)} companies")
            return result
        
        self.logger.debug(f"[CompanyName] No company name matches found for '{query}'")
        return []

    def rerank_with_embeddings(self, query: str, candidates: List[int], topk: int = 20) -> List[int]:
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

    def tfidf_rank(self, query: str, candidates: List[int], topk: int = 50) -> List[int]:
        self.ensure_built()
        self._ensure_tfidf()
        if not _TFIDF_OK or self._tfidf_matrix is None or not candidates:
            return candidates[:topk]
        try:
            import numpy as np  # local import
            qv = self._tfidf.transform([query])
            cand_rows = self._tfidf_matrix[candidates]
            sims = cand_rows @ qv.T   # (len(candidates), 1)
            sims = np.asarray(sims.todense()).ravel()
            order = np.argsort(-sims)[:topk]
            return [candidates[i] for i in order]
        except Exception as e:
            self.logger.warning(f"[TF-IDF] Rank failed, fallback ({e})")
            return candidates[:topk]


# =============================== Intent Logic ===============================

class IntentProcessor:
    def __init__(self, cm: CachedConfigurationManager, logger: logging.Logger):
        self.cm = cm
        self.logger = logger
        self._cache: Dict[str, Tuple[str, ...]] = {}
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
        
        # Special handling for location queries - if query contains "location" word, force location intent
        if "location" in q:
            scores["location"] = scores.get("location", 0) + 10  # Boost location score
        
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
        if m: return m.group(1).zfill(3)  # Return just the number without any prefix
        return f"UNK{idx+1:03d}"

    def from_integrated(self, d: Dict, idx: int) -> Company:
        # Extract data from nested structure
        company_details = d.get("CompanyDetails", {})
        certifications = d.get("QualityAndCompliance", {}).get("CertificationsList", [])
        testing = d.get("QualityAndCompliance", {}).get("TestingCapabilitiesList", [])
        rd = d.get("ResearchAndDevelopment", {}).get("RDCapabilitiesList", [])
        
        ref_no = company_details.get("company_ref_no", f"UNK{idx+1:03d}")
        
        ci = ContactInfo(
            email=str(company_details.get("email","") or company_details.get("poc_email","")),
            website=str(company_details.get("website","")),
            phone=str(company_details.get("phone","")),
            poc_email=str(company_details.get("poc_email","")),
        )
        
        def _as_list(x):
            if x is None: return []
            if isinstance(x, (list, tuple, set)): return [str(v) for v in x]
            return [str(x)]
        
        # Extract certifications
        cert_list = []
        for cert in certifications:
            cert_type = cert.get("certification_type_master", "")
            cert_detail = cert.get("certification_detail", "")
            if cert_type and cert_type != "nan":
                cert_list.append(cert_type)
            if cert_detail and cert_detail != "nan" and cert_detail != cert_type:
                cert_list.append(cert_detail)
        
        # Extract R&D categories
        rd_list = []
        for rd_item in rd:
            rd_cat = rd_item.get("rd_category", "")
            rd_subcat = rd_item.get("rd_subcategory", "")
            if rd_cat and rd_cat != "nan":
                rd_list.append(rd_cat)
            if rd_subcat and rd_subcat != "nan" and rd_subcat != rd_cat:
                rd_list.append(rd_subcat)
        
        # Extract testing categories
        test_list = []
        for test_item in testing:
            test_cat = test_item.get("test_category", "")
            test_subcat = test_item.get("test_subcategory", "")
            if test_cat and test_cat != "nan":
                test_list.append(test_cat)
            if test_subcat and test_subcat != "nan" and test_subcat != test_cat:
                test_list.append(test_subcat)
        
        # Extract products
        products = d.get("ProductsAndServices", {}).get("ProductList", [])
        product_list = []
        for product in products:
            product_name = product.get("product_name", "")
            if product_name and product_name != "nan":
                product_list.append(product_name)
        
        return Company(
            ref_no=ref_no,
            name=str(company_details.get("company_name","Unknown")),
            contact_info=ci,
            domain=str(company_details.get("core_expertise","")),
            industry_domain=str(company_details.get("industry_domain","")),
            address=str(company_details.get("address","")),
            pincode=str(company_details.get("pincode","")),
            city=str(company_details.get("city","")),
            state=str(company_details.get("state","")),
            country=str(company_details.get("country","")),
            certifications=cert_list,
            rd_categories=rd_list,
            testing_categories=test_list,
            products=product_list,
            score=1.0,  # Default score
        )


# ============================= Strategy Helpers =============================

def compute_confidence(prefilter_hits: int, embed_used: bool, tfidf_used: bool) -> float:
    # Simple bounded heuristic
    base = min(1.0, prefilter_hits / 1000.0)  # lexical evidence
    uplift = 0.0
    if embed_used: uplift += 0.35
    if tfidf_used: uplift += 0.25
    return min(1.0, 0.4 * base + uplift)

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
        
        # Initialize optional LLM enhancer
        self.llm_enhancer = None
        if _LLM_ENHANCER_OK:
            try:
                self.llm_enhancer = LocalLLMQueryEnhancer(config, logger)
                if self.llm_enhancer.is_available():
                    self.logger.info("[LLM] Query enhancer initialized successfully")
                else:
                    self.logger.info("[LLM] Query enhancer available but model not loaded (fallback mode)")
            except Exception as e:
                self.logger.warning(f"[LLM] Failed to initialize enhancer: {e}")
                self.llm_enhancer = None
        else:
            self.logger.info("[LLM] Query enhancer not available (missing dependencies)")

    def _intent_answer(self, companies: List[Company], intents: Tuple[str, ...], topk: int = None) -> str:
        if not companies:
            return "No matching companies found."

        lines = []
        show_location = ("location" in intents)
        # Only show domain info if location is NOT the primary intent
        show_domain   = not show_location and (("business_domain" in intents) or ("products_services" in intents) or ("basic_info" in intents))
        
        # Intent-based display logic
        self.logger.debug(f"Intent answer - intents: {intents}, show_location: {show_location}, show_domain: {show_domain}")

        # Show all companies if topk is None, otherwise limit to topk
        companies_to_show = companies if topk is None else companies[:topk]
        for c in companies_to_show:
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

            # Only show contact info if fields have actual values
            contact_bits = []
            if c.contact_info.email and c.contact_info.email.strip():
                contact_bits.append(f"Email: {c.contact_info.email}")
            if c.contact_info.website and c.contact_info.website.strip():
                contact_bits.append(f"Website: {c.contact_info.website}")
            if c.contact_info.phone and c.contact_info.phone.strip():
                contact_bits.append(f"Phone: {c.contact_info.phone}")
            if contact_bits:
                parts.append(" | ".join(contact_bits))

            if "certifications" in intents and c.certifications:
                parts.append("Certifications: " + ", ".join(c.certifications))

            if "rd_capabilities" in intents and c.rd_categories:
                parts.append("R&D: " + ", ".join(c.rd_categories))
            if "testing_capabilities" in intents and c.testing_categories:
                parts.append("Testing: " + ", ".join(c.testing_categories))

            lines.append(" - " + "  ·  ".join(parts))

        header = []
        if show_location: header.append("Location details")
        elif show_domain: header.append("Core expertise")
        else: header.append("Company details")

        return f"{' & '.join(header)}:\n" + "\n".join(lines)

    # ---- Query flow ----
    def process_query(self, user_query: str, topk: int = None) -> QueryResult:
        t0 = time.time()
        timings = {}

        self.engine.ensure_built()
        timings["index_ready_ms"] = (time.time() - t0) * 1000

        # 1) intents
        t1 = time.time()
        intents = self.intent.detect(user_query)
        timings["intent_ms"] = (time.time() - t1) * 1000

        # 2) Company name search strategy (first priority)
        t2 = time.time()
        company_name_hits = self.engine.search_by_company_name(user_query)
        
        # Check if company name search found meaningful results
        # Only use company name results if we found a reasonable number of matches
        # For very generic queries that happen to match company name words, fall back to general search
        use_company_name_results = False
        if company_name_hits:
            # Use company name results if:
            # 1. We found a small number of highly relevant matches (likely exact matches)
            # 2. OR the query looks like it's specifically asking for a company name
            query_words = user_query.lower().split()
            looks_like_company_query = any(word in user_query.lower() for word in ['company', 'ltd', 'limited', 'inc', 'corp'])
            
            if len(company_name_hits) <= 10 or looks_like_company_query:
                use_company_name_results = True
                self.logger.info(f"[Strategy] Company name match found: {len(company_name_hits)} companies")
            else:
                self.logger.info(f"[Strategy] Company name search found {len(company_name_hits)} companies, but falling back to general search for broader query")
        
        if use_company_name_results:
            companies_raw = self.cm.get_integrated_companies()
            companies = [self.builder.from_integrated(companies_raw[i], i) for i in company_name_hits if 0 <= i < len(companies_raw)]
            
            total = time.time() - t0
            md = QueryMetadata(
                strategy=ProcessingStrategy.HYBRID,  # Use hybrid for company name matches
                confidence=1.0,  # High confidence for exact matches
                companies_count=len(companies),
                processing_time=total,
                intent_enhancement_used=False,
                timings_ms={"company_name_search_ms": (time.time() - t2) * 1000},
            )
            summary = self._summarize(companies, intents)
            intent_answer = self._intent_answer(companies, intents)
            return QueryResult(
                companies=companies,
                metadata=md,
                enhanced_summary=summary,
                raw_results={"company_name_hits": len(company_name_hits)},
                intent_info={"intents": intents, "answer": intent_answer}
            )

        # 3) Standard prefilter (fallback if no company name matches)
        tokens = [t for t in tokenize(user_query) if len(t) > 2]
        loc_hits = []
        for tok in tokens:
            loc_hits += self.engine.search_by_location(tok)
        dom_hits = self.engine.search_by_domain_terms({"_": tokens})
        kw_hits = self.engine.search_by_keywords(tokens)
        pre = list(dict.fromkeys(loc_hits + dom_hits + kw_hits))
        timings["prefilter_ms"] = (time.time() - t2) * 1000

        # 3) embeddings (optional) and TF-IDF on the same candidate pool
        # Use all candidates if topk is None, otherwise use expanded topk for ranking
        ranking_limit = len(pre) if topk is None else topk * 2
        t3 = time.time()
        emb_ranked = self.engine.rerank_with_embeddings(user_query, pre, topk=ranking_limit)
        tfidf_ranked = self.engine.tfidf_rank(user_query, pre, topk=ranking_limit)
        fused_ranked = rrf_merge([emb_ranked, tfidf_ranked])
        if topk is not None:
            fused_ranked = fused_ranked[:topk]
        timings["rerank_ms"] = (time.time() - t3) * 1000
        embed_used = _EMBED_OK and len(emb_ranked) > 0
        tfidf_used = _TFIDF_OK and len(tfidf_ranked) > 0

        # 4) confidence & strategy
        confidence = compute_confidence(len(pre), embed_used, tfidf_used)
        has_loc_or_company = any(x in intents for x in ("location", "company"))
        strategy = choose_strategy(intents, confidence, has_loc_or_company)

        # 5) select ids - show all matches if topk is None
        if strategy in (ProcessingStrategy.EMBEDDING_PRIMARY, ProcessingStrategy.HYBRID):
            ids = fused_ranked if topk is None else fused_ranked[:topk]
        elif strategy == ProcessingStrategy.INTENT_ENHANCED:
            ids = (fused_ranked or pre) if topk is None else (fused_ranked or pre)[:topk]
        else:
            ids = (tfidf_ranked or kw_hits or pre) if topk is None else (tfidf_ranked or kw_hits or pre)[:topk]

        # 6) materialize
        t4 = time.time()
        companies_raw = self.cm.get_integrated_companies()
        companies = [self.builder.from_integrated(companies_raw[i], i) for i in ids if 0 <= i < len(companies_raw)]
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


# ============================= Public API / CLI =============================

def _try_project_load_config():
    try:
        from src.load_config import load_config as _lc  # type: ignore
        cfg, logger = _lc()
        return cfg, logger
    except Exception:
        return None, None

def execute_enhanced_query_with_summary(user_query: str, config: Optional[Dict[str, Any]] = None,
                                        logger: Optional[logging.Logger] = None,
                                        topk: int = None) -> Dict[str, Any]:
    if config is None or logger is None:
        cfg2, log2 = _try_project_load_config()
        config = config or (cfg2 or {"company_data": "./processed_data_store"})
        logger = logger or (log2 or _make_default_logger())

    # Fix config structure for project integration
    if config and isinstance(config.get("company_mapped_data"), dict):
        # Project config has nested structure, extract the processed_data_store path
        try:
            from src.load_config import get_company_mapped_data_processed_data_store
            processed_data_store_path = get_company_mapped_data_processed_data_store(config)
            config = dict(config)  # Make a copy
            config["company_data"] = str(processed_data_store_path)
        except Exception:
            # Fallback if the function fails
            config = dict(config)
            config["company_data"] = config["company_mapped_data"].get("processed_data_store", "./processed_data_store/company_mapped_store")

    proc = EnhancedQueryProcessor(config, logger)
    res = proc.process_query(user_query, topk=topk)

    # Detect query intent to determine what fields to include
    intents = res.intent_info.get("intents", ()) if res.intent_info else ()
    
    # Determine what additional fields to include based on query intent
    include_location = "location" in intents or any(word in user_query.lower() for word in ["where", "located", "location", "address"])
    include_domain = "business_domain" in intents or any(word in user_query.lower() for word in ["domain", "expertise", "specialization"])
    include_certifications = "certifications" in intents or any(word in user_query.lower() for word in ["certification", "certified", "iso", "nabl"])
    include_products = "products_services" in intents or any(word in user_query.lower() for word in ["product", "products", "made", "manufacture", "list"])

    # Return companies with context-aware fields
    return {
        "companies": [c.to_dict(
            include_location=include_location,
            include_domain=include_domain, 
            include_certifications=include_certifications,
            include_products=include_products
        ) for c in res.companies]
    }


def _demo_queries():
    return [
        #"companies having ISO certification with testing facilities",
        #"electrical companies in Sweden",
        #"Company_036 location",
        "small scale manufacturing companies",
        #"r&d capabilities in power systems",
    ]

def main():
    import argparse
    logger = _make_default_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--company-data", default="./processed_data_store/company_mapped_store",
                        help="Directory containing integrated_company_search.json")
    parser.add_argument("--intent-mapping", default=None, help="Path to intent_mapping.json")
    parser.add_argument("--domain-mapping", default=None, help="Path to domain_mapping.json (optional)")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    config = {"company_data": args.company_data}
    if args.intent_mapping: config["intent_mapping"] = args.intent_mapping
    if args.domain_mapping: config["domain_mapping"] = args.domain_mapping

    ipath = Path(config["company_data"]) / "integrated_company_search.json"
    logger.info(f"Using integrated file: {ipath} | Exists={ipath.exists()}")

    logger.info("Enhanced Query Execution - Single File (TF-IDF + Embeddings + RRF)")
    for q in _demo_queries():
        # Use the processor directly to get full metadata
        proc = EnhancedQueryProcessor(config, logger)
        result = proc.process_query(q, topk=args.topk)
        
        logger.info(f"Query: {q}")
        logger.info(f"  Strategy={result.metadata.strategy} conf={result.metadata.confidence:.2f} time={result.metadata.processing_time:.3f}s")
        logger.info(f"  Summary: {result.enhanced_summary}")
        if result.intent_info:
            logger.info(f"  Intent answer:\n{result.intent_info.get('answer', 'N/A')}")
        logger.info(f"  Companies: {len(result.companies)}")
        logger.info("-"*60)

if __name__ == "__main__":
    main()

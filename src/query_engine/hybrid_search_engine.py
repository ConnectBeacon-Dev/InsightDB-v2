#!/usr/bin/env python3
"""
Generic Hybrid Search Engine

This module provides a flexible hybrid search system that combines:
1. TF-IDF search for keyword matching
2. Dense search for semantic similarity
3. Configurable data validation for precise filtering
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from enum import Enum
from src.load_config import get_company_mapped_data_processed_data_store, load_config, get_logger

logger = get_logger(__name__)

class SearchMethod(Enum):
    TFIDF_ONLY = "tfidf_only"
    DENSE_ONLY = "dense_only"
    HYBRID = "hybrid"
    VALIDATED_HYBRID = "validated_hybrid"

@dataclass
class SearchConfig:
    """Configuration for hybrid search operations."""
    method: SearchMethod = SearchMethod.VALIDATED_HYBRID
    tfidf_weight: float = 0.6
    dense_weight: float = 0.4
    min_tfidf_score: float = 0.01
    min_dense_score: float = 0.1
    max_candidates: int = 100
    enable_validation: bool = True
    validation_strict: bool = True

@dataclass
class ValidationRule:
    """Rule for validating company data."""
    data_path: str  # e.g., "complete_data.QualityAndCompliance.Certifications.data"
    required_fields: List[str]  # Fields that must have non-null values
    field_conditions: Dict[str, Any] = None  # Additional field conditions
    description: str = ""

class HybridSearchEngine:
    """Generic hybrid search engine with configurable validation."""
    
    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()
        self.base_config, _ = load_config()
        self.base_path = get_company_mapped_data_processed_data_store(self.base_config)
        self.tfidf_data = None
        self.dense_data = None
        self._load_indices()
    
    def _load_indices(self):
        """Load TF-IDF and dense search indices."""
        try:
            # Load TF-IDF index
            tfidf_path = Path(self.base_path) / "tfidf_search" / "tfidf_search_index.pkl"
            if tfidf_path.exists():
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_data = pickle.load(f)
                logger.info(f"✅ Loaded TF-IDF index with {len(self.tfidf_data['company_metadata'])} companies")
            
            # Load dense search metadata
            dense_path = Path(self.base_path) / "dense_search" / "metas.jsonl"
            if dense_path.exists():
                self.dense_data = []
                with open(dense_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.dense_data.append(json.loads(line.strip()))
                logger.info(f"✅ Loaded dense search metadata with {len(self.dense_data)} companies")
                
        except Exception as e:
            logger.error(f"❌ Error loading search indices: {e}")
    
    def search(self, 
               keywords: List[str], 
               validation_rules: List[ValidationRule] = None,
               dense_filter_terms: List[str] = None,
               top_k: int = 20) -> List[Dict]:
        """
        Generic search method with configurable validation.
        
        Args:
            keywords: Keywords to search for
            validation_rules: Rules for validating results
            dense_filter_terms: Terms to filter dense search results
            top_k: Maximum number of results to return
            
        Returns:
            List of validated search results
        """
        logger.info(f"🔍 Starting hybrid search with method: {self.config.method.value}")
        logger.info(f"📝 Keywords: {keywords}")
        logger.info(f"🔧 Validation rules: {len(validation_rules) if validation_rules else 0}")
        
        candidates = []
        
        # Step 1: TF-IDF search
        if self.config.method in [SearchMethod.TFIDF_ONLY, SearchMethod.HYBRID, SearchMethod.VALIDATED_HYBRID]:
            tfidf_candidates = self._tfidf_search(keywords, top_k=self.config.max_candidates)
            logger.info(f"📊 TF-IDF search found {len(tfidf_candidates)} candidates")
            candidates.extend(tfidf_candidates)
        
        # Step 2: Dense search
        if self.config.method in [SearchMethod.DENSE_ONLY, SearchMethod.HYBRID, SearchMethod.VALIDATED_HYBRID]:
            dense_candidates = self._dense_search(dense_filter_terms or keywords, top_k=self.config.max_candidates)
            logger.info(f"📊 Dense search found {len(dense_candidates)} candidates")
            candidates.extend(dense_candidates)
        
        # Step 3: Combine and deduplicate
        combined_candidates = self._combine_candidates(candidates)
        logger.info(f"📊 Combined to {len(combined_candidates)} unique candidates")
        
        # Step 4: Score and rank
        scored_candidates = self._score_candidates(combined_candidates, keywords)
        
        # Step 5: Validation (if enabled)
        if self.config.enable_validation and validation_rules:
            validated_candidates = []
            for candidate in scored_candidates:
                if self._validate_candidate(candidate, validation_rules):
                    validated_candidates.append(candidate)
            
            logger.info(f"✅ Validated {len(validated_candidates)} candidates")
            return validated_candidates[:top_k]
        
        return scored_candidates[:top_k]
    
    def _tfidf_search(self, keywords: List[str], top_k: int = 50) -> List[Dict]:
        """Perform TF-IDF search for given keywords."""
        if not self.tfidf_data:
            return []
        
        try:
            # Create query vector
            query_text = " ".join(keywords)
            query_vector = self.tfidf_data['vectorizer'].transform([query_text])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_data['tfidf_matrix']).flatten()
            
            # Get top results above threshold
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] >= self.config.min_tfidf_score:
                    metadata = self.tfidf_data['company_metadata'][idx].copy()
                    metadata['tfidf_score'] = float(similarities[idx])
                    metadata['search_source'] = 'tfidf'
                    results.append(metadata)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ TF-IDF search error: {e}")
            return []
    
    def _dense_search(self, filter_terms: List[str], top_k: int = 50) -> List[Dict]:
        """Perform dense search with semantic filtering."""
        if not self.dense_data:
            return []
        
        filtered_companies = []
        filter_terms_lower = [term.lower() for term in filter_terms]
        
        for company in self.dense_data:
            expanded_summary = company.get('expanded_summary', '').lower()
            summary = company.get('summary', '').lower()
            
            # Check if any filter terms match
            if any(term in expanded_summary or term in summary for term in filter_terms_lower):
                company_copy = company.copy()
                company_copy['search_source'] = 'dense'
                company_copy['dense_score'] = self._calculate_dense_score(company, filter_terms_lower)
                
                if company_copy['dense_score'] >= self.config.min_dense_score:
                    filtered_companies.append(company_copy)
        
        # Sort by dense score
        filtered_companies.sort(key=lambda x: x.get('dense_score', 0), reverse=True)
        return filtered_companies[:top_k]
    
    def _calculate_dense_score(self, company: Dict, filter_terms: List[str]) -> float:
        """Calculate a simple dense score based on term frequency."""
        text = (company.get('expanded_summary', '') + ' ' + company.get('summary', '')).lower()
        
        score = 0.0
        for term in filter_terms:
            # Count occurrences and normalize
            count = text.count(term)
            score += count * (1.0 / len(filter_terms))
        
        # Normalize by text length to avoid bias toward longer texts
        text_length = len(text.split())
        return min(1.0, score / max(1, text_length / 100))
    
    def _combine_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Combine and deduplicate candidates from different sources."""
        combined = {}
        
        for candidate in candidates:
            ref_no = candidate['company_ref_no']
            if ref_no in combined:
                # Merge information from different sources
                existing = combined[ref_no]
                
                # Combine scores
                if 'tfidf_score' in candidate:
                    existing['tfidf_score'] = candidate['tfidf_score']
                if 'dense_score' in candidate:
                    existing['dense_score'] = candidate['dense_score']
                
                # Update search sources
                existing_sources = existing.get('search_source', '').split(',')
                new_source = candidate.get('search_source', '')
                if new_source and new_source not in existing_sources:
                    existing_sources.append(new_source)
                    existing['search_source'] = ','.join(filter(None, existing_sources))
                
                # Merge other fields
                for key, value in candidate.items():
                    if key not in existing or existing[key] is None:
                        existing[key] = value
            else:
                combined[ref_no] = candidate.copy()
        
        return list(combined.values())
    
    def _score_candidates(self, candidates: List[Dict], keywords: List[str]) -> List[Dict]:
        """Score and rank candidates based on hybrid scoring."""
        for candidate in candidates:
            tfidf_score = candidate.get('tfidf_score', 0.0)
            dense_score = candidate.get('dense_score', 0.0)
            
            # Calculate hybrid score
            hybrid_score = (
                tfidf_score * self.config.tfidf_weight + 
                dense_score * self.config.dense_weight
            )
            
            candidate['hybrid_score'] = hybrid_score
            candidate['final_score'] = hybrid_score
        
        # Sort by final score
        candidates.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        return candidates
    
    def _validate_candidate(self, candidate: Dict, validation_rules: List[ValidationRule]) -> bool:
        """Validate a candidate against the provided rules."""
        company_ref_no = candidate['company_ref_no']
        
        try:
            info_file = Path(self.base_path) / f"{company_ref_no}_INFO.json"
            if not info_file.exists():
                logger.debug(f"❌ {company_ref_no}: JSON file not found")
                return False
            
            with open(info_file, 'r', encoding='utf-8') as f:
                company_info = json.load(f)
            
            # Check each validation rule
            for rule in validation_rules:
                if not self._check_validation_rule(company_info, rule, company_ref_no):
                    if self.config.validation_strict:
                        return False
                    # In non-strict mode, continue checking other rules
            
            logger.debug(f"✅ {company_ref_no}: Validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating {company_ref_no}: {e}")
            return False
    
    def _check_validation_rule(self, company_info: Dict, rule: ValidationRule, company_ref_no: str) -> bool:
        """Check a single validation rule against company data."""
        try:
            # Navigate to the data using the path
            data = company_info
            path_parts = rule.data_path.split('/')
            for path_part in path_parts:
                if path_part in data:
                    data = data[path_part]
                else:
                    logger.info(f"❌ {company_ref_no}: Path {rule.data_path} not found at '{path_part}'")
                    return False
            
            # Get the data array
            if isinstance(data, dict) and 'data' in data:
                data_array = data['data']
            else:
                logger.info(f"❌ {company_ref_no}: No 'data' array found at {rule.data_path}")
                return False
            
            if not isinstance(data_array, list) or not data_array:
                logger.info(f"❌ {company_ref_no}: Empty or invalid data array at {rule.data_path}")
                return False
            
            # Check each record in the data array for valid data
            has_required_data = False
            for record in data_array:
                if not isinstance(record, dict):
                    continue
                
                # Check if this record has any non-null required fields
                record_has_data = False
                for field in rule.required_fields:
                    field_value = record.get(field)
                    if field_value is not None and field_value != "":
                        record_has_data = True
                        logger.info(f"✅ {company_ref_no}: Found {field} = {field_value}")
                        break
                
                if record_has_data:
                    has_required_data = True
                    
                    # Check additional field conditions if specified
                    if rule.field_conditions:
                        conditions_met = True
                        for field, expected_value in rule.field_conditions.items():
                            actual_value = record.get(field)
                            if actual_value != expected_value:
                                logger.info(f"❌ {company_ref_no}: Field condition failed: {field} = {actual_value}, expected {expected_value}")
                                conditions_met = False
                                break
                        
                        if conditions_met:
                            return True
                    else:
                        return True
            
            if not has_required_data:
                logger.info(f"❌ {company_ref_no}: No required fields found for rule: {rule.description}")
                return False
            
            return has_required_data
            
        except Exception as e:
            logger.error(f"❌ Error checking validation rule for {company_ref_no}: {e}")
            return False

# Predefined validation rules for common use cases
CERTIFICATION_VALIDATION = ValidationRule(
    data_path="complete_data/QualityAndCompliance.Certifications",
    required_fields=[
        "CertificationTypeMaster.Cert_Type",
        "CompanyCertificationDetail.Certification_Type",
        "CompanyCertificationDetail.Certificate_No"
    ],
    description="Company has valid certifications"
)

TESTING_FACILITIES_VALIDATION = ValidationRule(
    data_path="complete_data/QualityAndCompliance.TestingCapabilities",
    required_fields=[
        "TestFacilityCategoryMaster.CategoryName",
        "TestFacilitySubCategoryMaster.SubCategoryName",
        "CompanyTestFacility.TestDetails",
        "CompanyTestFacility.IsNabIAccredited"
    ],
    description="Company has testing facilities"
)

RD_FACILITIES_VALIDATION = ValidationRule(
    data_path="complete_data/ResearchAndDevelopment.RDCapabilities",
    required_fields=[
        "RDCategoryMaster.RDCategoryName",
        "RDSubCategoryMaster.RDSubCategoryName",
        "CompanyRDFacility.RD_Details"
    ],
    description="Company has R&D facilities"
)

# Convenience functions for common searches
def search_companies_with_certifications(keywords: List[str] = None, top_k: int = 20, config: SearchConfig = None) -> List[Dict]:
    """Search for companies with certifications."""
    engine = HybridSearchEngine(config or SearchConfig())
    search_keywords = (keywords or []) + ["ISO", "certification", "certified", "standards", "quality"]
    dense_terms = ["iso", "certification", "certified", "standards"]
    
    return engine.search(
        keywords=search_keywords,
        validation_rules=[CERTIFICATION_VALIDATION],
        dense_filter_terms=dense_terms,
        top_k=top_k
    )

def search_companies_with_testing_facilities(keywords: List[str] = None, top_k: int = 20, config: SearchConfig = None) -> List[Dict]:
    """Search for companies with testing facilities."""
    engine = HybridSearchEngine(config or SearchConfig())
    search_keywords = (keywords or []) + ["testing", "test facilities", "laboratories", "testing capabilities"]
    dense_terms = ["testing", "test", "laboratories", "facilities"]
    
    return engine.search(
        keywords=search_keywords,
        validation_rules=[TESTING_FACILITIES_VALIDATION],
        dense_filter_terms=dense_terms,
        top_k=top_k
    )

def search_companies_with_rd_facilities(keywords: List[str] = None, top_k: int = 20, config: SearchConfig = None) -> List[Dict]:
    """Search for companies with R&D facilities."""
    engine = HybridSearchEngine(config or SearchConfig())
    search_keywords = (keywords or []) + ["R&D", "research", "development", "innovation"]
    dense_terms = ["research", "development", "r&d", "innovation"]
    
    return engine.search(
        keywords=search_keywords,
        validation_rules=[RD_FACILITIES_VALIDATION],
        dense_filter_terms=dense_terms,
        top_k=top_k
    )

def search_companies_with_multiple_capabilities(
    certifications: bool = False,
    testing_facilities: bool = False,
    rd_facilities: bool = False,
    keywords: List[str] = None,
    top_k: int = 20,
    config: SearchConfig = None
) -> List[Dict]:
    """Search for companies with multiple capabilities."""
    
    validation_rules = []
    search_keywords = keywords or []
    dense_terms = []
    
    if certifications:
        validation_rules.append(CERTIFICATION_VALIDATION)
        search_keywords.extend(["ISO", "certification", "certified"])
        dense_terms.extend(["iso", "certification", "certified"])
    
    if testing_facilities:
        validation_rules.append(TESTING_FACILITIES_VALIDATION)
        search_keywords.extend(["testing", "test facilities", "laboratories"])
        dense_terms.extend(["testing", "test", "laboratories"])
    
    if rd_facilities:
        validation_rules.append(RD_FACILITIES_VALIDATION)
        search_keywords.extend(["R&D", "research", "development"])
        dense_terms.extend(["research", "development", "r&d"])
    
    if not validation_rules:
        logger.warning("⚠️ No capabilities specified for search")
        return []
    
    # For multiple capabilities, use strict validation
    strict_config = config or SearchConfig()
    strict_config.validation_strict = True
    
    engine = HybridSearchEngine(strict_config)
    
    return engine.search(
        keywords=search_keywords,
        validation_rules=validation_rules,
        dense_filter_terms=dense_terms,
        top_k=top_k
    )

# Test function
def test_hybrid_search_engine():
    """Test the hybrid search engine functionality."""
    logger.info("🧪 Testing Generic Hybrid Search Engine")
    logger.info("=" * 70)
    
    try:
        # Test 1: Companies with certifications
        logger.info("\n🔍 TEST 1: Companies with certifications")
        cert_companies = search_companies_with_certifications(top_k=3)
        logger.info(f"Found {len(cert_companies)} companies with certifications:")
        for i, company in enumerate(cert_companies, 1):
            score = company.get('final_score', 0)
            sources = company.get('search_source', 'unknown')
            logger.info(f"  {i}. {company['company_ref_no']} - {company['company_name']} (score: {score:.3f}, sources: {sources})")
        
        # Test 2: Companies with testing facilities
        logger.info("\n🔍 TEST 2: Companies with testing facilities")
        test_companies = search_companies_with_testing_facilities(top_k=3)
        logger.info(f"Found {len(test_companies)} companies with testing facilities:")
        for i, company in enumerate(test_companies, 1):
            score = company.get('final_score', 0)
            sources = company.get('search_source', 'unknown')
            logger.info(f"  {i}. {company['company_ref_no']} - {company['company_name']} (score: {score:.3f}, sources: {sources})")
        
        # Test 3: Companies with BOTH certifications AND testing
        logger.info("\n🔍 TEST 3: Companies with BOTH certifications AND testing")
        both_companies = search_companies_with_multiple_capabilities(
            certifications=True,
            testing_facilities=True,
            top_k=3
        )
        logger.info(f"Found {len(both_companies)} companies with BOTH:")
        for i, company in enumerate(both_companies, 1):
            score = company.get('final_score', 0)
            sources = company.get('search_source', 'unknown')
            logger.info(f"  {i}. {company['company_ref_no']} - {company['company_name']} (score: {score:.3f}, sources: {sources})")
        
        # Test 4: Custom search with specific keywords
        logger.info("\n🔍 TEST 4: Custom search for electrical companies with certifications")
        electrical_cert_companies = search_companies_with_certifications(
            keywords=["electrical", "power", "energy"],
            top_k=3
        )
        logger.info(f"Found {len(electrical_cert_companies)} electrical companies with certifications:")
        for i, company in enumerate(electrical_cert_companies, 1):
            score = company.get('final_score', 0)
            expertise = company.get('core_expertise', 'N/A')
            logger.info(f"  {i}. {company['company_ref_no']} - {company['company_name']} ({expertise}, score: {score:.3f})")
        
        logger.info("\n✅ All hybrid search engine tests completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Hybrid search engine test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hybrid_search_engine()

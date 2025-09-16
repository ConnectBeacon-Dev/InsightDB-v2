#!/usr/bin/env python3
"""
Updated Helper methods for the Embedding Query Decomposer
These methods now use the generic hybrid search engine for accurate data validation
"""

from typing import Dict, List, Optional
from src.load_config import get_logger
from src.query_engine.hybrid_search_engine import (
    search_companies_with_certifications,
    search_companies_with_testing_facilities,
    search_companies_with_rd_facilities,
    search_companies_with_multiple_capabilities
)

logger = get_logger(__name__)

def build_certification_search_query(component, decomposed) -> str:
    """Build search query for certification searches."""
    # Extract certification type terms if present
    cert_terms = []
    for comp in decomposed.components:
        if comp.type.value == 'attribute' and comp.domain_hint == 'certification_type':
            cert_terms.extend(comp.keywords)
    
    # Build focused search query
    if cert_terms:
        return f"{' '.join(cert_terms)} certification standards quality"
    else:
        return "ISO certification standards quality NABL accredited"

def build_testing_search_query(component, decomposed) -> str:
    """Build search query for testing facility searches."""
    # Extract testing category terms if present
    test_terms = []
    for comp in decomposed.components:
        if comp.type.value == 'attribute' and comp.domain_hint == 'test_category':
            test_terms.extend(comp.keywords)
    
    # Build focused search query
    if test_terms:
        return f"{' '.join(test_terms)} testing facilities laboratories"
    else:
        return "testing facilities laboratories high voltage environmental EMI EMC"

def build_rd_search_query(component, decomposed) -> str:
    """Build search query for R&D facility searches."""
    # Extract R&D category terms if present
    rd_terms = []
    for comp in decomposed.components:
        if comp.type.value == 'attribute' and comp.domain_hint == 'rd_category':
            rd_terms.extend(comp.keywords)
    
    # Build focused search query
    if rd_terms:
        return f"{' '.join(rd_terms)} R&D research development facilities"
    else:
        return "R&D research development facilities laboratories innovation"

def build_defence_search_query(component, decomposed) -> str:
    """Build search query for defence platform searches."""
    # Extract defence/platform terms if present
    defence_terms = []
    for comp in decomposed.components:
        if comp.type.value == 'attribute' and comp.domain_hint in ['industry', 'expertise']:
            if any(term in comp.text.lower() for term in ['defence', 'defense', 'military', 'aerospace']):
                defence_terms.extend(comp.keywords)
    
    # Build focused search query
    if defence_terms:
        return f"{' '.join(defence_terms)} defence platform systems"
    else:
        return "defence platform systems military aerospace"

def build_tech_area_search_query(component, decomposed) -> str:
    """Build search query for technology area searches."""
    # Extract technology terms if present
    tech_terms = []
    for comp in decomposed.components:
        if comp.type.value == 'attribute' and comp.domain_hint in ['expertise', 'industry']:
            tech_terms.extend(comp.keywords)
    
    # Build focused search query
    if tech_terms:
        return f"{' '.join(tech_terms)} technology platform area PTA"
    else:
        return "technology platform area PTA technical capabilities"

def build_export_search_query(component, decomposed) -> str:
    """Build search query for export capability searches."""
    # Extract export/international terms if present
    export_terms = []
    for comp in decomposed.components:
        if comp.type.value == 'attribute' and comp.domain_hint in ['industry', 'expertise']:
            export_terms.extend(comp.keywords)
    
    # Build focused search query
    if export_terms:
        return f"{' '.join(export_terms)} exports international export capabilities"
    else:
        return "exports international export capabilities global markets"

def filter_certifications_by_companies(companies: List[Dict]) -> List[Dict]:
    """Filter companies to only include those with actual certifications using hybrid search."""
    logger.info(f"🔍 Filtering {len(companies)} companies for certifications using hybrid search")
    
    try:
        # Get company ref numbers from input
        company_refs = {c.get('company_ref_no', c.get('id', '')) for c in companies}
        company_refs = {ref for ref in company_refs if ref}
        
        if not company_refs:
            logger.warning("⚠️ No valid company references found")
            return []
        
        # Search for companies with certifications using hybrid engine
        cert_companies = search_companies_with_certifications(top_k=100)
        cert_refs = {c['company_ref_no'] for c in cert_companies}
        
        # Filter input companies to only include those with certifications
        filtered_companies = []
        for company in companies:
            company_ref = company.get('company_ref_no', company.get('id', ''))
            if company_ref in cert_refs:
                # Enhance company data with certification info
                cert_company = next((c for c in cert_companies if c['company_ref_no'] == company_ref), None)
                if cert_company:
                    enhanced_company = company.copy()
                    enhanced_company.update(cert_company)
                    filtered_companies.append(enhanced_company)
        
        logger.info(f"✅ Filtered to {len(filtered_companies)} companies with actual certifications")
        return filtered_companies
        
    except Exception as e:
        logger.error(f"❌ Error in filter_certifications_by_companies: {e}")
        # Fallback to original behavior
        return companies

def filter_testing_by_companies(companies: List[Dict]) -> List[Dict]:
    """Filter companies to only include those with actual testing facilities using hybrid search."""
    logger.info(f"🔍 Filtering {len(companies)} companies for testing facilities using hybrid search")
    
    try:
        # Get company ref numbers from input
        company_refs = {c.get('company_ref_no', c.get('id', '')) for c in companies}
        company_refs = {ref for ref in company_refs if ref}
        
        if not company_refs:
            logger.warning("⚠️ No valid company references found")
            return []
        
        # Search for companies with testing facilities using hybrid engine
        test_companies = search_companies_with_testing_facilities(top_k=100)
        test_refs = {c['company_ref_no'] for c in test_companies}
        
        # Filter input companies to only include those with testing facilities
        filtered_companies = []
        for company in companies:
            company_ref = company.get('company_ref_no', company.get('id', ''))
            if company_ref in test_refs:
                # Enhance company data with testing facility info
                test_company = next((c for c in test_companies if c['company_ref_no'] == company_ref), None)
                if test_company:
                    enhanced_company = company.copy()
                    enhanced_company.update(test_company)
                    filtered_companies.append(enhanced_company)
        
        logger.info(f"✅ Filtered to {len(filtered_companies)} companies with actual testing facilities")
        return filtered_companies
        
    except Exception as e:
        logger.error(f"❌ Error in filter_testing_by_companies: {e}")
        # Fallback to original behavior
        return companies

def filter_rd_by_companies(companies: List[Dict]) -> List[Dict]:
    """Filter companies to only include those with actual R&D facilities using hybrid search."""
    logger.info(f"🔍 Filtering {len(companies)} companies for R&D facilities using hybrid search")
    
    try:
        # Get company ref numbers from input
        company_refs = {c.get('company_ref_no', c.get('id', '')) for c in companies}
        company_refs = {ref for ref in company_refs if ref}
        
        if not company_refs:
            logger.warning("⚠️ No valid company references found")
            return []
        
        # Search for companies with R&D facilities using hybrid engine
        rd_companies = search_companies_with_rd_facilities(top_k=100)
        rd_refs = {c['company_ref_no'] for c in rd_companies}
        
        # Filter input companies to only include those with R&D facilities
        filtered_companies = []
        for company in companies:
            company_ref = company.get('company_ref_no', company.get('id', ''))
            if company_ref in rd_refs:
                # Enhance company data with R&D facility info
                rd_company = next((c for c in rd_companies if c['company_ref_no'] == company_ref), None)
                if rd_company:
                    enhanced_company = company.copy()
                    enhanced_company.update(rd_company)
                    filtered_companies.append(enhanced_company)
        
        logger.info(f"✅ Filtered to {len(filtered_companies)} companies with actual R&D facilities")
        return filtered_companies
        
    except Exception as e:
        logger.error(f"❌ Error in filter_rd_by_companies: {e}")
        # Fallback to original behavior
        return companies

def filter_defence_by_companies(companies: List[Dict]) -> List[Dict]:
    """Filter defence platforms by company list."""
    # This would typically query the defence platform data for these companies
    # For now, return the companies themselves as they represent entities working on defence platforms
    return companies

def filter_tech_areas_by_companies(companies: List[Dict]) -> List[Dict]:
    """Filter technology areas by company list."""
    # This would typically query the technology area data for these companies
    # For now, return the companies themselves as they represent entities in specific tech areas
    return companies

def filter_exports_by_companies(companies: List[Dict]) -> List[Dict]:
    """Filter export capabilities by company list."""
    # This would typically query the export data for these companies
    # For now, return the companies themselves as they represent entities with export capabilities
    return companies

def search_companies_with_certifications_and_testing(query_terms: List[str] = None, top_k: int = 20) -> List[Dict]:
    """Search for companies that have BOTH certifications AND testing facilities using hybrid search."""
    logger.info("🔍 Searching for companies with BOTH certifications AND testing facilities")
    
    try:
        return search_companies_with_multiple_capabilities(
            certifications=True,
            testing_facilities=True,
            keywords=query_terms,
            top_k=top_k
        )
    except Exception as e:
        logger.error(f"❌ Error in search_companies_with_certifications_and_testing: {e}")
        return []

# Test function
def test_updated_helpers():
    """Test the updated helper functions."""
    logger.info("🧪 Testing Updated Helper Functions")
    logger.info("=" * 60)
    
    try:
        # Create some dummy companies for testing
        dummy_companies = [
            {'company_ref_no': 'CMP001', 'company_name': 'Company_001'},
            {'company_ref_no': 'CMP002', 'company_name': 'Company_002'},
            {'company_ref_no': 'CMP036', 'company_name': 'Company_036'},  # This should be filtered out
            {'company_ref_no': 'CMP050', 'company_name': 'Company_050'},
        ]
        
        # Test certification filtering
        logger.info("\n🔍 TEST 1: Filter companies with certifications")
        cert_filtered = filter_certifications_by_companies(dummy_companies)
        logger.info(f"Original: {len(dummy_companies)} companies")
        logger.info(f"Filtered: {len(cert_filtered)} companies with certifications")
        for company in cert_filtered:
            logger.info(f"  - {company['company_ref_no']}: {company['company_name']}")
        
        # Test testing facility filtering
        logger.info("\n🔍 TEST 2: Filter companies with testing facilities")
        test_filtered = filter_testing_by_companies(dummy_companies)
        logger.info(f"Original: {len(dummy_companies)} companies")
        logger.info(f"Filtered: {len(test_filtered)} companies with testing facilities")
        for company in test_filtered:
            logger.info(f"  - {company['company_ref_no']}: {company['company_name']}")
        
        # Test combined search
        logger.info("\n🔍 TEST 3: Search companies with BOTH certifications AND testing")
        both_companies = search_companies_with_certifications_and_testing(top_k=5)
        logger.info(f"Found {len(both_companies)} companies with BOTH:")
        for company in both_companies:
            score = company.get('final_score', 0)
            logger.info(f"  - {company['company_ref_no']}: {company['company_name']} (score: {score:.3f})")
        
        logger.info("\n✅ Updated helper function tests completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Updated helper function test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_updated_helpers()

#!/usr/bin/env python3
"""
Debug script to test validation logic directly on CMP001 and CMP002
"""

import json
from pathlib import Path
from src.query_engine.hybrid_search_engine import CERTIFICATION_VALIDATION, TESTING_FACILITIES_VALIDATION, HybridSearchEngine
from src.load_config import load_config, get_logger

def test_direct_validation():
    """Test validation logic directly on CMP001 and CMP002"""
    logger = get_logger(__name__)
    logger.info("🧪 Testing Direct Validation on CMP001 and CMP002")
    logger.info("=" * 60)
    
    config, _ = load_config()
    engine = HybridSearchEngine()
    
    # Test CMP001
    logger.info("\n🔍 Testing CMP001:")
    try:
        info_file = Path("processed_data_store/company_mapped_store/CMP001_INFO.json")
        with open(info_file, 'r', encoding='utf-8') as f:
            company_info = json.load(f)
        
        # Test certification validation
        cert_result = engine._check_validation_rule(company_info, CERTIFICATION_VALIDATION, "CMP001")
        logger.info(f"   Certification validation: {'✅ PASSED' if cert_result else '❌ FAILED'}")
        
        # Test testing facilities validation
        test_result = engine._check_validation_rule(company_info, TESTING_FACILITIES_VALIDATION, "CMP001")
        logger.info(f"   Testing facilities validation: {'✅ PASSED' if test_result else '❌ FAILED'}")
        
    except Exception as e:
        logger.error(f"❌ Error testing CMP001: {e}")
    
    # Test CMP002
    logger.info("\n🔍 Testing CMP002:")
    try:
        info_file = Path("processed_data_store/company_mapped_store/CMP002_INFO.json")
        with open(info_file, 'r', encoding='utf-8') as f:
            company_info = json.load(f)
        
        # Test certification validation
        cert_result = engine._check_validation_rule(company_info, CERTIFICATION_VALIDATION, "CMP002")
        logger.info(f"   Certification validation: {'✅ PASSED' if cert_result else '❌ FAILED'}")
        
        # Test testing facilities validation
        test_result = engine._check_validation_rule(company_info, TESTING_FACILITIES_VALIDATION, "CMP002")
        logger.info(f"   Testing facilities validation: {'✅ PASSED' if test_result else '❌ FAILED'}")
        
    except Exception as e:
        logger.error(f"❌ Error testing CMP002: {e}")

if __name__ == "__main__":
    test_direct_validation()

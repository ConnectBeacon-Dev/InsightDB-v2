#!/usr/bin/env python3
"""
Integration Test for Query Planning and Company Search API

This script tests the integration between build_query_domain.py and company_search_api.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.company_index.company_search_api import CompanySearchAPI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_integration():
    """Test the integration between query planning and search API."""
    logger.info("🧪 Starting Integration Test")
    logger.info("=" * 60)
    
    try:
        # Initialize API with query planning enabled
        logger.info("Initializing CompanySearchAPI with query planning...")
        api = CompanySearchAPI(enable_query_planning=True)
        
        # Check status
        status = api.status()
        logger.info(f"Indices ready: {status['indices_ready']}")
        logger.info(f"Query planning enabled: {api.query_planning_enabled}")
        
        if not status['indices_ready']:
            logger.warning("Search indices not ready. Some tests may be limited.")
        
        # Test queries
        test_queries = [
            #"List companies with ISO 9001 certificate",
            #"Show me companies with expertise in production and operations", 
            #"List all products supplied by medium scale companies",
            #"Companies in aerospace industry with R&D capabilities",
            "give me list of all products which are supplied by medium scale companies",
        ]
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Test {i}: {query}")
            logger.info(f"{'='*60}")
            
            try:
                # Test intelligent search
                result = api.intelligent_search(query, top_k=5)
                
                logger.info(f"✓ Query executed successfully")
                logger.info(f"Planning used: {result['metadata']['planning_used']}")
                logger.info(f"Plan type: {result['metadata'].get('plan_type', 'N/A')}")
                logger.info(f"Total steps: {result['metadata']['total_steps']}")
                logger.info(f"Final results: {len(result['results'])}")
                logger.info(f"Confidence: {result['metadata']['confidence']}")
                
                # Show execution steps
                if result['execution_steps']:
                    logger.info("Execution steps:")
                    for step in result['execution_steps']:
                        step_info = f"  - {step.get('step_id', step.get('step', 'unknown'))}"
                        if 'results_count' in step:
                            step_info += f": {step['results_count']} results"
                        if 'dependencies' in step and step['dependencies']:
                            step_info += f" (depends on: {step['dependencies']})"
                        logger.info(step_info)
                
                # Show sample results
                if result['results']:
                    logger.info("Sample results:")
                    for j, res in enumerate(result['results'][:2], 1):
                        company_name = res.get('company_name', 'Unknown')
                        expertise = res.get('core_expertise', 'N/A')
                        logger.info(f"  {j}. {company_name} - {expertise}")
                
            except Exception as e:
                logger.error(f"✗ Test failed: {e}")
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 Integration test completed!")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise

def test_query_planning_only():
    """Test query planning without search execution."""
    logger.info("🧪 Testing Query Planning Only")
    logger.info("=" * 40)
    
    try:
        from src.build_query_domain import IndustrialQueryPlanner
        
        planner = IndustrialQueryPlanner()
        
        test_queries = [
            "List all products supplied by medium scale companies",
            "Show companies with ISO certification and electrical expertise"
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting: {query}")
            plan = planner.plan_query(query, use_smart_fallback=True)
            
            logger.info(f"Plan type: {plan.plan_type.value}")
            logger.info(f"Confidence: {plan.confidence}")
            logger.info(f"Sub-queries: {len(plan.sub_queries)}")
            
            for sq in plan.sub_queries:
                logger.info(f"  - {sq.id}: {sq.query}")
                if sq.dependencies:
                    logger.info(f"    Dependencies: {sq.dependencies}")
        
        logger.info("✓ Query planning test completed")
        
    except Exception as e:
        logger.error(f"Query planning test failed: {e}")

if __name__ == "__main__":
    # Test query planning first
    test_query_planning_only()
    
    # Then test full integration
    test_integration()

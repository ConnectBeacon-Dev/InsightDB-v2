#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced logging functionality
in the embedding query decomposer.
"""

from src.query_engine.embedding_query_decomposer import execute_enhanced_embedding_query
from src.load_config import load_config

def test_enhanced_logging():
    """Test the enhanced logging functionality with various queries."""
    
    # Load configuration and logger
    config, logger = load_config()
    
    # Test queries with different complexity levels
    test_queries = [
        # Simple query (high confidence expected)
        "list companies",
        
        # Medium complexity with location and expertise filters
        "small scale electrical companies from Karnataka",
        
        # Complex query with multiple entities and relationships
        "companies having ISO certification with testing facilities",
        
        # Query that should trigger LLM validation (low confidence)
        "advanced manufacturing capabilities"
    ]
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n{'='*100}")
        logger.info(f"TEST {i}/{len(test_queries)}: TESTING QUERY")
        logger.info(f"{'='*100}")
        
        try:
            # Execute with enhanced logging
            result = execute_enhanced_embedding_query(
                user_query=query,
                config=config,
                logger=logger,
                enable_llm_validation=True
            )
            
            logger.info(f"\n🎯 TEST {i} COMPLETED SUCCESSFULLY")
            logger.info(f"📊 Results Summary:")
            logger.info(f"   - Strategy: {result.get('strategy', 'Unknown')}")
            logger.info(f"   - Components: {result.get('components', 0)}")
            logger.info(f"   - Dependencies: {result.get('dependencies', 0)}")
            logger.info(f"   - Confidence: {result.get('confidence', 0):.2f}")
            logger.info(f"   - LLM Used: {result.get('llm_validation_used', False)}")
            
            if result.get('results'):
                results = result['results']
                logger.info(f"   - Companies Found: {results.get('companies_count', 0)}")
                logger.info(f"   - Products Found: {results.get('products_count', 0)}")
            
        except Exception as e:
            logger.error(f"❌ TEST {i} FAILED: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    logger.info(f"\n{'='*100}")
    logger.info("🏁 ALL TESTS COMPLETED")
    logger.info(f"{'='*100}")

if __name__ == "__main__":
    test_enhanced_logging()

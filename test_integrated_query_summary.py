#!/usr/bin/env python3
"""
Test script for the integrated enhanced query execution with summarization.
"""

from src.query_engine.enhanced_query_with_summary import execute_enhanced_query_with_summary
from src.load_config import load_config

def main():
    """Test the integrated enhanced query execution with summarization."""
    
    # Load configuration and logger
    (config, logger) = load_config()
    
    # Test query
    test_query = "small scale electrical companies"
    
    logger.info("🧪 TESTING INTEGRATED QUERY EXECUTION WITH SUMMARIZATION")
    logger.info("=" * 80)
    logger.info(f"Test Query: '{test_query}'")
    logger.info("=" * 80)
    
    try:
        # Execute the integrated query with summarization
        result = execute_enhanced_query_with_summary(
            user_query=test_query,
            config=config,
            logger=logger,
            enable_llm_validation=True
        )
        
        # Display final results
        logger.info("\n" + "🎯 FINAL INTEGRATED RESULTS:")
        logger.info("=" * 60)
        
        if 'error' in result:
            logger.error(f"❌ Query failed: {result['error']}")
        else:
            logger.info(f"✅ Query executed successfully")
            logger.info(f"📊 Confidence: {result.get('confidence', 0):.2f}")
            logger.info(f"🤖 LLM Validation: {result.get('llm_validation_used', False)}")
            logger.info(f"🔍 Strategy: {result.get('strategy', 'Unknown')}")
            
            if result.get('results'):
                results = result['results']
                logger.info(f"📈 Companies: {results.get('companies_count', 0)}")
                logger.info(f"📈 Products: {results.get('products_count', 0)}")
            
            # Show the enhanced summary
            if result.get('enhanced_summary'):
                logger.info("\n" + "📋 FINAL ENHANCED SUMMARY:")
                logger.info("=" * 50)
                print(result['enhanced_summary'])  # Print to console for visibility
                logger.info("=" * 50)
            else:
                logger.warning("⚠️ No enhanced summary generated")
        
        logger.info("\n✅ Integration test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()

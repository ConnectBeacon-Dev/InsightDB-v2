#!/usr/bin/env python3
"""
Test script for embedding query decomposer integration with company search API
"""

import sys
import os
from  src.load_config import  load_config
sys.path.append('src')

from src.query_engine.embedding_query_decomposer import execute_embedding_query

def test_integration():
    """Test the integration between embedding decomposer and company search API"""

    (config, logger) = load_config()

    print("🧪 Testing Embedding Query Decomposer Integration")
    print("=" * 60)
    
    test_queries = [
        "list small scale companies from Karnataka",
        "show companies with electrical expertise", 
        "find products from medium scale companies",
        "companies having ISO certification"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        print("-" * 40)
        
        try:
            result = execute_embedding_query(query)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            print(f"✅ Query executed successfully!")
            print(f"Strategy: {result.get('strategy', 'Unknown')}")
            print(f"Components: {result.get('components', 0)}")
            print(f"Dependencies: {result.get('dependencies', 0)}")
            
            if result.get('results'):
                results = result['results']
                print(f"Companies found: {results.get('companies_count', 0)}")
                print(f"Products found: {results.get('products_count', 0)}")
                
                # Show sample results
                if results.get('companies'):
                    print("\nSample companies:")
                    for j, company in enumerate(results['companies'][:3], 1):
                        name = company.get('company_name', 'Unknown')
                        expertise = company.get('core_expertise', 'N/A')
                        print(f"  {j}. {name} - {expertise}")
                
                if results.get('products'):
                    print("\nSample products:")
                    for j, product in enumerate(results['products'][:3], 1):
                        name = product.get('product_name', 'Unknown')
                        company = product.get('company_name', 'N/A')
                        print(f"  {j}. {name} (by {company})")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_integration()

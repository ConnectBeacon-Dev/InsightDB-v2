#!/usr/bin/env python3
"""
Test script to demonstrate the LLM integration approach in embedding query decomposer
"""

import sys
import os
sys.path.append('src')

from src.embedding_query_decomposer import execute_embedding_query, execute_enhanced_embedding_query

def test_both_approaches():
    """Test both embedding-only and enhanced embedding+LLM approaches"""
    print("🧪 Testing Both Approaches: Embedding vs Enhanced Embedding+LLM")
    print("=" * 80)
    
    test_queries = [
        # High confidence query (should not trigger LLM)
        "list small scale companies from Karnataka",
        
        # Medium confidence query (may trigger LLM)
        "show electrical expertise companies", 
        
        # Low confidence/complex query (should trigger LLM)
        "advanced defense manufacturing capabilities with R&D facilities",
        
        # Ambiguous query (should trigger LLM)
        "innovative solutions for aerospace industry"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"🔍 Test {i}: '{query}'")
        print(f"{'='*60}")
        
        # Test 1: Embedding-only approach (fast, no LLM)
        print(f"\n📊 APPROACH 1: Embedding-Only (No LLM)")
        print("-" * 40)
        
        try:
            result1 = execute_embedding_query(query)
            
            if 'error' in result1:
                print(f"❌ query={query}Error: {result1['error']}")
            else:
                print(f"✅ Strategy: {result1.get('strategy', 'Unknown')}")
                print(f"✅ Components: {result1.get('components', 0)}")
                print(f"✅ Dependencies: {result1.get('dependencies', 0)}")
                
                if result1.get('results'):
                    results = result1['results']
                    print(f"✅ Companies found: {results.get('companies_count', 0)}")
                    print(f"✅ Products found: {results.get('products_count', 0)}")
                
        except Exception as e:
            print(f"❌ Embedding-only approach failed: {e}")
        
        # Test 2: Enhanced approach with LLM validation
        print(f"\n🤖 APPROACH 2: Enhanced Embedding + LLM Validation")
        print("-" * 40)
        
        try:
            result2 = execute_enhanced_embedding_query(query, enable_llm_validation=True)
            
            if 'error' in result2:
                print(f"❌ Error: {result2['error']}")
            else:
                print(f"✅ Strategy: {result2.get('strategy', 'Unknown')}")
                print(f"✅ Components: {result2.get('components', 0)}")
                print(f"✅ Dependencies: {result2.get('dependencies', 0)}")
                print(f"📊 Confidence: {result2.get('confidence', 0):.2f}")
                print(f"🤖 LLM Validation Used: {result2.get('llm_validation_used', False)}")
                print(f"🎯 Threshold: {result2.get('llm_confidence_threshold', 0.7)}")
                
                if result2.get('results'):
                    results = result2['results']
                    print(f"✅ Companies found: {results.get('companies_count', 0)}")
                    print(f"✅ Products found: {results.get('products_count', 0)}")
                
        except Exception as e:
            print(f"❌ Enhanced approach failed: {e}")
        
        # Comparison
        print(f"\n📈 COMPARISON:")
        print("-" * 20)
        try:
            if 'error' not in result1 and 'error' not in result2:
                approach1_results = result1.get('results', {}).get('companies_count', 0) + result1.get('results', {}).get('products_count', 0)
                approach2_results = result2.get('results', {}).get('companies_count', 0) + result2.get('results', {}).get('products_count', 0)
                
                print(f"Approach 1 (Embedding-only): {approach1_results} total results")
                print(f"Approach 2 (Enhanced+LLM): {approach2_results} total results")
                
                if result2.get('llm_validation_used', False):
                    print("🎯 LLM validation was triggered for this query")
                else:
                    print("⚡ High confidence - LLM validation skipped")
        except:
            print("Could not compare results")

def test_confidence_calculation():
    """Test the confidence calculation mechanism"""
    print(f"\n{'='*80}")
    print("🧪 Testing Confidence Calculation Mechanism")
    print("=" * 80)
    
    confidence_test_queries = [
        ("list companies", "Simple, clear query - should have high confidence"),
        ("small scale companies from Karnataka", "Well-structured query - should have high confidence"),
        ("electrical expertise", "Missing entity - should have medium confidence"),
        ("advanced innovative solutions", "Vague query - should have low confidence"),
        ("xyz abc def", "Nonsense query - should have very low confidence")
    ]
    
    for query, description in confidence_test_queries:
        print(f"\n🔍 Query: '{query}'")
        print(f"📝 Expected: {description}")
        
        try:
            result = execute_enhanced_embedding_query(query, enable_llm_validation=False)  # Just test confidence
            confidence = result.get('confidence', 0)
            components = result.get('components', 0)
            dependencies = result.get('dependencies', 0)
            
            print(f"📊 Confidence: {confidence:.2f}")
            print(f"🔧 Components: {components}")
            print(f"🔗 Dependencies: {dependencies}")
            
            if confidence >= 0.7:
                print("✅ HIGH confidence - LLM validation would be skipped")
            elif confidence >= 0.4:
                print("⚠️ MEDIUM confidence - LLM validation might be triggered")
            else:
                print(f"❌ query={query} LOW confidence - LLM validation would definitely be triggered")
                
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting LLM Integration Test Suite")
    print("=" * 80)
    
    # Test both approaches
    test_both_approaches()
    
    # Test confidence calculation
    test_confidence_calculation()
    
    print(f"\n{'='*80}")
    print("✅ LLM Integration Test Suite Completed")
    print("=" * 80)
    
    print(f"\n📋 SUMMARY:")
    print("- Approach 1: Fast embedding-based decomposition (no LLM)")
    print("- Approach 2: Enhanced with LLM validation for low-confidence queries")
    print("- LLM is only used when confidence < threshold (default 0.7)")
    print("- This provides the best of both worlds: speed + accuracy")

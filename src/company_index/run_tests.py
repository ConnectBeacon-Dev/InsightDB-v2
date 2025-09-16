#!/usr/bin/env python3
"""
Test Runner for Company Search API

Simple script to run the comprehensive test suite for the Company Search API.

Usage:
    python src/company_index/run_tests.py [options]
    
Options:
    --quick     Run only essential tests (faster)
    --verbose   Enable verbose logging
    --help      Show this help message
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def run_quick_tests():
    """Run a subset of essential tests for quick validation."""
    import unittest
    from src.company_index.test_company_search_api import TestCompanySearchAPI
    
    # Select essential tests
    essential_tests = [
        'test_01_initial_status',
        'test_02_cleanup_api', 
        'test_03_create_api',
        'test_04_search_api_basic',
        'test_10_full_workflow'
    ]
    
    # Create test suite with selected tests
    suite = unittest.TestSuite()
    for test_name in essential_tests:
        suite.addTest(TestCompanySearchAPI(test_name))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_all_tests():
    """Run the complete test suite."""
    from src.company_index.test_company_search_api import run_tests
    return run_tests()

def main():
    parser = argparse.ArgumentParser(
        description="Test Runner for Company Search API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/company_index/run_tests.py                # Run all tests
    python src/company_index/run_tests.py --quick        # Run essential tests only
    python src/company_index/run_tests.py --verbose      # Run with verbose output
        """
    )
    
    parser.add_argument('--quick', action='store_true', 
                       help='Run only essential tests (faster)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🧪 Company Search API Test Runner")
    print("=" * 50)
    
    if args.quick:
        print("Running essential tests only...")
        success = run_quick_tests()
    else:
        print("Running complete test suite...")
        success = run_all_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

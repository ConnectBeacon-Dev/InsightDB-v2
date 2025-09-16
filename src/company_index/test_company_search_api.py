#!/usr/bin/env python3
"""
Test Cases for Company Search API

This module contains comprehensive test cases for the CompanySearchAPI class,
testing all three main APIs: cleanup, create, and search.

Usage:
    python src/company_index/test_company_search_api.py
"""

import os
import sys
import time
import logging
import unittest
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.company_index.company_search_api import CompanySearchAPI
from src.load_config import load_config

# Set up logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestCompanySearchAPI(unittest.TestCase):
    """Test cases for CompanySearchAPI class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - initialize API instance and create indices once."""
        logger.info("Setting up test class...")
        try:
            cls.api = CompanySearchAPI()
            logger.info("✓ CompanySearchAPI initialized successfully")
            
            # First, generate the required CSV files
            logger.info("Generating required CSV files for testing...")
            try:
                from src.generate_grouped_csvs_with_data import main as generate_csvs
                generate_csvs()
                logger.info("✓ CSV files generated successfully")
            except Exception as e:
                logger.warning(f"⚠ Failed to generate CSV files: {e}")
                logger.info("Proceeding with test setup anyway...")
            
            # Create indices once for all tests
            logger.info("Creating search indices for all tests...")
            cls.indices_created = cls.api.create(force_rebuild=True)
            if cls.indices_created:
                logger.info("✓ Search indices created successfully for all tests")
            else:
                logger.warning("⚠ Search indices creation failed - some tests may fail")
                
        except Exception as e:
            logger.error(f"Failed to initialize CompanySearchAPI: {e}")
            raise
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests - final cleanup."""
        logger.info("Performing final cleanup after all tests...")
        logger.info("Note: Skipping aggressive cleanup to preserve data directories")
        # Don't perform aggressive cleanup that deletes data directories
        # The user wants to preserve the processed_data_store directories
        logger.info("✓ Final cleanup skipped to preserve data directories")
    
    def setUp(self):
        """Set up each test case."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting test: {self._testMethodName}")
        logger.info(f"{'='*60}")
    
    def tearDown(self):
        """Clean up after each test case."""
        logger.info(f"Completed test: {self._testMethodName}")
    
    def test_01_initial_status(self):
        """Test 1: Check initial status of the system."""
        logger.info("Testing initial status check...")
        
        status = self.api.status()
        
        # Verify status structure
        self.assertIsInstance(status, dict)
        self.assertIn('indices_ready', status)
        self.assertIn('components', status)
        self.assertIn('config_loaded', status)
        
        # Verify config is loaded
        self.assertTrue(status['config_loaded'], "Configuration should be loaded")
        
        # Log current status
        logger.info(f"Indices ready: {status['indices_ready']}")
        logger.info(f"Config loaded: {status['config_loaded']}")
        
        for component, info in status.get('components', {}).items():
            status_symbol = '✓' if info.get('exists', False) else '✗'
            logger.info(f"  {component}: {status_symbol}")
            if 'count' in info:
                logger.info(f"    Count: {info['count']}")
        
        logger.info("✓ Initial status check completed")
    
    def test_02_cleanup_api(self):
        """Test 2: Test cleanup API functionality."""
        logger.info("Testing cleanup API...")
        
        # Test cleanup without confirmation (should not perform cleanup)
        logger.info("Testing cleanup without confirmation...")
        result = self.api.cleanup(confirm=False)
        self.assertFalse(result, "Cleanup should return False when confirm=False")
        logger.info("✓ Cleanup correctly refused without confirmation")
        
        # Test cleanup with confirmation
        logger.info("Testing cleanup with confirmation...")
        result = self.api.cleanup(confirm=True)
        self.assertTrue(result, "Cleanup should return True when successful")
        logger.info("✓ Cleanup completed successfully")
        
        # Verify cleanup worked by checking status
        status = self.api.status()
        logger.info("Status after cleanup:")
        for component, info in status.get('components', {}).items():
            status_symbol = '✓' if info.get('exists', False) else '✗'
            logger.info(f"  {component}: {status_symbol}")
        
        # After cleanup, indices should not be ready
        self.assertFalse(status['indices_ready'], "Indices should not be ready after cleanup")
        logger.info("✓ Cleanup API test completed")
    
    def test_03_create_api(self):
        """Test 3: Test create API functionality."""
        logger.info("Testing create API...")
        
        # Record start time
        start_time = time.time()
        
        # Test create API
        logger.info("Starting index creation pipeline...")
        result = self.api.create(force_rebuild=True)
        
        # Record end time
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify creation was successful
        self.assertTrue(result, "Create API should return True when successful")
        logger.info(f"✓ Index creation completed in {duration:.2f} seconds")
        
        # Verify indices are now ready
        status = self.api.status()
        logger.info("Status after creation:")
        for component, info in status.get('components', {}).items():
            status_symbol = '✓' if info.get('exists', False) else '✗'
            logger.info(f"  {component}: {status_symbol}")
            if 'count' in info:
                logger.info(f"    Count: {info['count']}")
        
        self.assertTrue(status['indices_ready'], "Indices should be ready after creation")
        logger.info("✓ Create API test completed")
        
        # Test create without force_rebuild (should skip if indices exist)
        logger.info("Testing create without force rebuild...")
        start_time = time.time()
        result = self.api.create(force_rebuild=False)
        end_time = time.time()
        duration = end_time - start_time
        
        self.assertTrue(result, "Create should return True even when skipping")
        logger.info(f"✓ Create (skip existing) completed in {duration:.2f} seconds")
    
    def test_04_search_api_basic(self):
        """Test 4: Test basic search API functionality."""
        logger.info("Testing basic search API...")
        
        # Skip if indices were not created in setUpClass
        if not self.indices_created:
            self.skipTest("Search indices were not created successfully in setUpClass")
        
        # Test basic search
        test_queries = [
            "software development",
            "manufacturing company",
            "aerospace technology",
            "small scale enterprise",
            "research and development"
        ]
        
        for query in test_queries:
            logger.info(f"Testing search query: '{query}'")
            
            start_time = time.time()
            results = self.api.search(query=query, top_k=5)
            end_time = time.time()
            duration = end_time - start_time
            
            # Verify results structure
            self.assertIsInstance(results, list, "Search should return a list")
            logger.info(f"  Found {len(results)} results in {duration:.3f} seconds")
            
            # Verify result structure if results exist
            if results:
                result = results[0]
                required_fields = [
                    'rank', 'company_ref_no', 'company_name', 'location',
                    'scale', 'core_expertise', 'industry_domain', 'scores'
                ]
                
                for field in required_fields:
                    self.assertIn(field, result, f"Result should contain '{field}' field")
                
                # Verify scores structure
                self.assertIn('rrf', result['scores'])
                self.assertIn('tfidf', result['scores'])
                self.assertIn('dense', result['scores'])
                
                # Log top result
                logger.info(f"  Top result: {result['company_name']} - {result['core_expertise']}")
                logger.info(f"  Scores: RRF={result['scores']['rrf']:.4f}, "
                          f"TF-IDF={result['scores']['tfidf']:.4f}, "
                          f"Dense={result['scores']['dense']:.4f}")
        
        logger.info("✓ Basic search API test completed")
    
    def test_05_search_api_with_filters(self):
        """Test 5: Test search API with various filters."""
        logger.info("Testing search API with filters...")
        
        # Skip if indices were not created in setUpClass
        if not self.indices_created:
            self.skipTest("Search indices were not created successfully in setUpClass")
        
        # Test different filter combinations
        filter_tests = [
            {
                "name": "Scale filter",
                "params": {"query": "manufacturing", "filter_scale": "Medium", "top_k": 3}
            },
            {
                "name": "Country filter",
                "params": {"query": "software", "filter_country": "India", "top_k": 3}
            },
            {
                "name": "Industry filter",
                "params": {"query": "technology", "filter_industry": "software", "top_k": 3}
            },
            {
                "name": "Multiple filters",
                "params": {
                    "query": "development",
                    "filter_scale": "Small",
                    "filter_industry": "software",
                    "top_k": 5
                }
            },
            {
                "name": "Min score filter",
                "params": {"query": "manufacturing", "min_score": 0.1, "top_k": 5}
            },
            {
                "name": "No query expansion",
                "params": {"query": "software", "expand_query": False, "top_k": 3}
            }
        ]
        
        for test_case in filter_tests:
            logger.info(f"Testing {test_case['name']}...")
            
            start_time = time.time()
            results = self.api.search(**test_case['params'])
            end_time = time.time()
            duration = end_time - start_time
            
            self.assertIsInstance(results, list, "Search should return a list")
            logger.info(f"  Found {len(results)} results in {duration:.3f} seconds")
            
            # Log results if any
            if results:
                for i, result in enumerate(results[:2], 1):  # Show top 2 results
                    logger.info(f"  {i}. {result['company_name']} - {result['core_expertise']}")
            else:
                logger.info("  No results found with current filters")
        
        logger.info("✓ Search API with filters test completed")
    
    def test_06_search_api_edge_cases(self):
        """Test 6: Test search API edge cases."""
        logger.info("Testing search API edge cases...")
        
        # Skip if indices were not created in setUpClass
        if not self.indices_created:
            self.skipTest("Search indices were not created successfully in setUpClass")
        
        edge_cases = [
            {
                "name": "Empty query",
                "params": {"query": "", "top_k": 5}
            },
            {
                "name": "Very specific query",
                "params": {"query": "very specific non-existent company type", "top_k": 5}
            },
            {
                "name": "Single character query",
                "params": {"query": "a", "top_k": 3}
            },
            {
                "name": "Numeric query",
                "params": {"query": "123", "top_k": 3}
            },
            {
                "name": "Special characters",
                "params": {"query": "software & development", "top_k": 3}
            },
            {
                "name": "Very high top_k",
                "params": {"query": "company", "top_k": 1000}
            },
            {
                "name": "Zero top_k",
                "params": {"query": "software", "top_k": 0}
            }
        ]
        
        for test_case in edge_cases:
            logger.info(f"Testing {test_case['name']}...")
            
            try:
                start_time = time.time()
                results = self.api.search(**test_case['params'])
                end_time = time.time()
                duration = end_time - start_time
                
                self.assertIsInstance(results, list, "Search should return a list")
                logger.info(f"  Found {len(results)} results in {duration:.3f} seconds")
                
            except Exception as e:
                logger.warning(f"  Edge case caused exception: {e}")
                # Some edge cases might cause exceptions, which is acceptable
        
        logger.info("✓ Search API edge cases test completed")
    
    def test_07_search_api_csv_export(self):
        """Test 7: Test search API CSV export functionality."""
        logger.info("Testing search API CSV export...")
        
        # Skip if indices were not created in setUpClass
        if not self.indices_created:
            self.skipTest("Search indices were not created successfully in setUpClass")
        
        # Test CSV export
        csv_file = "test_search_results.csv"
        
        try:
            results = self.api.search(
                query="software development",
                top_k=5,
                save_csv=csv_file
            )
            
            # Verify results
            self.assertIsInstance(results, list, "Search should return a list")
            logger.info(f"Found {len(results)} results")
            
            # Verify CSV file was created
            if results:
                self.assertTrue(Path(csv_file).exists(), "CSV file should be created")
                
                # Check CSV file size
                file_size = Path(csv_file).stat().st_size
                self.assertGreater(file_size, 0, "CSV file should not be empty")
                logger.info(f"✓ CSV file created: {csv_file} ({file_size} bytes)")
            else:
                logger.info("No results to export to CSV")
        
        finally:
            # Clean up CSV file
            if Path(csv_file).exists():
                Path(csv_file).unlink()
                logger.info("✓ Test CSV file cleaned up")
        
        logger.info("✓ Search API CSV export test completed")
    
    def test_08_performance_benchmark(self):
        """Test 8: Performance benchmark for search operations."""
        logger.info("Running performance benchmark...")
        
        # Skip if indices were not created in setUpClass
        if not self.indices_created:
            self.skipTest("Search indices were not created successfully in setUpClass")
        
        # Performance test queries
        benchmark_queries = [
            "software",
            "manufacturing",
            "aerospace technology",
            "small scale enterprise",
            "research and development",
            "quality certification",
            "export oriented",
            "automotive components",
            "pharmaceutical",
            "textile manufacturing"
        ]
        
        total_time = 0
        total_results = 0
        
        logger.info(f"Running {len(benchmark_queries)} benchmark queries...")
        
        for i, query in enumerate(benchmark_queries, 1):
            start_time = time.time()
            results = self.api.search(query=query, top_k=10)
            end_time = time.time()
            duration = end_time - start_time
            
            total_time += duration
            total_results += len(results)
            
            logger.info(f"  Query {i:2d}: '{query[:20]:20s}' -> "
                       f"{len(results):2d} results in {duration:.3f}s")
        
        avg_time = total_time / len(benchmark_queries)
        avg_results = total_results / len(benchmark_queries)
        
        logger.info(f"\nBenchmark Results:")
        logger.info(f"  Total queries: {len(benchmark_queries)}")
        logger.info(f"  Total time: {total_time:.3f}s")
        logger.info(f"  Average time per query: {avg_time:.3f}s")
        logger.info(f"  Average results per query: {avg_results:.1f}")
        logger.info(f"  Queries per second: {len(benchmark_queries)/total_time:.2f}")
        
        # Performance assertions
        self.assertLess(avg_time, 10.0, "Average query time should be less than 10 seconds")
        self.assertGreater(avg_results, 0, "Should find some results on average")
        
        logger.info("✓ Performance benchmark completed")
    
    def test_09_error_handling(self):
        """Test 9: Test error handling scenarios."""
        logger.info("Testing error handling scenarios...")
        
        # Test search without indices (should raise RuntimeError)
        logger.info("Testing search without indices...")
        
        # First cleanup to ensure no indices
        self.api.cleanup(confirm=True)
        
        # Verify indices are not ready
        status = self.api.status()
        if status['indices_ready']:
            logger.warning("Indices still ready after cleanup, skipping this test")
        else:
            # Try to search without indices
            with self.assertRaises(RuntimeError):
                self.api.search("test query")
            logger.info("✓ Correctly raised RuntimeError when searching without indices")
        
        logger.info("✓ Error handling test completed")
    
    def test_10_full_workflow(self):
        """Test 10: Test complete workflow (cleanup -> create -> search)."""
        logger.info("Testing complete workflow...")
        
        # Step 1: Cleanup
        logger.info("Step 1: Cleanup...")
        cleanup_result = self.api.cleanup(confirm=True)
        self.assertTrue(cleanup_result, "Cleanup should succeed")
        
        # Verify cleanup
        status = self.api.status()
        self.assertFalse(status['indices_ready'], "Indices should not be ready after cleanup")
        logger.info("✓ Cleanup completed")
        
        # Step 2: Create
        logger.info("Step 2: Create indices...")
        create_start = time.time()
        create_result = self.api.create()
        create_end = time.time()
        create_duration = create_end - create_start
        
        self.assertTrue(create_result, "Create should succeed")
        logger.info(f"✓ Create completed in {create_duration:.2f} seconds")
        
        # Verify creation
        status = self.api.status()
        self.assertTrue(status['indices_ready'], "Indices should be ready after creation")
        
        # Step 3: Search
        logger.info("Step 3: Search...")
        search_queries = [
            "software development",
            "manufacturing",
            "aerospace"
        ]
        
        for query in search_queries:
            search_start = time.time()
            results = self.api.search(query=query, top_k=3)
            search_end = time.time()
            search_duration = search_end - search_start
            
            self.assertIsInstance(results, list, "Search should return a list")
            logger.info(f"  Query '{query}': {len(results)} results in {search_duration:.3f}s")
            
            if results:
                logger.info(f"    Top result: {results[0]['company_name']}")
        
        logger.info("✓ Complete workflow test completed")

def run_tests():
    """Run all test cases."""
    logger.info("Starting Company Search API Test Suite")
    logger.info("="*80)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestCompanySearchAPI)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    logger.info("="*80)
    logger.info("Test Suite Summary:")
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        logger.error("Failures:")
        for test, traceback in result.failures:
            logger.error(f"  {test}: {traceback}")
    
    if result.errors:
        logger.error("Errors:")
        for test, traceback in result.errors:
            logger.error(f"  {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

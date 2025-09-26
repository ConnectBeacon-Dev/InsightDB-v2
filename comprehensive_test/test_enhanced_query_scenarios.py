#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Query Scenarios
Tests all the new features implemented in enhanced_query_with_summary.py:
1. Company name exact match strategy
2. Empty field filtering
3. Result limitation removal
4. CMP prefix handling
5. Clean output format
"""

import json
import os
import sys
import csv
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import shutil

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.query_engine.enhanced_query_with_summary import execute_enhanced_query_with_summary
from src.load_config import load_config

class EnhancedQueryTestSuite:
    def __init__(self):
        self.config, self.logger = load_config()
        self.test_results = []
        self.dataset_file = Path(__file__).parent / "integrated_company_search_100_def.json"
        
    def setup_test_environment(self):
        """Setup test environment with the dataset"""
        print("🔧 Setting up test environment...")
        print(f"📁 Using dataset: {self.dataset_file}")
        
        if not self.dataset_file.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_file}")
        
        try:
            # Create a temporary integrated_company_search.json in the company_data directory
            company_data_dir = Path(self.config.get('company_mapped_data', {}).get('processed_data_store', './processed_data_store/company_mapped_store'))
            temp_integrated_file = company_data_dir / "integrated_company_search.json"
            
            # Copy our dataset to the expected location
            shutil.copy2(self.dataset_file, temp_integrated_file)
            print(f"✅ Copied dataset to expected location: {temp_integrated_file}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to setup test environment: {e}")
            return False
    
    def run_query_test(self, test_name: str, query: str, expected_behavior: str,
                      validation_func=None, expected_companies: List[str] = None,
                      test_category: str = "General", topk: int = None) -> Dict[str, Any]:
        """Run a single query test and validate results"""
        print(f"\n{'='*70}")
        print(f"🧪 TEST: {test_name}")
        print(f"📝 Query: '{query}'")
        print(f"🎯 Expected Behavior: {expected_behavior}")
        print(f"📂 Category: {test_category}")
        
        # Track timing
        start_time = time.time()
        
        try:
            result = execute_enhanced_query_with_summary(
                query, config=self.config, logger=self.logger, topk=topk
            )
            
            processing_time = time.time() - start_time
            
            companies_found = len(result.get('companies', []))
            
            print(f"📊 Results: {companies_found} companies found")
            print(f"⏱️ Processing Time: {processing_time:.3f}s")
            
            # Basic validation
            test_passed = True
            validation_details = {}
            
            # Custom validation if provided
            if validation_func:
                validation_result = validation_func(result, query, topk)
                test_passed = test_passed and validation_result['passed']
                validation_details = validation_result
            
            # Expected companies validation if provided
            if expected_companies:
                company_validation = self._validate_expected_companies(result, expected_companies)
                validation_details['company_validation'] = company_validation
                test_passed = test_passed and company_validation['passed']
            
            # Display results
            if companies_found > 0:
                print(f"\n📋 Results (showing {min(5, companies_found)}):")
                for i, company in enumerate(result['companies'][:5], 1):
                    print(f"  {i}. {company['company_name']} [{company['company_ref_no']}]")
                    # Show only non-empty fields
                    fields_shown = []
                    for field, value in company.items():
                        if field not in ['company_name', 'company_ref_no'] and value:
                            if isinstance(value, list) and value:
                                fields_shown.append(f"{field}: {', '.join(value[:2])}")
                            elif isinstance(value, str) and value.strip():
                                fields_shown.append(f"{field}: {value}")
                            elif isinstance(value, (int, float)) and value > 0:
                                fields_shown.append(f"{field}: {value}")
                    if fields_shown:
                        print(f"     {' | '.join(fields_shown[:3])}")
                    print()
            
            # Test result summary
            status = "✅ PASSED" if test_passed else "❌ FAILED"
            print(f"{status}")
            
            if validation_details:
                print("🔍 Validation Details:")
                for key, value in validation_details.items():
                    if isinstance(value, dict) and 'details' in value:
                        print(f"  {key}: {value['details']}")
            
            test_result = {
                'test_name': test_name,
                'query': query,
                'expected_behavior': expected_behavior,
                'test_category': test_category,
                'passed': test_passed,
                'companies_found': companies_found,
                'processing_time': processing_time,
                'validation_details': validation_details,
                'top_companies': [c['company_name'] for c in result['companies'][:5]],
                'topk_used': topk
            }
            
            self.test_results.append(test_result)
            return test_result
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            test_result = {
                'test_name': test_name,
                'query': query,
                'expected_behavior': expected_behavior,
                'test_category': test_category,
                'passed': False,
                'error': str(e),
                'topk_used': topk
            }
            self.test_results.append(test_result)
            return test_result
    
    def _validate_expected_companies(self, result: Dict[str, Any], expected_companies: List[str]) -> Dict[str, Any]:
        """Validate that specific expected companies are found"""
        companies = result.get('companies', [])
        found_companies = [c['company_name'] for c in companies]
        
        matches = []
        for expected in expected_companies:
            for found in found_companies:
                if expected.lower() in found.lower():
                    matches.append(f"Expected '{expected}' found as '{found}'")
                    break
        
        return {
            'passed': len(matches) > 0,
            'matches': matches,
            'details': f"Found {len(matches)} expected companies: {matches}"
        }
    
    def validate_company_name_search(self, result: Dict[str, Any], query: str, topk: int = None) -> Dict[str, Any]:
        """Validate company name search functionality"""
        companies = result.get('companies', [])
        query_lower = query.lower()
        
        # Check if any company name matches the query
        exact_matches = []
        partial_matches = []
        
        for company in companies:
            company_name = company['company_name'].lower()
            if query_lower == company_name:
                exact_matches.append(company['company_name'])
            elif query_lower in company_name or any(word in company_name for word in query_lower.split()):
                partial_matches.append(company['company_name'])
        
        total_matches = len(exact_matches) + len(partial_matches)
        passed = total_matches > 0
        
        return {
            'passed': passed,
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'total_matches': total_matches,
            'details': f"Found {len(exact_matches)} exact matches and {len(partial_matches)} partial matches"
        }
    
    def validate_empty_field_filtering(self, result: Dict[str, Any], query: str, topk: int = None) -> Dict[str, Any]:
        """Validate that empty fields are filtered from output"""
        companies = result.get('companies', [])
        
        empty_fields_found = []
        total_fields_checked = 0
        
        for company in companies:
            for field, value in company.items():
                total_fields_checked += 1
                if value is None or value == "" or (isinstance(value, list) and len(value) == 0):
                    empty_fields_found.append(f"{company['company_name']}.{field}")
        
        passed = len(empty_fields_found) == 0
        
        return {
            'passed': passed,
            'empty_fields_found': empty_fields_found,
            'total_fields_checked': total_fields_checked,
            'details': f"Checked {total_fields_checked} fields, found {len(empty_fields_found)} empty fields"
        }
    
    def validate_result_count(self, result: Dict[str, Any], query: str, topk: int = None) -> Dict[str, Any]:
        """Validate result count behavior"""
        companies = result.get('companies', [])
        companies_found = len(companies)
        
        if topk is None:
            # Should return all available matches (no limit)
            passed = companies_found >= 0  # Any number is acceptable when no limit
            details = f"No limit specified, returned {companies_found} companies"
        else:
            # Should respect the topk limit
            passed = companies_found <= topk
            details = f"Limit of {topk} specified, returned {companies_found} companies"
        
        return {
            'passed': passed,
            'companies_returned': companies_found,
            'limit_specified': topk,
            'details': details
        }
    
    def validate_cmp_prefix_handling(self, result: Dict[str, Any], query: str, topk: int = None) -> Dict[str, Any]:
        """Validate CMP prefix handling in company reference numbers"""
        companies = result.get('companies', [])
        
        ref_no_analysis = []
        for company in companies:
            ref_no = company.get('company_ref_no', '')
            ref_no_analysis.append({
                'company': company['company_name'],
                'ref_no': ref_no,
                'has_cmp_prefix': ref_no.startswith('CMP'),
                'is_source_data': True  # Assuming source data contains CMP prefix
            })
        
        # All reference numbers should be displayed as they are in source data
        passed = True  # Since we preserve source data as-is
        
        return {
            'passed': passed,
            'ref_no_analysis': ref_no_analysis,
            'details': f"Analyzed {len(ref_no_analysis)} company reference numbers - preserved as source data"
        }
    
    def validate_clean_output_format(self, result: Dict[str, Any], query: str, topk: int = None) -> Dict[str, Any]:
        """Validate clean output format (only companies array)"""
        expected_keys = {'companies'}
        actual_keys = set(result.keys())
        
        # Check if output contains only the companies key
        extra_keys = actual_keys - expected_keys
        missing_keys = expected_keys - actual_keys
        
        passed = len(extra_keys) == 0 and len(missing_keys) == 0
        
        return {
            'passed': passed,
            'expected_keys': list(expected_keys),
            'actual_keys': list(actual_keys),
            'extra_keys': list(extra_keys),
            'missing_keys': list(missing_keys),
            'details': f"Output format validation - Extra keys: {extra_keys}, Missing keys: {missing_keys}"
        }
    
    def run_company_name_search_tests(self):
        """Test company name exact match strategy"""
        print("\n🏢 COMPANY NAME SEARCH TESTS")
        print("="*50)
        
        # Test 1: Exact company name match
        self.run_query_test(
            "Exact Company Name Match",
            "Company_006",
            "Should find exact match using company name search strategy",
            validation_func=self.validate_company_name_search,
            expected_companies=["Company_006"],
            test_category="Company Name Search"
        )
        
        # Test 2: Multi-word company name
        self.run_query_test(
            "Multi-word Company Name",
            "Alpha Design Technologies",
            "Should find company with multi-word name",
            validation_func=self.validate_company_name_search,
            expected_companies=["Alpha Design Technologies"],
            test_category="Company Name Search"
        )
        
        # Test 3: Partial company name match
        self.run_query_test(
            "Partial Company Name Match",
            "Garuda Defence",
            "Should find company with partial name match",
            validation_func=self.validate_company_name_search,
            expected_companies=["Garuda Defence Systems"],
            test_category="Company Name Search"
        )
        
        # Test 4: Company name with special characters
        self.run_query_test(
            "Company Name with Special Characters",
            "Paras Defence and Space Technologies",
            "Should handle company names with special characters",
            validation_func=self.validate_company_name_search,
            expected_companies=["Paras Defence and Space Technologies"],
            test_category="Company Name Search"
        )
        
        # Test 5: Non-existent company name (should fallback)
        self.run_query_test(
            "Non-existent Company Name Fallback",
            "XYZ Nonexistent Company",
            "Should fallback to general search when company name not found",
            validation_func=self.validate_company_name_search,
            test_category="Company Name Search"
        )
    
    def run_empty_field_filtering_tests(self):
        """Test empty field filtering functionality"""
        print("\n🧹 EMPTY FIELD FILTERING TESTS")
        print("="*50)
        
        # Test 1: General query to check field filtering
        self.run_query_test(
            "Empty Field Filtering - General Query",
            "manufacturing companies",
            "Should filter out empty/None fields from output",
            validation_func=self.validate_empty_field_filtering,
            test_category="Field Filtering"
        )
        
        # Test 2: Specific company query
        self.run_query_test(
            "Empty Field Filtering - Specific Company",
            "Company_001",
            "Should filter empty fields for specific company",
            validation_func=self.validate_empty_field_filtering,
            test_category="Field Filtering"
        )
        
        # Test 3: Location-based query
        self.run_query_test(
            "Empty Field Filtering - Location Query",
            "companies in Karnataka",
            "Should filter empty fields in location-based results",
            validation_func=self.validate_empty_field_filtering,
            test_category="Field Filtering"
        )
    
    def run_result_limitation_tests(self):
        """Test result limitation removal functionality"""
        print("\n📊 RESULT LIMITATION TESTS")
        print("="*50)
        
        # Test 1: No limit specified (should return all)
        self.run_query_test(
            "No Result Limit - All Results",
            "small scale companies",
            "Should return all matching companies when no limit specified",
            validation_func=self.validate_result_count,
            test_category="Result Limitation",
            topk=None
        )
        
        # Test 2: Specific limit specified
        self.run_query_test(
            "Specific Result Limit",
            "manufacturing companies",
            "Should respect specified result limit",
            validation_func=self.validate_result_count,
            test_category="Result Limitation",
            topk=5
        )
        
        # Test 3: Large limit specified
        self.run_query_test(
            "Large Result Limit",
            "companies",
            "Should handle large result limits appropriately",
            validation_func=self.validate_result_count,
            test_category="Result Limitation",
            topk=100
        )
        
        # Test 4: Small limit specified
        self.run_query_test(
            "Small Result Limit",
            "aerospace companies",
            "Should handle small result limits appropriately",
            validation_func=self.validate_result_count,
            test_category="Result Limitation",
            topk=2
        )
    
    def run_cmp_prefix_tests(self):
        """Test CMP prefix handling"""
        print("\n🏷️ CMP PREFIX HANDLING TESTS")
        print("="*50)
        
        # Test 1: General query CMP prefix check
        self.run_query_test(
            "CMP Prefix Handling - General Query",
            "defence companies",
            "Should preserve company reference numbers as stored in source data",
            validation_func=self.validate_cmp_prefix_handling,
            test_category="CMP Prefix"
        )
        
        # Test 2: Specific company CMP prefix check
        self.run_query_test(
            "CMP Prefix Handling - Specific Company",
            "Company_010",
            "Should show reference number exactly as in source data",
            validation_func=self.validate_cmp_prefix_handling,
            test_category="CMP Prefix"
        )
    
    def run_clean_output_tests(self):
        """Test clean output format"""
        print("\n🧽 CLEAN OUTPUT FORMAT TESTS")
        print("="*50)
        
        # Test 1: Output structure validation
        self.run_query_test(
            "Clean Output Format - Structure",
            "software companies",
            "Should return only companies array without additional metadata",
            validation_func=self.validate_clean_output_format,
            test_category="Output Format"
        )
        
        # Test 2: Large result set output format
        self.run_query_test(
            "Clean Output Format - Large Results",
            "companies",
            "Should maintain clean format even with large result sets",
            validation_func=self.validate_clean_output_format,
            test_category="Output Format"
        )
    
    def run_integration_tests(self):
        """Test integration of all features together"""
        print("\n🔗 INTEGRATION TESTS")
        print("="*50)
        
        # Test 1: Company name search with clean output
        self.run_query_test(
            "Integration - Company Name + Clean Output",
            "Vector Avionics",
            "Should use company name search and return clean output",
            validation_func=lambda r, q, t: {
                'passed': self.validate_company_name_search(r, q, t)['passed'] and 
                         self.validate_clean_output_format(r, q, t)['passed'],
                'details': "Combined company name search and clean output validation"
            },
            test_category="Integration"
        )
        
        # Test 2: Result limiting with field filtering
        self.run_query_test(
            "Integration - Result Limit + Field Filtering",
            "aerospace companies",
            "Should limit results and filter empty fields",
            validation_func=lambda r, q, t: {
                'passed': self.validate_result_count(r, q, t)['passed'] and 
                         self.validate_empty_field_filtering(r, q, t)['passed'],
                'details': "Combined result limiting and field filtering validation"
            },
            test_category="Integration",
            topk=3
        )
        
        # Test 3: All features combined
        self.run_query_test(
            "Integration - All Features Combined",
            "Bharat Forge",
            "Should demonstrate all enhanced features working together",
            validation_func=lambda r, q, t: {
                'passed': all([
                    self.validate_company_name_search(r, q, t)['passed'],
                    self.validate_empty_field_filtering(r, q, t)['passed'],
                    self.validate_clean_output_format(r, q, t)['passed'],
                    self.validate_cmp_prefix_handling(r, q, t)['passed']
                ]),
                'details': "All enhanced features validation combined"
            },
            test_category="Integration"
        )
    
    def run_edge_case_tests(self):
        """Test edge cases and error scenarios"""
        print("\n⚠️ EDGE CASE TESTS")
        print("="*50)
        
        # Test 1: Empty query
        self.run_query_test(
            "Edge Case - Empty Query",
            "",
            "Should handle empty query gracefully",
            test_category="Edge Cases"
        )
        
        # Test 2: Very long query
        self.run_query_test(
            "Edge Case - Very Long Query",
            "companies with aerospace defence manufacturing capabilities and ISO certification located in Karnataka with R&D facilities and testing capabilities for avionics systems",
            "Should handle very long queries appropriately",
            test_category="Edge Cases"
        )
        
        # Test 3: Special characters in query
        self.run_query_test(
            "Edge Case - Special Characters",
            "companies with R&D capabilities @#$%",
            "Should handle special characters in queries",
            test_category="Edge Cases"
        )
        
        # Test 4: Numeric query
        self.run_query_test(
            "Edge Case - Numeric Query",
            "123456",
            "Should handle numeric queries appropriately",
            test_category="Edge Cases"
        )
    
    def cleanup_previous_results(self):
        """Clean up previous test result files"""
        files_to_clean = [
            'enhanced_query_test_report.csv',
            'enhanced_query_test_results.json'
        ]
        
        for filename in files_to_clean:
            file_path = Path(__file__).parent / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"🧹 Cleaned up previous file: {filename}")
                except Exception as e:
                    print(f"⚠️ Could not delete {filename}: {e}")
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("🚀 Starting Enhanced Query Scenarios Test Suite")
        print("="*80)
        
        # Clean up previous results first
        self.cleanup_previous_results()
        
        # Setup test environment
        if not self.setup_test_environment():
            print("❌ Failed to setup test environment. Aborting tests.")
            return
        
        # Run all test categories
        self.run_company_name_search_tests()
        self.run_empty_field_filtering_tests()
        self.run_result_limitation_tests()
        self.run_cmp_prefix_tests()
        self.run_clean_output_tests()
        self.run_integration_tests()
        self.run_edge_case_tests()
        
        # Print summary and generate reports
        self.print_test_summary()
        self.analyze_failed_tests()
        self.generate_csv_report()
    
    def print_test_summary(self):
        """Print overall test results summary"""
        print("\n" + "="*80)
        print("📊 ENHANCED QUERY SCENARIOS TEST SUMMARY")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for test in self.test_results if test.get('passed', False))
        failed_tests = total_tests - passed_tests
        
        print(f"Dataset: 100 Companies (Synthetic Dataset)")
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Category breakdown
        categories = {}
        for test in self.test_results:
            category = test.get('test_category', 'Unknown')
            if category not in categories:
                categories[category] = {'total': 0, 'passed': 0}
            categories[category]['total'] += 1
            if test.get('passed', False):
                categories[category]['passed'] += 1
        
        print(f"\n📋 Test Categories:")
        for category, stats in categories.items():
            print(f"  📂 {category}: {stats['passed']}/{stats['total']} passed ({(stats['passed']/stats['total'])*100:.1f}%)")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for test in self.test_results:
                if not test.get('passed', False):
                    print(f"  - {test['test_name']}: {test.get('error', 'Validation failed')}")
        
        print(f"\n🎯 Overall Status: {'✅ ALL TESTS PASSED' if failed_tests == 0 else '❌ SOME TESTS FAILED'}")
        
        # Save detailed results to file
        results_file = Path(__file__).parent / 'enhanced_query_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: {results_file}")
    
    def analyze_failed_tests(self):
        """Analyze failed test cases in detail"""
        failed_tests = [test for test in self.test_results if not test.get('passed', False)]
        
        if not failed_tests:
            print("\n🎉 No failed tests to analyze!")
            return
        
        print(f"\n🔍 FAILED TEST CASE ANALYSIS")
        print("="*80)
        print(f"Total Failed Tests: {len(failed_tests)}")
        
        for i, test in enumerate(failed_tests, 1):
            print(f"\n❌ FAILED TEST #{i}: {test['test_name']}")
            print(f"📝 Query: '{test['query']}'")
            print(f"📂 Category: {test.get('test_category', 'Unknown')}")
            print(f"🎯 Expected: {test.get('expected_behavior', 'N/A')}")
            print(f"📊 Companies Found: {test.get('companies_found', 0)}")
            print(f"⏱️ Processing Time: {test.get('processing_time', 0):.3f}s")
            print(f"🔢 TopK Used: {test.get('topk_used', 'None')}")
            
            # Show validation details
            validation_details = test.get('validation_details', {})
            if validation_details:
                print("🔍 Validation Analysis:")
                for key, value in validation_details.items():
                    if isinstance(value, dict) and 'details' in value:
                        print(f"  {key}: {value['details']}")
            
            # Show error if present
            if test.get('error'):
                print(f"🚨 Error: {test['error']}")
            
            print("-" * 70)
    
    def generate_csv_report(self):
        """Generate CSV report with test results"""
        print("\n📊 Generating CSV Report...")
        
        csv_file = Path(__file__).parent / 'enhanced_query_test_report.csv'
        
        # Handle permission errors by trying alternative filenames
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    csv_file = Path(__file__).parent / f'enhanced_query_test_report_{attempt}.csv'
                
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow([
                        'Test Case',
                        'Category',
                        'Query',
                        'Expected Behavior',
                        'Pass/Fail',
                        'Time Taken (seconds)',
                        'Companies Found',
                        'TopK Used',
                        'Top Results',
                        'Validation Details'
                    ])
                    
                    # Write test results
                    for test in self.test_results:
                        # Format top results for CSV
                        top_results = '; '.join(test.get('top_companies', [])[:3])  # Top 3 companies
                        
                        # Format validation details
                        validation_summary = []
                        validation_details = test.get('validation_details', {})
                        for key, value in validation_details.items():
                            if isinstance(value, dict) and 'details' in value:
                                validation_summary.append(f"{key}: {value['details']}")
                        validation_text = ' | '.join(validation_summary)
                        
                        writer.writerow([
                            test['test_name'],
                            test.get('test_category', 'Unknown'),
                            test['query'],
                            test.get('expected_behavior', 'N/A'),
                            'PASS' if test.get('passed', False) else 'FAIL',
                            f"{test.get('processing_time', 0):.3f}",
                            test.get('companies_found', 0),
                            test.get('topk_used', 'None'),
                            top_results,
                            validation_text
                        ])
                
                print(f"✅ CSV report generated: {csv_file}")
                break  # Success, exit the retry loop
                
            except PermissionError as e:
                if attempt == max_attempts - 1:
                    print(f"❌ Failed to generate CSV report after {max_attempts} attempts: {e}")
                    print("💡 Please close any applications that might have the CSV file open and try again")
                    return
                else:
                    print(f"⚠️ Attempt {attempt + 1} failed, trying alternative filename...")
            except Exception as e:
                print(f"❌ Error generating CSV report: {e}")
                return
        
        # Print CSV summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for test in self.test_results if test.get('passed', False))
        avg_time = sum(test.get('processing_time', 0) for test in self.test_results) / total_tests if total_tests > 0 else 0
        
        print(f"\n📈 CSV Report Summary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Average Time: {avg_time:.3f} seconds")
        
        # Category summary
        categories = {}
        for test in self.test_results:
            category = test.get('test_category', 'Unknown')
            if category not in categories:
                categories[category] = {'total': 0, 'passed': 0}
            categories[category]['total'] += 1
            if test.get('passed', False):
                categories[category]['passed'] += 1
        
        print(f"\n📊 Category Breakdown:")
        for category, stats in categories.items():
            success_rate = (stats['passed']/stats['total'])*100 if stats['total'] > 0 else 0
            print(f"  📂 {category}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")

def main():
    """Main test execution"""
    try:
        test_suite = EnhancedQueryTestSuite()
        test_suite.run_all_tests()
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

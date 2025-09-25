#!/usr/bin/env python3
"""
Comprehensive Test Suite for 100 Company Dataset using TF-IDF API
Tests company_tfidf_api() with integrated_company_search_100_def.json
Includes test cases for certifications, locations, and LLM scenarios
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

from src.company_index.company_tfidf_api import company_tfidf_api
from src.query_engine.enhanced_query_with_summary import execute_enhanced_query_with_summary
from src.load_config import load_config

class CompanyTFIDFTestSuite:
    def __init__(self):
        self.config, self.logger = load_config()
        self.test_results = []
        self.dataset_file = Path(__file__).parent / "integrated_company_search_100_def.json"
        self.temp_tfidf_dir = None
        
    def setup_tfidf_index(self):
        """Setup TF-IDF index using the 100 company dataset"""
        print("🔧 Setting up TF-IDF index for 100 company dataset...")
        print(f"📁 Using dataset: {self.dataset_file}")
        
        if not self.dataset_file.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_file}")
        
        try:
            # Call company_tfidf_api with the dataset
            company_tfidf_api(str(self.dataset_file))
            print("✅ TF-IDF index created successfully")
            
            # Create a temporary integrated_company_search.json in the company_data directory
            # This ensures the query system uses our dataset
            company_data_dir = Path(self.config.get('company_mapped_data', {}).get('processed_data_store', './processed_data_store/company_mapped_store'))
            temp_integrated_file = company_data_dir / "integrated_company_search.json"
            
            # Copy our dataset to the expected location
            import shutil
            shutil.copy2(self.dataset_file, temp_integrated_file)
            print(f"✅ Copied dataset to expected location: {temp_integrated_file}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to create TF-IDF index: {e}")
            return False
    
    def run_query_test(self, test_name: str, query: str, expected_min_results: int = 1, 
                      validation_func=None, expected_keywords: List[str] = None,
                      expected_companies: List[str] = None, is_llm_specific: bool = False) -> Dict[str, Any]:
        """Run a single query test and validate results"""
        print(f"\n{'='*70}")
        print(f"🧪 TEST: {test_name}")
        print(f"📝 Query: '{query}'")
        print(f"🎯 Expected minimum results: {expected_min_results}")
        
        # Track timing
        start_time = time.time()
        
        try:
            result = execute_enhanced_query_with_summary(
                query, config=self.config, logger=self.logger
            )
            
            processing_time = time.time() - start_time
            
            companies_found = result['results']['companies_count']
            confidence = result['confidence']
            strategy = result['strategy']
            
            print(f"📊 Results: {companies_found} companies found")
            print(f"📈 Confidence: {confidence:.2f}")
            print(f"🔧 Strategy: {strategy}")
            
            # Basic validation
            test_passed = companies_found >= expected_min_results
            
            # Custom validation if provided
            validation_details = {}
            if validation_func:
                validation_result = validation_func(result)
                test_passed = test_passed and validation_result['passed']
                validation_details = validation_result
            
            # Keyword validation if provided
            if expected_keywords:
                keyword_validation = self._validate_keywords(result, expected_keywords)
                test_passed = test_passed and keyword_validation['passed']
                validation_details['keyword_validation'] = keyword_validation
            
            # Expected companies validation if provided
            if expected_companies:
                company_validation = self._validate_expected_companies(result, expected_companies)
                validation_details['company_validation'] = company_validation
            
            # Display top results
            if companies_found > 0:
                print(f"\n📋 Top {min(5, companies_found)} Results:")
                for i, company in enumerate(result['results']['companies'][:5], 1):
                    print(f"  {i}. {company['company_name']} [{company['company_ref_no']}]")
                    print(f"     Domain: {company.get('domain', 'N/A')}")
                    print(f"     Industry: {company.get('industry_domain', 'N/A')}")
                    print(f"     Location: {company.get('city', 'N/A')}, {company.get('state', 'N/A')}")
                    if company.get('certifications'):
                        print(f"     Certifications: {', '.join(company['certifications'][:3])}")
                    if company.get('rd_categories'):
                        print(f"     R&D: {', '.join(company['rd_categories'][:3])}")
                    if company.get('testing_categories'):
                        print(f"     Testing: {', '.join(company['testing_categories'][:3])}")
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
                'passed': test_passed,
                'companies_found': companies_found,
                'confidence': confidence,
                'strategy': strategy,
                'processing_time': processing_time,
                'is_llm_specific': is_llm_specific,
                'validation_details': validation_details,
                'top_companies': [c['company_name'] for c in result['results']['companies'][:5]]
            }
            
            self.test_results.append(test_result)
            return test_result
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            test_result = {
                'test_name': test_name,
                'query': query,
                'passed': False,
                'error': str(e)
            }
            self.test_results.append(test_result)
            return test_result
    
    def _validate_keywords(self, result: Dict[str, Any], expected_keywords: List[str]) -> Dict[str, Any]:
        """Validate that results contain expected keywords"""
        companies = result['results']['companies']
        keyword_matches = {keyword: 0 for keyword in expected_keywords}
        
        for company in companies:
            company_text = ' '.join([
                str(company.get('company_name', '')),
                str(company.get('domain', '')),
                str(company.get('industry_domain', '')),
                str(company.get('industry_subdomain', '')),
                str(company.get('core_expertise', '')),
                str(company.get('city', '')),
                str(company.get('state', '')),
                ' '.join(company.get('certifications', [])),
                ' '.join(company.get('rd_categories', [])),
                ' '.join(company.get('testing_categories', []))
            ]).lower()
            
            for keyword in expected_keywords:
                if keyword.lower() in company_text:
                    keyword_matches[keyword] += 1
        
        total_matches = sum(keyword_matches.values())
        passed = total_matches > 0
        
        return {
            'passed': passed,
            'matches': keyword_matches,
            'total_matches': total_matches,
            'details': f"Found {total_matches} keyword matches: {keyword_matches}"
        }
    
    def _validate_expected_companies(self, result: Dict[str, Any], expected_companies: List[str]) -> Dict[str, Any]:
        """Validate that specific expected companies are found"""
        companies = result['results']['companies']
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
    
    def validate_certification_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate certification search results"""
        companies = result['results']['companies']
        cert_found = False
        cert_details = []
        
        for company in companies:
            if company.get('certifications'):
                cert_found = True
                cert_details.append(f"{company['company_name']}: {', '.join(company['certifications'])}")
        
        return {
            'passed': cert_found,
            'details': f"Found {len(cert_details)} companies with certifications"
        }
    
    def validate_location_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate location-based search results"""
        companies = result['results']['companies']
        location_found = False
        location_details = []
        
        for company in companies:
            city = company.get('city', '')
            state = company.get('state', '')
            if city or state:
                location_found = True
                location_details.append(f"{company['company_name']}: {city}, {state}")
        
        return {
            'passed': location_found,
            'details': f"Found {len(location_details)} companies with location data"
        }
    
    def validate_defence_domain_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate defence domain search results"""
        companies = result['results']['companies']
        defence_found = False
        defence_details = []
        
        for company in companies:
            domain = company.get('industry_domain', '').lower()
            subdomain = company.get('industry_subdomain', '').lower()
            if 'defence' in domain or 'aerospace' in domain or any(term in subdomain for term in ['missile', 'naval', 'radar', 'avionics']):
                defence_found = True
                defence_details.append(f"{company['company_name']}: {domain} - {subdomain}")
        
        return {
            'passed': defence_found,
            'details': f"Found {len(defence_details)} defence/aerospace companies"
        }
    
    def validate_rd_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate R&D capability search results"""
        companies = result['results']['companies']
        rd_found = False
        rd_details = []
        
        for company in companies:
            if company.get('rd_categories'):
                rd_found = True
                rd_details.append(f"{company['company_name']}: {', '.join(company['rd_categories'])}")
        
        return {
            'passed': rd_found,
            'details': f"Found {len(rd_details)} companies with R&D capabilities"
        }
    
    def run_certification_tests(self):
        """Run certification-related test cases"""
        print("\n🏆 CERTIFICATION TEST CASES")
        print("="*50)
        
        # Test 1: ISO 9001 Certification
        self.run_query_test(
            "ISO 9001 Certification Search",
            "companies with ISO 9001 certification",
            expected_min_results=1,
            validation_func=self.validate_certification_results,
            expected_keywords=["ISO", "9001"],
            expected_companies=["Cochin Shipyard Limited", "Solar Industries", "Garuda Defence Systems"]
        )
        
        # Test 2: AS9100D Certification
        self.run_query_test(
            "AS9100D Aerospace Certification",
            "companies with AS9100D certification",
            expected_min_results=1,
            validation_func=self.validate_certification_results,
            expected_keywords=["AS9100D"],
            expected_companies=["Alpha Design Technologies", "Pragati Metals", "Suryansh Composites"]
        )
        
        # Test 3: NABL Accreditation
        self.run_query_test(
            "NABL Accredited Companies",
            "NABL accredited companies",
            expected_min_results=1,
            validation_func=self.validate_certification_results,
            expected_keywords=["NABL", "accreditation"],
            expected_companies=["Agni NavCom", "Goa Shipyard Limited", "Bharat Forge"]
        )
        
        # Test 4: ISO 27001 Security Certification
        self.run_query_test(
            "ISO 27001 Security Certification",
            "companies with ISO 27001 security certification",
            expected_min_results=1,
            validation_func=self.validate_certification_results,
            expected_keywords=["ISO", "27001"],
            expected_companies=["Vector Avionics", "Mahindra Defence Systems", "BEL Optronic Devices"]
        )
    
    def run_location_tests(self):
        """Run location-based test cases"""
        print("\n🌍 LOCATION TEST CASES")
        print("="*50)
        
        # Test 1: Karnataka Companies
        self.run_query_test(
            "Karnataka Based Companies",
            "companies in Karnataka",
            expected_min_results=1,
            validation_func=self.validate_location_results,
            expected_keywords=["Karnataka", "Bengaluru", "Mangaluru"],
            expected_companies=["Vayu Metals", "Alpha Design Technologies"]
        )
        
        # Test 2: Bengaluru Companies
        self.run_query_test(
            "Bengaluru Companies",
            "companies in Bengaluru",
            expected_min_results=1,
            validation_func=self.validate_location_results,
            expected_keywords=["Bengaluru", "Karnataka"],
            expected_companies=["Solar Industries", "ideaForge Technology"]
        )
        
        # Test 3: Hyderabad Companies
        self.run_query_test(
            "Hyderabad Companies",
            "companies in Hyderabad",
            expected_min_results=1,
            validation_func=self.validate_location_results,
            expected_keywords=["Hyderabad", "Telangana"],
            expected_companies=["Bharat Forge", "Paras Defence"]
        )
        
        # Test 4: Goa Based Companies
        self.run_query_test(
            "Goa Based Companies",
            "companies in Goa",
            expected_min_results=1,
            validation_func=self.validate_location_results,
            expected_keywords=["Goa", "Panaji", "Vasco"],
            expected_companies=["Agnikul Mechatronics", "Agni NavCom"]
        )
        
        # Test 5: Multi-state Search
        self.run_query_test(
            "Multi-state Companies",
            "companies in Karnataka, Tamil Nadu, and Andhra Pradesh",
            expected_min_results=2,
            validation_func=self.validate_location_results,
            expected_keywords=["Karnataka", "Tamil Nadu", "Andhra Pradesh"]
        )
    
    def run_llm_scenario_tests(self):
        """Run LLM-specific scenario test cases"""
        print("\n🤖 LLM SCENARIO TEST CASES")
        print("="*50)
        
        # Test 1: Complex Multi-criteria Query
        self.run_query_test(
            "Complex Multi-criteria Query",
            "Find aerospace companies with ISO certification and avionics expertise",
            expected_min_results=1,
            validation_func=self.validate_defence_domain_results,
            expected_keywords=["aerospace", "avionics", "ISO"]
        )
        
        # Test 2: Natural Language Query
        self.run_query_test(
            "Natural Language Query",
            "Which companies can manufacture missile components and have testing facilities?",
            expected_min_results=1,
            expected_keywords=["missile", "testing", "manufacturing"]
        )
        
        # Test 3: Capability-based Query
        self.run_query_test(
            "R&D Capability Query",
            "Companies with rocket motor lab and composite layup capabilities",
            expected_min_results=1,
            validation_func=self.validate_rd_results,
            expected_keywords=["rocket", "motor", "composite", "layup"]
        )
        
        # Test 4: Scale and Domain Query
        self.run_query_test(
            "Scale and Domain Query",
            "Large scale companies with RF and microwave expertise",
            expected_min_results=1,
            expected_keywords=["large", "RF", "microwave"]
        )
        
        # Test 5: Product-specific Query
        self.run_query_test(
            "Product-specific Query",
            "Companies manufacturing flight control computers and inertial navigation units",
            expected_min_results=1,
            expected_keywords=["flight", "control", "computer", "inertial", "navigation"]
        )
        
        # Test 6: Platform-specific Query
        self.run_query_test(
            "Platform-specific Query",
            "Companies working on LCA Tejas and Su-30MKI platforms",
            expected_min_results=1,
            expected_keywords=["LCA", "Tejas", "Su-30MKI", "platform"]
        )
        
        # Test 7: Technology Area Query
        self.run_query_test(
            "Technology Area Query",
            "Companies with expertise in EW systems and radar technology",
            expected_min_results=1,
            expected_keywords=["EW", "electronic", "warfare", "radar"]
        )
        
        # Test 8: Export-oriented Query
        self.run_query_test(
            "Export-oriented Query",
            "Companies that export products with ITAR-free components",
            expected_min_results=1,
            expected_keywords=["export", "ITAR-free"],
            is_llm_specific=True
        )
        
        # Test 9: Finance-related Query - Revenue Scale
        self.run_query_test(
            "Revenue Scale Query",
            "Companies with high revenue and strong financial performance",
            expected_min_results=1,
            expected_keywords=["revenue", "financial", "turnover"],
            is_llm_specific=True
        )
        
        # Test 10: Finance-related Query - Investment Capability
        self.run_query_test(
            "Investment Capability Query",
            "Companies seeking investment or funding for expansion",
            expected_min_results=1,
            expected_keywords=["investment", "funding", "expansion"],
            is_llm_specific=True
        )
        
        # Test 11: Finance-related Query - Cost-effective Solutions
        self.run_query_test(
            "Cost-effective Solutions Query",
            "Companies offering cost-effective and budget-friendly solutions",
            expected_min_results=1,
            expected_keywords=["cost-effective", "budget", "affordable"],
            is_llm_specific=True
        )
    
    def run_combination_scenario_tests(self):
        """Run combination scenario test cases"""
        print("\n🔗 COMBINATION SCENARIO TEST CASES")
        print("="*50)
        
        # Test 1: Radar + Location + R&D + Testing
        self.run_query_test(
            "Radar Technologies in India with R&D and Testing",
            "List companies having radar technologies located in India, having test & RD facilities",
            expected_min_results=1,
            validation_func=self.validate_rd_results,
            expected_keywords=["radar", "India", "test", "RD", "R&D"],
            is_llm_specific=True
        )
        
        # Test 2: Aerospace + Certification + Location
        self.run_query_test(
            "Certified Aerospace Companies in South India",
            "Find aerospace companies with ISO certification located in Karnataka or Tamil Nadu",
            expected_min_results=1,
            validation_func=self.validate_defence_domain_results,
            expected_keywords=["aerospace", "ISO", "Karnataka", "Tamil Nadu"],
            is_llm_specific=True
        )
        
        # Test 3: Manufacturing + Scale + Export
        self.run_query_test(
            "Large Manufacturing Companies with Export Capability",
            "Large scale manufacturing companies that can export defence products internationally",
            expected_min_results=1,
            expected_keywords=["large", "manufacturing", "export", "defence", "international"],
            is_llm_specific=True
        )
        
        # Test 4: Electronics + Testing + Certification
        self.run_query_test(
            "Electronics Companies with Testing and Quality Certification",
            "Electronics companies having testing facilities and quality certifications like NABL or ISO",
            expected_min_results=1,
            expected_keywords=["electronics", "testing", "NABL", "ISO", "quality"],
            is_llm_specific=True
        )
        
        # Test 5: Composites + R&D + Location
        self.run_query_test(
            "Composite Material Companies with R&D in Western India",
            "Companies specializing in composite materials with R&D capabilities in Maharashtra or Gujarat",
            expected_min_results=1,
            validation_func=self.validate_rd_results,
            expected_keywords=["composite", "materials", "R&D", "Maharashtra", "Gujarat"],
            is_llm_specific=True
        )
        
        # Test 6: Software + Defence + Certification
        self.run_query_test(
            "Software Companies for Defence with Security Certification",
            "Software development companies working on defence projects with ISO 27001 security certification",
            expected_min_results=1,
            expected_keywords=["software", "defence", "ISO 27001", "security"],
            is_llm_specific=True
        )
    
    def run_company_specific_tests(self):
        """Run company-specific test cases"""
        print("\n🏢 COMPANY-SPECIFIC TEST CASES")
        print("="*50)
        
        # Test 1: Location of Garuda Defence Systems
        self.run_query_test(
            "Location of Garuda Defence Systems",
            "location of Garuda Defence Systems",
            expected_min_results=1,
            validation_func=self.validate_location_results,
            expected_keywords=["Garuda", "Defence", "Systems"],
            expected_companies=["Garuda Defence Systems"],
            is_llm_specific=True
        )
        
        # Test 2: Revenue of Agni NavCom
        self.run_query_test(
            "Revenue of Agni NavCom",
            "Revenue of Agni NavCom",
            expected_min_results=1,
            expected_keywords=["Agni", "NavCom"],
            expected_companies=["Agni NavCom"],
            is_llm_specific=True
        )
        
        # Test 3: Companies having hard terrain testing
        self.run_query_test(
            "Companies with Hard Terrain Testing",
            "Companies having hard terrain testing",
            expected_min_results=1,
            expected_keywords=["hard", "terrain", "testing"],
            is_llm_specific=True
        )
        
        # Test 4: Specific company domain query
        self.run_query_test(
            "Alpha Design Technologies Domain",
            "What is the domain of Alpha Design Technologies",
            expected_min_results=1,
            expected_keywords=["Alpha", "Design", "Technologies"],
            expected_companies=["Alpha Design Technologies"],
            is_llm_specific=True
        )
        
        # Test 5: Company certification query
        self.run_query_test(
            "Vector Avionics Certifications",
            "What certifications does Vector Avionics have",
            expected_min_results=1,
            validation_func=self.validate_certification_results,
            expected_keywords=["Vector", "Avionics", "certifications"],
            expected_companies=["Vector Avionics"],
            is_llm_specific=True
        )
        
        # Test 6: Company R&D capabilities
        self.run_query_test(
            "Paras Defence R&D Capabilities",
            "R&D capabilities of Paras Defence and Space Technologies",
            expected_min_results=1,
            validation_func=self.validate_rd_results,
            expected_keywords=["Paras", "Defence", "R&D", "capabilities"],
            expected_companies=["Paras Defence and Space Technologies"],
            is_llm_specific=True
        )
    
    def cleanup_previous_results(self):
        """Clean up previous test result files"""
        files_to_clean = [
            'company_100_def_test_report.csv',
            'company_100_def_test_results.json'
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
        print("🚀 Starting 100 Company TF-IDF Comprehensive Test Suite")
        print("="*80)
        
        # Clean up previous results first
        self.cleanup_previous_results()
        
        # Setup TF-IDF index first
        if not self.setup_tfidf_index():
            print("❌ Failed to setup TF-IDF index. Aborting tests.")
            return
        
        # Run all test categories
        self.run_certification_tests()
        self.run_location_tests()
        self.run_llm_scenario_tests()
        self.run_combination_scenario_tests()
        self.run_company_specific_tests()
        
        # Print summary and generate CSV report
        self.print_test_summary()
        self.analyze_failed_tests()
        self.generate_csv_report()
    
    def print_test_summary(self):
        """Print overall test results summary"""
        print("\n" + "="*80)
        print("📊 100 COMPANY TF-IDF TEST SUMMARY")
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
        cert_tests = [t for t in self.test_results if 'certification' in t['test_name'].lower()]
        location_tests = [t for t in self.test_results if 'location' in t['test_name'].lower() or any(loc in t['test_name'].lower() for loc in ['karnataka', 'bengaluru', 'hyderabad', 'goa'])]
        llm_tests = [t for t in self.test_results if t not in cert_tests and t not in location_tests]
        
        print(f"\n📋 Test Categories:")
        print(f"  🏆 Certification Tests: {len(cert_tests)} ({sum(1 for t in cert_tests if t.get('passed', False))}/{len(cert_tests)} passed)")
        print(f"  🌍 Location Tests: {len(location_tests)} ({sum(1 for t in location_tests if t.get('passed', False))}/{len(location_tests)} passed)")
        print(f"  🤖 LLM Scenario Tests: {len(llm_tests)} ({sum(1 for t in llm_tests if t.get('passed', False))}/{len(llm_tests)} passed)")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for test in self.test_results:
                if not test.get('passed', False):
                    print(f"  - {test['test_name']}: {test.get('error', 'Validation failed')}")
        
        print(f"\n🎯 Overall Status: {'✅ ALL TESTS PASSED' if failed_tests == 0 else '❌ SOME TESTS FAILED'}")
        
        # Save detailed results to file
        results_file = Path(__file__).parent / 'company_100_def_test_results.json'
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
            print(f"🔧 Strategy: {test.get('strategy', 'N/A')}")
            print(f"📊 Companies Found: {test.get('companies_found', 0)}")
            print(f"📈 Confidence: {test.get('confidence', 0):.3f}")
            print(f"⏱️ Processing Time: {test.get('processing_time', 0):.3f}s")
            print(f"🤖 LLM Specific: {'YES' if test.get('is_llm_specific', False) else 'NO'}")
            
            # Show validation details
            validation_details = test.get('validation_details', {})
            if validation_details:
                print("🔍 Validation Analysis:")
                
                # Keyword validation analysis
                if 'keyword_validation' in validation_details:
                    kv = validation_details['keyword_validation']
                    print(f"  📋 Keyword Matches: {kv.get('total_matches', 0)}")
                    matches = kv.get('matches', {})
                    for keyword, count in matches.items():
                        status = "✅" if count > 0 else "❌"
                        print(f"    {status} '{keyword}': {count} matches")
                
                # Company validation analysis
                if 'company_validation' in validation_details:
                    cv = validation_details['company_validation']
                    expected_found = len(cv.get('matches', []))
                    print(f"  🏢 Expected Companies Found: {expected_found}")
                    for match in cv.get('matches', []):
                        print(f"    ✅ {match}")
            
            # Show top results for context
            top_companies = test.get('top_companies', [])
            if top_companies:
                print(f"🏆 Top Results Found:")
                for j, company in enumerate(top_companies[:3], 1):
                    print(f"  {j}. {company}")
            
            # Failure reason analysis
            print("💡 Failure Analysis:")
            if test.get('error'):
                print(f"  🚨 Error: {test['error']}")
            elif test.get('companies_found', 0) == 0:
                print("  📭 No companies found - query may be too specific or dataset lacks relevant data")
            elif validation_details.get('keyword_validation', {}).get('total_matches', 0) == 0:
                print("  🔤 No keyword matches - expected keywords not found in results")
                print("  💭 Suggestion: Check if keywords exist in the dataset or adjust expectations")
            else:
                print("  ❓ Validation criteria not met despite finding companies")
            
            print("-" * 70)
        
        # Summary of failure patterns
        print(f"\n📊 FAILURE PATTERN ANALYSIS")
        print("="*50)
        
        # Group by failure type
        keyword_failures = [t for t in failed_tests if t.get('validation_details', {}).get('keyword_validation', {}).get('total_matches', 0) == 0]
        no_results_failures = [t for t in failed_tests if t.get('companies_found', 0) == 0]
        llm_specific_failures = [t for t in failed_tests if t.get('is_llm_specific', False)]
        
        print(f"🔤 Keyword Match Failures: {len(keyword_failures)}")
        print(f"📭 No Results Failures: {len(no_results_failures)}")
        print(f"🤖 LLM-Specific Failures: {len(llm_specific_failures)}")
        
        # Most common failure reasons
        print(f"\n💡 RECOMMENDATIONS:")
        if keyword_failures:
            print("  1. Review expected keywords - some may not exist in the synthetic dataset")
        if llm_specific_failures:
            print("  2. LLM-specific queries may need dataset enhancement with financial/export data")
        if no_results_failures:
            print("  3. Consider broadening query terms or checking dataset coverage")
        
        print("  4. Failed tests help identify areas where the search system could be improved")
        print("  5. Consider adding synonyms or alternative terms for better matching")
    
    def generate_csv_report(self):
        """Generate CSV report with test results"""
        print("\n📊 Generating CSV Report...")
        
        csv_file = Path(__file__).parent / 'company_100_def_test_report.csv'
        
        # Handle permission errors by trying alternative filenames
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    csv_file = Path(__file__).parent / f'company_100_def_test_report_{attempt}.csv'
                
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow([
                        'Test Case',
                        'Query',
                        'Pass/Fail',
                        'Time Taken (seconds)',
                        'LLM Specific',
                        'Companies Found',
                        'Confidence',
                        'Strategy',
                        'Category',
                        'Top Results'
                    ])
                    
                    # Write test results
                    for test in self.test_results:
                        # Determine category
                        test_name_lower = test['test_name'].lower()
                        if 'certification' in test_name_lower:
                            category = 'Certification'
                        elif any(loc in test_name_lower for loc in ['location', 'karnataka', 'bengaluru', 'hyderabad', 'goa']):
                            category = 'Location'
                        elif any(combo in test_name_lower for combo in ['radar technologies', 'certified aerospace', 'large manufacturing', 'electronics companies', 'composite material', 'software companies']):
                            category = 'Combination'
                        else:
                            category = 'LLM Scenario'
                        
                        # Format top results for CSV
                        top_results = '; '.join(test.get('top_companies', [])[:3])  # Top 3 companies
                        
                        writer.writerow([
                            test['test_name'],
                            test['query'],
                            'PASS' if test.get('passed', False) else 'FAIL',
                            f"{test.get('processing_time', 0):.3f}",
                            'YES' if test.get('is_llm_specific', False) else 'NO',
                            test.get('companies_found', 0),
                            f"{test.get('confidence', 0):.3f}",
                            test.get('strategy', 'N/A'),
                            category,
                            top_results
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
        llm_specific_tests = sum(1 for test in self.test_results if test.get('is_llm_specific', False))
        avg_time = sum(test.get('processing_time', 0) for test in self.test_results) / total_tests if total_tests > 0 else 0
        
        print(f"\n📈 CSV Report Summary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  LLM Specific: {llm_specific_tests}")
        print(f"  Average Time: {avg_time:.3f} seconds")

def main():
    """Main test execution"""
    try:
        test_suite = CompanyTFIDFTestSuite()
        test_suite.run_all_tests()
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

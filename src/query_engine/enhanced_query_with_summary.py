#!/usr/bin/env python3
"""
Enhanced Query Execution with Integrated Summarization

This module integrates the enhanced embedding query decomposer with 
the summarization functionality to provide comprehensive query results
with detailed summaries.
"""

import pandas as pd
from typing import Dict, List, Optional
from src.query_engine.embedding_query_decomposer import execute_enhanced_embedding_query
from src.query_engine.summarize_result import summarize_locally_enhanced
from src.load_config import load_config, get_company_mapped_data_processed_data_store

def _extract_company_contact_info(company_info: Dict) -> Dict[str, str]:
    """Extract contact information from company data."""
    contact_details = {
        'email': '',
        'website': '',
        'phone': ''
    }
    
    try:
        # Extract from complete data
        complete_data = company_info.get('complete_data', {})
        
        # Basic info for contact details
        basic_info = complete_data.get('CompanyProfile.BasicInfo', {}).get('data', [])
        if basic_info:
            basic = basic_info[0]
            contact_details['email'] = basic.get('CompanyMaster.EmailId', '') or basic.get('CompanyMaster.POC_Email', '')
            contact_details['website'] = basic.get('CompanyMaster.Website', '')
            contact_details['phone'] = basic.get('CompanyMaster.Phone', '')
        
        # Contact info section
        contact_info = complete_data.get('CompanyProfile.ContactInfo', {}).get('data', [])
        if contact_info:
            contact = contact_info[0]
            # Use contact info if basic info doesn't have these details
            if not contact_details['email']:
                contact_details['email'] = contact.get('CompanyMaster.EmailId', '') or contact.get('CompanyMaster.POC_Email', '')
            if not contact_details['website']:
                contact_details['website'] = contact.get('CompanyMaster.Website', '')
            if not contact_details['phone']:
                contact_details['phone'] = contact.get('CompanyMaster.Phone', '')
                
    except Exception:
        pass
    
    return contact_details

def _convert_results_to_dataframe(result: Dict, logger) -> pd.DataFrame:
    """Convert query execution results to DataFrame format for summarization."""
    try:
        # Extract companies from results
        results_data = result.get('results', {})
        companies = results_data.get('companies', [])
        
        if not companies:
            logger.info("No companies found in results to convert to DataFrame")
            return pd.DataFrame()
        
        # Convert to DataFrame format expected by summarize_result.py
        df_data = []
        for i, company in enumerate(companies):
            # Extract CompanyRefNo - this is crucial for loading detailed company info
            company_ref_no = company.get('company_ref_no', company.get('id', ''))
            
            # If no CompanyRefNo, try to extract from company_name or create one
            if not company_ref_no:
                company_name = company.get('company_name', company.get('name', ''))
                if 'Company_' in company_name:
                    # Extract number from Company_036 format
                    import re
                    match = re.search(r'Company_(\d+)', company_name)
                    if match:
                        company_ref_no = f"CMP{match.group(1).zfill(3)}"
                else:
                    company_ref_no = f'CMP{i+1:03d}'
            
            # Map company data to expected DataFrame columns
            row = {
                'CompanyRefNo': company_ref_no,
                'CompanyNumber': company.get('company_number', company_ref_no),
                'CompanyName': company.get('company_name', company.get('name', '')),
                'Domain': company.get('domain', company.get('industry', company.get('core_expertise', ''))),
                '_score': company.get('score', company.get('tfidf_score', 0.0)),
                '_doc': company.get('reason', company.get('core_expertise', company.get('description', ''))),
                '__store': company.get('domain', company.get('industry', '')),
                '__value': company.get('company_name', company.get('name', ''))
            }
            df_data.append(row)
            
            logger.debug(f"Mapped company: {row['CompanyName']} -> {row['CompanyRefNo']}")
        
        df = pd.DataFrame(df_data)
        logger.info(f"Converted {len(df)} companies to DataFrame for summarization")
        logger.info(f"CompanyRefNos: {df['CompanyRefNo'].tolist()}")
        return df
        
    except ImportError:
        logger.error("pandas not available for DataFrame conversion")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to convert results to DataFrame: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def execute_enhanced_query_with_summary(user_query: str, config=None, logger=None, enable_llm_validation=True) -> Dict:
    """
    Execute enhanced embedding query with integrated summarization.
    
    This function:
    1. Executes the enhanced embedding query decomposition
    2. Generates a comprehensive summary of results
    3. Returns both detailed results and human-readable summary
    
    Args:
        user_query: The user's natural language query
        config: Configuration dictionary (optional, will load if not provided)
        logger: Logger instance (optional, will create if not provided)
        enable_llm_validation: Enable LLM validation for low-confidence queries
    
    Returns:
        Dictionary with search results, execution details, and enhanced summary
    """
    # Load config and logger if not provided
    if config is None or logger is None:
        (config, logger) = load_config()
    
    logger.info("=" * 80)
    logger.info("🚀 ENHANCED QUERY EXECUTION WITH SUMMARIZATION")
    logger.info("=" * 80)
    
    # Step 1: Execute enhanced embedding query
    logger.info(f"📝 USER QUERY: '{user_query}'")
    logger.info("🔍 Executing enhanced embedding query...")
    
    try:
        # Execute the enhanced query
        result = execute_enhanced_embedding_query(
            user_query=user_query,
            config=config,
            logger=logger,
            enable_llm_validation=enable_llm_validation
        )
        
        # Step 2: Generate enhanced summary
        logger.info("\n" + "📝 STEP: RESULT SUMMARIZATION")
        logger.info("-" * 50)
        
        try:
            # Get the base path for company summaries
            base_path = get_company_mapped_data_processed_data_store(config)
            logger.info(f"📁 Using company data path: {base_path}")
            
            # Convert results to DataFrame format for summarization
            summary_df = _convert_results_to_dataframe(result, logger)
            
            if not summary_df.empty:
                # Generate company contact list instead of enhanced summary
                from src.query_engine.summarize_result import load_company_summary
                
                company_list = []
                for _, row in summary_df.iterrows():
                    company_ref_no = row.get('CompanyRefNo', '')
                    company_name = row.get('CompanyName', '')
                    
                    # Load detailed company information
                    company_info = load_company_summary(company_ref_no, str(base_path)) if company_ref_no else None
                    
                    # Extract contact information
                    contact_info = _extract_company_contact_info(company_info) if company_info else {
                        'email': '', 'website': '', 'phone': ''
                    }
                    
                    company_entry = {
                        'company_ref_no': company_ref_no,
                        'company_name': company_name,
                        'email': contact_info['email'],
                        'website': contact_info['website'],
                        'phone': contact_info['phone']
                    }
                    company_list.append(company_entry)
                    
                    logger.debug(f"Added company: {company_name} ({company_ref_no}) - Email: {contact_info['email']}, Website: {contact_info['website']}")
                
                # Add company list to results instead of enhanced_summary
                result['company_contacts'] = company_list
                logger.info(f"✅ Company contact list generated with {len(company_list)} companies")
                
                # Log the company list for visibility
                logger.info("\n" + "📋 COMPANY CONTACT LIST:")
                logger.info("-" * 50)
                for i, company in enumerate(company_list[:10], 1):
                    logger.info(f"{i}. {company['company_name']} ({company['company_ref_no']})")
                    if company['email']:
                        logger.info(f"   Email: {company['email']}")
                    if company['website']:
                        logger.info(f"   Website: {company['website']}")
                    if company['phone']:
                        logger.info(f"   Phone: {company['phone']}")
                    logger.info("")
                if len(company_list) > 10:
                    logger.info(f"... and {len(company_list) - 10} more companies")
                    
            else:
                logger.warning("⚠️ No results to process")
                result['company_contacts'] = []
                
        except Exception as e:
            logger.error(f"❌ Summary generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            result['enhanced_summary'] = f"Summary generation failed: {str(e)}"
        
        # Step 3: Final logging
        logger.info("\n" + "📋 FINAL EXECUTION SUMMARY")
        logger.info("-" * 50)
        logger.info(f"✅ Query Processing Complete")
        logger.info(f"📊 Final Confidence: {result.get('confidence', 0):.2f}")
        logger.info(f"🤖 LLM Validation Used: {result.get('llm_validation_used', False)}")
        logger.info(f"🔍 Search Strategy: {result.get('strategy', 'Unknown')}")
        logger.info(f"📈 Total Results: {result.get('results', {}).get('companies_count', 0)} companies, {result.get('results', {}).get('products_count', 0)} products")
        logger.info(f"📝 Company Contacts: {'Generated' if 'company_contacts' in result else 'Failed'}")
        if 'company_contacts' in result:
            logger.info(f"📞 Contact Info Available: {len(result['company_contacts'])} companies")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Enhanced query execution failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'error': str(e),
            'query': user_query,
            'enhanced_summary': f"Query execution failed: {str(e)}"
        }

def test_enhanced_query_with_summary():
    """Test the enhanced query execution with summarization."""
    (config, logger) = load_config()
    
    logger.info("🧪 Testing Enhanced Query Execution with Summarization")
    logger.info("=" * 80)
    
    # Test queries with different complexity levels
    test_queries = [
        # Simple query
        #"list companies",
        
        # Medium complexity with filters
        #"small scale electrical companies",
        
        # Complex query with multiple entities
        "companies having ISO certification with testing facilities",
    ]
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n🔍 TEST {i}/{len(test_queries)}: '{query}'")
        logger.info("-" * 60)
        
        try:
            # Execute with summarization
            result = execute_enhanced_query_with_summary(
                user_query=query,
                config=config,
                logger=logger,
                enable_llm_validation=True
            )
            
            # Log test results
            logger.info(f"\n🎯 TEST {i} RESULTS:")
            logger.info(f"   - Strategy: {result.get('strategy', 'Unknown')}")
            logger.info(f"   - Components: {result.get('components', 0)}")
            logger.info(f"   - Dependencies: {result.get('dependencies', 0)}")
            logger.info(f"   - Confidence: {result.get('confidence', 0):.2f}")
            logger.info(f"   - LLM Used: {result.get('llm_validation_used', False)}")
            
            if result.get('results'):
                results = result['results']
                logger.info(f"   - Companies Found: {results.get('companies_count', 0)}")
                logger.info(f"   - Products Found: {results.get('products_count', 0)}")
            
            # Show enhanced summary
            if result.get('enhanced_summary'):
                logger.info(f"\n📋 ENHANCED SUMMARY FOR TEST {i}:")
                logger.info("-" * 40)
                logger.info(result['enhanced_summary'])
            
            logger.info(f"✅ TEST {i} COMPLETED SUCCESSFULLY")
            
        except Exception as e:
            logger.error(f"❌ TEST {i} FAILED: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    logger.info(f"\n{'='*80}")
    logger.info("🏁 ALL TESTS COMPLETED")
    logger.info(f"{'='*80}")

if __name__ == "__main__":
    test_enhanced_query_with_summary()

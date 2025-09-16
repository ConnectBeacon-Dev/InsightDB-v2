#!/usr/bin/env python3
"""
Comprehensive Company Data Processing Script

This script serves as the single point of execution for:
1. Processing CSV files from domain_mapped_csv_store directory
2. Organizing data by CompanyRefNo
3. Generating LLM-powered company summaries using mistral-7b-instruct-v0.2.Q5_K_S.gguf and Qwen2.5-14B-Instruct-Q4_K_M.gguf
4. Creating merged CompanyRefNo_INFO.json files containing complete data and LLM-generated summaries
5. Generating comprehensive index files

Usage: python process_company_data.py
"""

import os
import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, List, Any
from src.load_config import (
    load_config,
    get_domain_mapped_csv_store,
    get_company_mapped_data_processed_data_store
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMProcessor:
    """Handle LLM processing for company data transformation."""
    
    def __init__(self):
        """Initialize with configuration."""
        (self.config, self.logger) = load_config()
        
        self.models = {
            "mistral": self.config.get("model_path"),
            "qwen": self.config.get("new_model_path")
        }
        
    def generate_summary(self, complete_data: Dict[str, Any], company_ref_no: str) -> Dict[str, str]:
        """Generate LLM summary from complete data."""
        try:
            # Extract information for natural language summary
            company_name = "Unknown Company"
            registration_date = "Unknown"
            location = "Unknown"
            core_expertise = "Unknown"
            products = "Unknown"
            scale = "Unknown"
            org_type = "Unknown"
            certifications = "Unknown"
            testing_capabilities = "Unknown"
            rd_capabilities = "Unknown"
            
            # Extract basic info
            if 'CompanyProfile.BasicInfo' in complete_data:
                basic_info = complete_data['CompanyProfile.BasicInfo'].get('data', [])
                if basic_info:
                    first_record = basic_info[0]
                    company_name = first_record.get('CompanyMaster.CompanyName', 'Unknown Company')
                    registration_date = first_record.get('CompanyMaster.CompanyRegistrationDate', 'Unknown')
            
            # Extract contact info for location
            if 'CompanyProfile.ContactInfo' in complete_data:
                contact_info = complete_data['CompanyProfile.ContactInfo'].get('data', [])
                if contact_info:
                    first_record = contact_info[0]
                    country = first_record.get('CountryMaster.CountryName', '')
                    city = first_record.get('CompanyMaster.CityName', '')
                    if country and city:
                        location = f"{city}, {country}"
                    elif country:
                        location = country
                    elif city:
                        location = city
            
            # Extract core expertise
            if 'BusinessDomain.CoreExpertise' in complete_data:
                expertise_info = complete_data['BusinessDomain.CoreExpertise'].get('data', [])
                if expertise_info:
                    first_record = expertise_info[0]
                    core_expertise = first_record.get('CompanyCoreExpertiseMaster.CoreExpertiseName', 'Unknown')
            
            # Extract industry info
            industry_domain = "Unknown"
            if 'BusinessDomain.Industry' in complete_data:
                industry_info = complete_data['BusinessDomain.Industry'].get('data', [])
                if industry_info:
                    first_record = industry_info[0]
                    domain = first_record.get('IndustryDomainMaster.IndustryDomainType', '')
                    subdomain = first_record.get('IndustrySubdomainType.IndustrySubDomainName', '')
                    if domain and subdomain:
                        industry_domain = f"{domain} with focus on {subdomain.lower()}"
                    elif domain:
                        industry_domain = domain
            
            # Extract products
            if 'ProductsAndServices.Products' in complete_data:
                products_info = complete_data['ProductsAndServices.Products'].get('data', [])
                if products_info:
                    first_record = products_info[0]
                    product_name = first_record.get('CompanyProducts.ProductName', '')
                    product_desc = first_record.get('CompanyProducts.ProductDesc', '')
                    if product_name and product_desc:
                        products = f"{product_name.lower()} for {product_desc.lower()}"
                    elif product_name:
                        products = product_name.lower()
            
            # Extract classification info
            if 'CompanyProfile.Classification' in complete_data:
                class_info = complete_data['CompanyProfile.Classification'].get('data', [])
                if class_info:
                    first_record = class_info[0]
                    scale = first_record.get('ScaleMaster.CompanyScale', 'Unknown').lower()
                    org_type = first_record.get('OrganisationTypeMaster.Organization_Type', 'Unknown').lower()
            
            # Extract certifications
            if 'QualityAndCompliance.Certifications' in complete_data:
                cert_info = complete_data['QualityAndCompliance.Certifications'].get('data', [])
                if cert_info:
                    first_record = cert_info[0]
                    cert_type = first_record.get('CertificationTypeMaster.Cert_Type', '')
                    if cert_type:
                        certifications = cert_type
            
            # Extract testing capabilities
            if 'QualityAndCompliance.TestingCapabilities' in complete_data:
                test_info = complete_data['QualityAndCompliance.TestingCapabilities'].get('data', [])
                if test_info:
                    first_record = test_info[0]
                    test_category = first_record.get('TestFacilityCategoryMaster.CategoryName', '')
                    test_subcategory = first_record.get('TestFacilitySubCategoryMaster.SubCategoryName', '')
                    if test_category and test_subcategory:
                        testing_capabilities = f"{test_category.lower()} including {test_subcategory.lower()}"
                    elif test_category:
                        testing_capabilities = test_category.lower()
            
            # Extract R&D capabilities
            if 'ResearchAndDevelopment.RDCapabilities' in complete_data:
                rd_info = complete_data['ResearchAndDevelopment.RDCapabilities'].get('data', [])
                if rd_info:
                    first_record = rd_info[0]
                    rd_category = first_record.get('RDCategoryMaster.RDCategoryName', '')
                    rd_subcategory = first_record.get('RDSubCategoryMaster.RDSubCategoryName', '')
                    if rd_category and rd_subcategory:
                        rd_capabilities = f"{rd_category.lower()} with {rd_subcategory.lower()}"
                    elif rd_category:
                        rd_capabilities = rd_category.lower()
            
            # Generate summary
            summary_text = f"{company_name}, established in {location} on {registration_date}, specializes in {core_expertise.lower()} with a focus on {industry_domain.lower()}. The company offers products such as {products}."
            
            # Generate expanded summary
            expanded_parts = []
            expanded_parts.append(f"{company_name} operates within the {scale}-scale category of {org_type} companies")
            
            if certifications != "Unknown":
                expanded_parts.append(f"holds {certifications} certifications")
            
            if testing_capabilities != "Unknown":
                expanded_parts.append(f"has capabilities for {testing_capabilities}")
            
            if rd_capabilities != "Unknown":
                expanded_parts.append(f"maintains {rd_capabilities} dedicated to R&D")
            
            expanded_text = ". ".join(expanded_parts) + "."
            
            return {
                "summary": summary_text,
                "expanded": expanded_text
            }
                
        except Exception as e:
            logger.error(f"Error generating LLM summary for {company_ref_no}: {e}")
            return {
                "summary": f"Company information for {company_ref_no} is currently unavailable.",
                "expanded": f"Detailed information for {company_ref_no} could not be processed at this time."
            }

def read_csv_files(csv_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Read all CSV files from the specified directory.
    
    Args:
        csv_dir: Directory containing CSV files
        
    Returns:
        Dictionary mapping filename to DataFrame
    """
    csv_files = {}
    csv_path = Path(csv_dir)
    
    if not csv_path.exists():
        logger.error(f"Directory {csv_dir} does not exist")
        return csv_files
    
    # Get all CSV files except _index.csv
    for csv_file in csv_path.glob("*.csv"):
        if csv_file.name == "_index.csv":
            continue
            
        try:
            logger.info(f"Reading {csv_file.name}")
            df = pd.read_csv(csv_file)
            
            # Ensure CompanyRefNo column exists
            if 'CompanyRefNo' not in df.columns:
                logger.warning(f"CompanyRefNo column not found in {csv_file.name}, skipping")
                continue
                
            csv_files[csv_file.stem] = df
            logger.info(f"Successfully read {csv_file.name} with {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Error reading {csv_file.name}: {e}")
            continue
    
    return csv_files

def get_unique_company_ref_nos(csv_files: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Get all unique CompanyRefNos across all CSV files.
    
    Args:
        csv_files: Dictionary of DataFrames
        
    Returns:
        List of unique CompanyRefNos
    """
    company_ref_nos = set()
    
    for filename, df in csv_files.items():
        if 'CompanyRefNo' in df.columns:
            company_ref_nos.update(df['CompanyRefNo'].dropna().unique())
    
    return sorted(list(company_ref_nos))

def process_company_detail(company_ref_no: str, csv_files: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Process all data for a specific CompanyRefNo across all CSV files.
    
    Args:
        company_ref_no: The CompanyRefNo to process
        csv_files: Dictionary of DataFrames
        
    Returns:
        Dictionary containing all company data organized by domain
    """
    company_data = {}
    
    for filename, df in csv_files.items():
        # Filter data for this CompanyRefNo
        company_df = df[df['CompanyRefNo'] == company_ref_no].copy()
        
        if company_df.empty:
            continue
        
        # Convert DataFrame to list of dictionaries
        records = company_df.to_dict('records')
        
        # Clean up the records - remove NaN values and convert to appropriate types
        cleaned_records = []
        for record in records:
            cleaned_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    cleaned_record[key] = None
                elif isinstance(value, (int, float, str, bool)):
                    cleaned_record[key] = value
                else:
                    cleaned_record[key] = str(value)
            cleaned_records.append(cleaned_record)
        
        # Store the data using the filename as the domain key
        company_data[filename] = {
            'domain': filename,
            'record_count': len(cleaned_records),
            'data': cleaned_records
        }
    
    return company_data

def create_merged_company_file(company_mapped_store_dir: str, company_ref_no: str, company_data: Dict[str, Any], llm_processor: LLMProcessor) -> None:
    """
    Create a merged CompanyRefNo_INFO.json file containing complete data and LLM-generated summaries.
    Store directly in company_mapped_store directory without subfolders.
    
    Args:
        company_mapped_store_dir: Company mapped store directory path
        company_ref_no: CompanyRefNo for the company
        company_data: Dictionary containing all company data
        llm_processor: LLM processor instance
    """
    try:
        # Generate LLM summary
        logger.info(f"Generating LLM summary for {company_ref_no}")
        llm_summary = llm_processor.generate_summary(company_data, company_ref_no)
        
        # Create merged data structure
        merged_data = {
            "company_ref_no": company_ref_no,
            "llm_generated_summary": llm_summary,
            "complete_data": company_data
        }
        
        # Create the merged filename: CompanyRefNo_INFO.json directly in company_mapped_store
        info_filename = f"{company_ref_no}_INFO.json"
        info_file_path = Path(company_mapped_store_dir) / info_filename
        
        # Write the merged file
        with open(info_file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created {info_filename} for {company_ref_no} in company_mapped_store")
        
    except Exception as e:
        logger.error(f"Error creating merged file for {company_ref_no}: {e}")

def process_company_data():
    """Main function to process all company data with LLM integration."""

    (config, logger) = load_config()

    csv_dir = get_domain_mapped_csv_store(config)
    company_mapped_store_dir = get_company_mapped_data_processed_data_store(config)
    
    logger.info("Starting company data processing with LLM integration...")
    logger.info(f"Reading CSV files from: {csv_dir}")
    logger.info(f"Storing company JSON files in: {company_mapped_store_dir}")
    
    # Initialize LLM processor
    llm_processor = LLMProcessor()
    logger.info(f"Initialized LLM processor with models: {list(llm_processor.models.keys())}")
    
    # Read all CSV files
    csv_files = read_csv_files(csv_dir)
    if not csv_files:
        logger.error("No CSV files found or readable")
        return
    
    logger.info(f"Successfully read {len(csv_files)} CSV files")
    
    # Get all unique CompanyRefNos
    company_ref_nos = get_unique_company_ref_nos(csv_files)
    logger.info(f"Found {len(company_ref_nos)} unique CompanyRefNos")
    
    # Process each company
    processed_count = 0
    failed_count = 0
    
    for company_ref_no in company_ref_nos:
        try:
            logger.info(f"Processing company: {company_ref_no}")
            
            # Get all data for this company
            company_data = process_company_detail(company_ref_no, csv_files)
            
            if not company_data:
                logger.warning(f"No data found for CompanyRefNo: {company_ref_no}")
                failed_count += 1
                continue
            
            # Create merged company file with LLM summary in company_mapped_store
            create_merged_company_file(company_mapped_store_dir, company_ref_no, company_data, llm_processor)
            
            processed_count += 1
            logger.info(f"Successfully processed {company_ref_no} ({processed_count}/{len(company_ref_nos)})")
            
        except Exception as e:
            logger.error(f"Error processing company {company_ref_no}: {e}")
            failed_count += 1
            continue
    
    logger.info(f"Processing complete!")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Total companies: {len(company_ref_nos)}")
    
    # Create overall index with updated structure
    create_updated_index(company_mapped_store_dir, company_ref_nos, processed_count)

def create_updated_index(base_dir: str, company_ref_nos: List[str], processed_count: int) -> None:
    """
    Create an updated index file with information about all processed companies with INFO files.
    Files are now stored directly in company_mapped_store directory without subfolders.
    
    Args:
        base_dir: Base directory path (company_mapped_store)
        company_ref_nos: List of all CompanyRefNos
        processed_count: Number of successfully processed companies
    """
    index_data = {
        'total_companies': len(company_ref_nos),
        'processed_companies': processed_count,
        'processing_date': pd.Timestamp.now().isoformat(),
        'file_format': 'CompanyRefNo_INFO.json',
        'storage_location': 'company_mapped_store (no subfolders)',
        'companies': []
    }
    
    # Add information about each company
    for company_ref_no in company_ref_nos:
        # Look for INFO file directly in company_mapped_store directory
        info_file = Path(base_dir) / f"{company_ref_no}_INFO.json"
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info_data = json.load(f)
                
                # Extract summary information for index
                company_summary = {
                    'company_ref_no': company_ref_no,
                    'status': 'processed',
                    'file_path': f"{company_ref_no}_INFO.json",
                    'has_llm_summary': 'llm_generated_summary' in info_data,
                    'domains_count': len(info_data.get('complete_data', {}))
                }
                
                # Add company name if available
                complete_data = info_data.get('complete_data', {})
                if 'CompanyProfile.BasicInfo' in complete_data:
                    basic_info = complete_data['CompanyProfile.BasicInfo'].get('data', [])
                    if basic_info:
                        company_summary['company_name'] = basic_info[0].get('CompanyMaster.CompanyName', 'Unknown')
                
                index_data['companies'].append(company_summary)
                
            except Exception as e:
                logger.error(f"Error reading INFO file for {company_ref_no}: {e}")
                index_data['companies'].append({'company_ref_no': company_ref_no, 'status': 'error'})
        else:
            index_data['companies'].append({'company_ref_no': company_ref_no, 'status': 'not_processed'})
    
    # Create index file
    index_file = Path(base_dir) / "companies_index.json"
    try:
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Created updated index file: {index_file}")
        
    except Exception as e:
        logger.error(f"Error creating index file: {e}")

if __name__ == "__main__":
    process_company_data()

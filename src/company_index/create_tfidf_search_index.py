#!/usr/bin/env python3
"""
TF-IDF Search Index Creator for Company Data

This script:
1. Merges all company JSON files from company_mapped_store directory
2. Creates a TF-IDF vectorizer for text-based search
3. Builds search indices for efficient NLP query processing
4. Saves the vectorizer and search index for later use

Usage: python create_tfidf_search_index.py
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.load_config import (
    load_config,
    get_company_mapped_data_processed_data_store,
    get_company_mapped_data_tfidf_search_store
)

class CompanyTfidfSearchIndex:
    """Create and manage TF-IDF search index for company data."""
    
    def __init__(self, config, logger):
        """Initialize with configuration."""
        self.config = config
        self.logger = logger
        self.company_mapped_store_dir = get_company_mapped_data_processed_data_store(self.config)
        self.vectorizer = None
        self.tfidf_matrix = None
        self.company_documents = []
        self.company_metadata = []
        
    def load_all_company_files(self) -> List[Dict[str, Any]]:
        """Load all company JSON files from company_mapped_store directory."""
        company_files = []
        company_path = Path(self.company_mapped_store_dir)
        
        if not company_path.exists():
            self.logger.error(f"Company mapped store directory does not exist: {company_path}")
            return company_files
        
        # Find all *_INFO.json files
        info_files = list(company_path.glob("*_INFO.json"))
        self.logger.info(f"Found {len(info_files)} company INFO files")
        
        for info_file in info_files:
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    company_data = json.load(f)
                    company_files.append(company_data)
                    self.logger.debug(f"Loaded {info_file.name}")
            except Exception as e:
                self.logger.error(f"Error loading {info_file.name}: {e}")
                continue
        
        self.logger.info(f"Successfully loaded {len(company_files)} company files")
        return company_files
    
    def extract_searchable_text(self, company_data: Dict[str, Any]) -> str:
        """Extract all searchable text from company data with keyword enrichment."""
        searchable_texts = []
        
        # Extract company reference number
        company_ref_no = company_data.get('company_ref_no', '')
        if company_ref_no:
            searchable_texts.append(company_ref_no)
        
        # Extract LLM generated summaries
        llm_summary = company_data.get('llm_generated_summary', {})
        if llm_summary:
            summary = llm_summary.get('summary', '')
            expanded = llm_summary.get('expanded', '')
            if summary:
                searchable_texts.append(summary)
            if expanded:
                searchable_texts.append(expanded)
        
        # Extract complete data
        complete_data = company_data.get('complete_data', {})
        
        # Extract from each domain with contextual keyword enrichment
        for domain_name, domain_data in complete_data.items():
            if isinstance(domain_data, dict) and 'data' in domain_data:
                records = domain_data['data']
                if isinstance(records, list):
                    for record in records:
                        if isinstance(record, dict):
                            # Extract text from all string fields
                            for key, value in record.items():
                                if isinstance(value, str) and value.strip():
                                    # Skip null-like values
                                    if value.lower() not in ['null', 'none', 'n/a', '']:
                                        searchable_texts.append(value.strip())
        
        # Add contextual keywords based on company data
        contextual_keywords = self.generate_contextual_keywords(complete_data)
        searchable_texts.extend(contextual_keywords)
        
        # Join all text with spaces
        full_text = ' '.join(searchable_texts)
        return full_text
    
    def generate_contextual_keywords(self, complete_data: Dict[str, Any]) -> List[str]:
        """Generate contextual keywords based on company data to improve search relevance."""
        keywords = []
        
        # Industry-specific keyword mapping
        industry_keywords = {
            'manufacturing': ['production', 'manufacturing', 'factory', 'assembly', 'fabrication', 'industrial'],
            'software': ['development', 'programming', 'coding', 'application', 'system', 'technology'],
            'services': ['services', 'consulting', 'support', 'maintenance', 'operation', 'management'],
            'supply': ['supply', 'logistics', 'distribution', 'procurement', 'sourcing', 'vendor'],
            'aerospace': ['aerospace', 'aviation', 'aircraft', 'defense', 'military', 'flight'],
            'automotive': ['automotive', 'vehicle', 'car', 'automobile', 'transport', 'mobility'],
            'pharmaceutical': ['pharmaceutical', 'medicine', 'drug', 'healthcare', 'medical', 'therapy'],
            'textile': ['textile', 'fabric', 'garment', 'clothing', 'apparel', 'fashion'],
            'food': ['food', 'beverage', 'nutrition', 'processing', 'packaging', 'agriculture'],
            'energy': ['energy', 'power', 'renewable', 'solar', 'wind', 'electricity'],
            'chemical': ['chemical', 'petrochemical', 'polymer', 'material', 'compound', 'synthesis'],
            'electronics': ['electronics', 'semiconductor', 'circuit', 'component', 'device', 'hardware']
        }
        
        # Capability-based keyword mapping
        capability_keywords = {
            'research': ['research', 'development', 'innovation', 'R&D', 'laboratory', 'testing'],
            'quality': ['quality', 'certification', 'ISO', 'compliance', 'standard', 'assurance'],
            'export': ['export', 'international', 'global', 'overseas', 'foreign', 'trade'],
            'testing': ['testing', 'validation', 'verification', 'inspection', 'analysis', 'evaluation'],
            'design': ['design', 'engineering', 'prototype', 'modeling', 'architecture', 'planning']
        }
        
        # Scale-based keywords
        scale_keywords = {
            'small': ['small', 'startup', 'SME', 'micro', 'emerging', 'boutique'],
            'medium': ['medium', 'mid-size', 'growing', 'established', 'regional'],
            'large': ['large', 'enterprise', 'corporation', 'multinational', 'major', 'leading']
        }
        
        # Analyze company data and add relevant keywords
        
        # Check industry domain
        if 'BusinessDomain.Industry' in complete_data:
            industry_info = complete_data['BusinessDomain.Industry'].get('data', [])
            for record in industry_info:
                if isinstance(record, dict):
                    domain = record.get('IndustryDomainMaster.IndustryDomainType', '') or ''
                    subdomain = record.get('IndustrySubdomainType.IndustrySubDomainName', '') or ''
                    
                    # Safely convert to lowercase
                    domain = domain.lower() if domain else ''
                    subdomain = subdomain.lower() if subdomain else ''
                    
                    # Add industry-specific keywords
                    for industry_key, industry_kw in industry_keywords.items():
                        if industry_key in domain or industry_key in subdomain:
                            keywords.extend(industry_kw)
        
        # Check core expertise
        if 'BusinessDomain.CoreExpertise' in complete_data:
            expertise_info = complete_data['BusinessDomain.CoreExpertise'].get('data', [])
            for record in expertise_info:
                if isinstance(record, dict):
                    expertise = record.get('CompanyCoreExpertiseMaster.CoreExpertiseName', '') or ''
                    
                    # Safely convert to lowercase
                    expertise = expertise.lower() if expertise else ''
                    
                    # Add capability-based keywords
                    for cap_key, cap_kw in capability_keywords.items():
                        if cap_key in expertise:
                            keywords.extend(cap_kw)
        
        # Check products and services
        if 'ProductsAndServices.Products' in complete_data:
            products_info = complete_data['ProductsAndServices.Products'].get('data', [])
            for record in products_info:
                if isinstance(record, dict):
                    product_name = record.get('CompanyProducts.ProductName', '') or ''
                    product_desc = record.get('CompanyProducts.ProductDesc', '') or ''
                    
                    # Safely convert to lowercase
                    product_name = product_name.lower() if product_name else ''
                    product_desc = product_desc.lower() if product_desc else ''
                    
                    # Add production-related keywords for manufacturing companies
                    if any(word in product_name + ' ' + product_desc for word in ['component', 'part', 'equipment', 'machinery']):
                        keywords.extend(['production', 'manufacturing', 'assembly', 'fabrication'])
                    
                    # Add service-related keywords
                    if any(word in product_name + ' ' + product_desc for word in ['service', 'support', 'maintenance', 'consulting']):
                        keywords.extend(['services', 'operation', 'support', 'maintenance'])
        
        # Check company scale
        if 'CompanyProfile.Classification' in complete_data:
            class_info = complete_data['CompanyProfile.Classification'].get('data', [])
            for record in class_info:
                if isinstance(record, dict):
                    scale = record.get('ScaleMaster.CompanyScale', '') or ''
                    
                    # Safely convert to lowercase
                    scale = scale.lower() if scale else ''
                    
                    # Add scale-based keywords
                    for scale_key, scale_kw in scale_keywords.items():
                        if scale_key in scale:
                            keywords.extend(scale_kw)
        
        # Check R&D capabilities
        if 'ResearchAndDevelopment.RDCapabilities' in complete_data:
            rd_info = complete_data['ResearchAndDevelopment.RDCapabilities'].get('data', [])
            if rd_info:
                keywords.extend(['research', 'development', 'innovation', 'R&D', 'technology'])
        
        # Check testing capabilities
        if 'QualityAndCompliance.TestingCapabilities' in complete_data:
            test_info = complete_data['QualityAndCompliance.TestingCapabilities'].get('data', [])
            if test_info:
                keywords.extend(['testing', 'quality', 'compliance', 'certification', 'validation'])
        
        # Check certifications
        if 'QualityAndCompliance.Certifications' in complete_data:
            cert_info = complete_data['QualityAndCompliance.Certifications'].get('data', [])
            if cert_info:
                keywords.extend(['certified', 'quality', 'standard', 'compliance', 'ISO'])
        
        # Check export information
        if 'ProductsAndServices.ExportInfo' in complete_data:
            export_info = complete_data['ProductsAndServices.ExportInfo'].get('data', [])
            if export_info:
                keywords.extend(['export', 'international', 'global', 'overseas', 'trade'])
        
        # Always add core business operation keywords
        keywords.extend(['production', 'operation', 'supply', 'services'])
        
        # Remove duplicates and return
        return list(set(keywords))
    
    def extract_company_metadata(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metadata for each company."""
        metadata = {
            'company_ref_no': company_data.get('company_ref_no', ''),
            'company_name': 'Unknown Company',
            'location': 'Unknown',
            'core_expertise': 'Unknown',
            'industry_domain': 'Unknown',
            'scale': 'Unknown',
            'organization_type': 'Unknown'
        }
        
        complete_data = company_data.get('complete_data', {})
        
        # Extract basic company information
        if 'CompanyProfile.BasicInfo' in complete_data:
            basic_info = complete_data['CompanyProfile.BasicInfo'].get('data', [])
            if basic_info:
                first_record = basic_info[0]
                metadata['company_name'] = first_record.get('CompanyMaster.CompanyName', 'Unknown Company')
        
        # Extract contact info for location
        if 'CompanyProfile.ContactInfo' in complete_data:
            contact_info = complete_data['CompanyProfile.ContactInfo'].get('data', [])
            if contact_info:
                first_record = contact_info[0]
                country = first_record.get('CountryMaster.CountryName', '')
                city = first_record.get('CompanyMaster.CityName', '')
                if country and city:
                    metadata['location'] = f"{city}, {country}"
                elif country:
                    metadata['location'] = country
                elif city:
                    metadata['location'] = city
        
        # Extract core expertise
        if 'BusinessDomain.CoreExpertise' in complete_data:
            expertise_info = complete_data['BusinessDomain.CoreExpertise'].get('data', [])
            if expertise_info:
                first_record = expertise_info[0]
                metadata['core_expertise'] = first_record.get('CompanyCoreExpertiseMaster.CoreExpertiseName', 'Unknown')
        
        # Extract industry domain
        if 'BusinessDomain.Industry' in complete_data:
            industry_info = complete_data['BusinessDomain.Industry'].get('data', [])
            if industry_info:
                first_record = industry_info[0]
                domain = first_record.get('IndustryDomainMaster.IndustryDomainType', '')
                subdomain = first_record.get('IndustrySubdomainType.IndustrySubDomainName', '')
                if domain and subdomain:
                    metadata['industry_domain'] = f"{domain} - {subdomain}"
                elif domain:
                    metadata['industry_domain'] = domain
        
        # Extract classification info
        if 'CompanyProfile.Classification' in complete_data:
            class_info = complete_data['CompanyProfile.Classification'].get('data', [])
            if class_info:
                first_record = class_info[0]
                metadata['scale'] = first_record.get('ScaleMaster.CompanyScale', 'Unknown')
                metadata['organization_type'] = first_record.get('OrganisationTypeMaster.Organization_Type', 'Unknown')
        
        # Add LLM summaries to metadata
        llm_summary = company_data.get('llm_generated_summary', {})
        metadata['summary'] = llm_summary.get('summary', '')
        metadata['expanded_summary'] = llm_summary.get('expanded', '')
        
        return metadata
    
    def create_tfidf_index(self, company_files: List[Dict[str, Any]]) -> None:
        """Create TF-IDF vectorizer and index from company data."""
        self.logger.info("Creating TF-IDF search index...")
        
        # Extract searchable text and metadata for each company
        for company_data in company_files:
            searchable_text = self.extract_searchable_text(company_data)
            metadata = self.extract_company_metadata(company_data)
            
            self.company_documents.append(searchable_text)
            self.company_metadata.append(metadata)
        
        self.logger.info(f"Extracted text from {len(self.company_documents)} companies")
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Limit vocabulary size
            stop_words='english',  # Remove common English stop words
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            min_df=1,  # Minimum document frequency
            max_df=1.0,  # Allow all terms (don't filter out common contextual keywords)
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Fit and transform documents
        self.tfidf_matrix = self.vectorizer.fit_transform(self.company_documents)
        
        self.logger.info(f"Created TF-IDF matrix with shape: {self.tfidf_matrix.shape}")
        self.logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def search_companies(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """Search companies using TF-IDF similarity."""
        if self.vectorizer is None or self.tfidf_matrix is None:
            self.logger.error("TF-IDF index not created. Call create_tfidf_index first.")
            return []
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero similarities
                results.append((self.company_metadata[idx], similarities[idx]))
        
        return results
    
    def save_index(self, index_file: str = None) -> None:
        """Save the TF-IDF index and metadata to files."""
        if index_file is None:
            # Get the tfidf search store directory and create the index file path
            tfidf_store_dir = get_company_mapped_data_tfidf_search_store(self.config)
            index_file = Path(tfidf_store_dir) / "tfidf_search_index.pkl"
        
        index_data = {
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'company_metadata': self.company_metadata,
            'company_documents': self.company_documents
        }
        
        try:
            # Ensure the directory exists
            Path(index_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(index_file, 'wb') as f:
                pickle.dump(index_data, f)
            self.logger.info(f"Saved TF-IDF search index to: {index_file}")
            
            # Also save metadata as JSON for easy inspection
            metadata_file = Path(self.company_mapped_store_dir) / "company_search_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.company_metadata, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved company metadata to: {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
    
    def load_index(self, index_file: str = None) -> bool:
        """Load the TF-IDF index from file."""
        if index_file is None:
            index_file = Path(self.company_mapped_store_dir) / "tfidf_search_index.pkl"
        
        try:
            with open(index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            self.vectorizer = index_data['vectorizer']
            self.tfidf_matrix = index_data['tfidf_matrix']
            self.company_metadata = index_data['company_metadata']
            self.company_documents = index_data['company_documents']
            
            self.logger.info(f"Loaded TF-IDF search index from: {index_file}")
            self.logger.info(f"Index contains {len(self.company_metadata)} companies")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def create_merged_dataset(self) -> pd.DataFrame:
        """Create a merged dataset from all company data."""
        self.logger.info("Creating merged dataset...")
        
        all_records = []
        
        for metadata in self.company_metadata:
            # Create a flattened record for each company
            record = {
                'CompanyRefNo': metadata['company_ref_no'],
                'CompanyName': metadata['company_name'],
                'Location': metadata['location'],
                'CoreExpertise': metadata['core_expertise'],
                'IndustryDomain': metadata['industry_domain'],
                'Scale': metadata['scale'],
                'OrganizationType': metadata['organization_type'],
                'Summary': metadata['summary'],
                'ExpandedSummary': metadata['expanded_summary']
            }
            all_records.append(record)
        
        df = pd.DataFrame(all_records)
        
        # Save merged dataset
        merged_file = Path(self.company_mapped_store_dir) / "merged_company_dataset.csv"
        df.to_csv(merged_file, index=False, encoding='utf-8')
        self.logger.info(f"Saved merged dataset to: {merged_file}")
        
        return df

def create_search_index(config, logger):
    """Main function to create TF-IDF search index."""
    logger.info("Starting TF-IDF search index creation...")
    
    # Initialize search index creator
    search_index = CompanyTfidfSearchIndex(config, logger)
    
    # Load all company files
    company_files = search_index.load_all_company_files()
    if not company_files:
        logger.error("No company files found. Exiting.")
        return
    
    # Create TF-IDF index
    search_index.create_tfidf_index(company_files)
    
    # Save the index
    search_index.save_index()
    
    # Create merged dataset
    df = search_index.create_merged_dataset()
    
    logger.info(f"TF-IDF search index creation complete!")
    logger.info(f"Total companies indexed: {len(company_files)}")
    logger.info(f"Merged dataset shape: {df.shape}")
    
    # Test the search functionality
    test_queries = [
        "software development",
        "manufacturing company",
        "aerospace technology",
        "small scale enterprise",
        "research and development"
    ]
    
    logger.info("Testing search functionality...")
    for query in test_queries:
        results = search_index.search_companies(query, top_k=3)
        logger.info(f"Query: '{query}' - Found {len(results)} results")
        for i, (metadata, score) in enumerate(results):
            logger.info(f"  {i+1}. {metadata['company_name']} (Score: {score:.4f})")

if __name__ == "__main__":
    (config, logger) = load_config()
    create_search_index(config, logger)

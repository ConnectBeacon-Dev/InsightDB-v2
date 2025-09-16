#!/usr/bin/env python3
"""
Hybrid Search for Company Data

This script provides hybrid search functionality combining:
1. TF-IDF sparse retrieval for keyword matching
2. Dense vector search using FAISS for semantic similarity
3. Reciprocal Rank Fusion (RRF) to combine results
4. Filtering capabilities by scale, country, industry, etc.

Usage: python hybrid_search.py
"""

import os
import json
import pickle
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.load_config import (
    load_config,
    get_company_mapped_data_tfidf_search_store,
    get_company_mapped_data_dense_index_store
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompanyHybridSearch:
    """Hybrid search combining TF-IDF and dense vector search for company data."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.tfidf_store_dir = get_company_mapped_data_tfidf_search_store(config)
        self.dense_store_dir = get_company_mapped_data_dense_index_store(config)
        
        # Get model configuration
        self.embed_model_path = config.get('sentence_transformer_model_name', 'sentence-transformers/all-mpnet-base-v2')
        self.embed_model_from_net = config.get('sentence_transformer_model_from_net', 'sentence-transformers/all-mpnet-base-v2')
        self.top_k = config.get('top_k', 10)
        
        # Initialize components
        self.vectorizer = None
        self.tfidf_matrix = None
        self.tfidf_metadata = []
        self.faiss_index = None
        self.dense_metadata = []
        self.sentence_model = None
        
    def expand_query_terms(self, query: str, enable: bool = True) -> str:
        """Expand query terms with synonyms and related terms."""
        if not enable:
            return query
        
        # Simple query expansion - can be enhanced with more sophisticated methods
        expansions = {
            'software': ['software', 'application', 'system', 'technology', 'development'],
            'manufacturing': ['manufacturing', 'production', 'factory', 'assembly', 'fabrication'],
            'aerospace': ['aerospace', 'aviation', 'aircraft', 'defense', 'flight'],
            'automotive': ['automotive', 'vehicle', 'car', 'automobile', 'transport'],
            'small': ['small', 'micro', 'SME', 'startup', 'emerging'],
            'medium': ['medium', 'mid-size', 'growing', 'established'],
            'large': ['large', 'enterprise', 'corporation', 'multinational'],
            'research': ['research', 'R&D', 'development', 'innovation', 'laboratory'],
            'quality': ['quality', 'certification', 'ISO', 'compliance', 'standard'],
            'export': ['export', 'international', 'global', 'overseas', 'trade']
        }
        
        expanded_terms = []
        query_words = query.lower().split()
        
        for word in query_words:
            expanded_terms.append(word)
            for key, synonyms in expansions.items():
                if key in word or word in key:
                    expanded_terms.extend([s for s in synonyms if s not in expanded_terms])
        
        return ' '.join(expanded_terms)
    
    def load_tfidf_index(self):
        """Load TF-IDF index and metadata."""
        tfidf_path = Path(self.tfidf_store_dir) / "tfidf_search_index.pkl"
        
        if not tfidf_path.exists():
            logger.error(f"TF-IDF index not found: {tfidf_path}")
            raise FileNotFoundError(f"TF-IDF index not found: {tfidf_path}")
        
        logger.debug(f"Loading TF-IDF index from: {tfidf_path}")
        
        try:
            with open(tfidf_path, "rb") as f:
                data = pickle.load(f)
            
            self.vectorizer = data["vectorizer"]
            self.tfidf_matrix = data["tfidf_matrix"]
            self.tfidf_metadata = data["company_metadata"]
            
            logger.debug(f"Loaded TF-IDF index with {len(self.tfidf_metadata)} companies")
            
        except Exception as e:
            logger.error(f"Error loading TF-IDF index: {e}")
            raise
    
    def load_dense_index(self):
        """Load FAISS dense index and metadata."""
        dense_dir = Path(self.dense_store_dir)
        
        if not dense_dir.exists():
            logger.error(f"Dense index directory not found: {dense_dir}")
            raise FileNotFoundError(f"Dense index directory not found: {dense_dir}")
        
        logger.debug(f"Loading dense index from: {dense_dir}")
        
        try:
            # Load FAISS index
            faiss_path = dense_dir / "faiss.index"
            if not faiss_path.exists():
                raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
            
            self.faiss_index = faiss.read_index(str(faiss_path))
            
            # Load metadata
            metas_path = dense_dir / "metas.jsonl"
            if not metas_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metas_path}")
            
            self.dense_metadata = []
            with open(metas_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    self.dense_metadata.append(json.loads(line))
            
            logger.debug(f"Loaded dense index with {len(self.dense_metadata)} companies")
            
            # Verify alignment with TF-IDF
            if len(self.dense_metadata) != len(self.tfidf_metadata):
                logger.warning(f"Dense metadata ({len(self.dense_metadata)}) != TF-IDF metadata ({len(self.tfidf_metadata)})")
                logger.warning("Using TF-IDF metadata as primary source")
                self.dense_metadata = self.tfidf_metadata
            
        except Exception as e:
            logger.error(f"Error loading dense index: {e}")
            raise
    
    def load_sentence_model(self):
        """Load sentence transformer model for query encoding."""
        # Use local model if available, otherwise use from network
        if os.path.exists(self.embed_model_path):
            model_name = self.embed_model_path
            logger.debug(f"Using local model: {self.embed_model_path}")
        else:
            model_name = self.embed_model_from_net
            logger.debug(f"Using network model: {self.embed_model_from_net}")
        
        try:
            self.sentence_model = SentenceTransformer(model_name, device="cpu")
            logger.debug("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {e}")
            raise
    
    def reciprocal_rank_fusion(self, rank_lists: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
        """Combine multiple ranking lists using Reciprocal Rank Fusion."""
        scores = {}
        for rank_list in rank_lists:
            for rank, idx in enumerate(rank_list):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1.0)
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, str]) -> bool:
        """Apply filters to company metadata."""
        if filters.get('scale'):
            scale = (metadata.get("scale_canonical") or metadata.get("scale") or "").lower()
            if scale != filters['scale'].lower():
                return False
        
        if filters.get('country'):
            location = metadata.get("location", "").lower()
            if filters['country'].lower() not in location:
                return False
        
        if filters.get('industry'):
            industry = metadata.get("industry_domain", "").lower()
            if filters['industry'].lower() not in industry:
                return False
        
        return True
    
    def search(self, query: str, top_k: int = None, min_score: float = 0.0, 
               expand_query: bool = True, filters: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining TF-IDF and dense retrieval."""
        if top_k is None:
            top_k = self.top_k
        
        if filters is None:
            filters = {}
        
        logger.info(f"Performing hybrid search for query: '{query}'")
        logger.info(f"Parameters: top_k={top_k}, min_score={min_score}, expand_query={expand_query}")
        
        # Expand query if requested
        expanded_query = self.expand_query_terms(query, expand_query)
        if expanded_query != query:
            logger.info(f"Expanded query: '{expanded_query}'")
        
        # TF-IDF sparse search
        logger.debug("Performing TF-IDF sparse search...")
        query_vector = self.vectorizer.transform([expanded_query])
        tfidf_scores = cosine_similarity(query_vector, self.tfidf_matrix).ravel()
        tfidf_indices = np.argsort(tfidf_scores)[::-1][:max(top_k * 3, 50)].tolist()
        
        # Dense vector search
        logger.debug("Performing dense vector search...")
        query_embedding = self.sentence_model.encode([expanded_query], normalize_embeddings=True)
        dense_scores, dense_indices = self.faiss_index.search(
            np.asarray(query_embedding, dtype="float32"), 
            max(top_k * 3, 50)
        )
        dense_indices = dense_indices[0].tolist()
        dense_scores = dense_scores[0].tolist()
        
        # Combine results using Reciprocal Rank Fusion
        logger.debug("Combining results using RRF...")
        fused_results = self.reciprocal_rank_fusion([tfidf_indices, dense_indices], k=60)
        
        # Process and filter results
        results = []
        for idx, rrf_score in fused_results:
            if idx < 0 or idx >= len(self.tfidf_metadata):
                continue
            
            metadata = self.tfidf_metadata[idx]
            
            # Apply filters
            if not self.apply_filters(metadata, filters):
                continue
            
            # Get individual scores
            tfidf_score = float(tfidf_scores[idx])
            
            # Get dense score if available
            try:
                dense_pos = dense_indices.index(idx)
                dense_score = float(dense_scores[dense_pos])
            except ValueError:
                dense_score = 0.0
            
            # Apply minimum score filter
            if tfidf_score < min_score:
                continue
            
            result = {
                'rank': len(results) + 1,
                'company_ref_no': metadata.get('company_ref_no', ''),
                'company_name': metadata.get('company_name', 'Unknown'),
                'location': metadata.get('location', 'Unknown'),
                'scale': metadata.get('scale', 'Unknown'),
                'core_expertise': metadata.get('core_expertise', 'Unknown'),
                'industry_domain': metadata.get('industry_domain', 'Unknown'),
                'organization_type': metadata.get('organization_type', 'Unknown'),
                'summary': metadata.get('summary', ''),
                'scores': {
                    'rrf': float(rrf_score),
                    'tfidf': tfidf_score,
                    'dense': dense_score
                }
            }
            
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        logger.info(f"Found {len(results)} results after filtering")
        return results
    
    def initialize(self):
        """Initialize all components for hybrid search."""
        logger.debug("Initializing hybrid search components...")
        
        self.load_tfidf_index()
        self.load_dense_index()
        self.load_sentence_model()
        
        logger.debug("Hybrid search initialization complete")
    
    def save_results_to_csv(self, results: List[Dict[str, Any]], filename: str):
        """Save search results to CSV file."""
        import csv
        
        logger.info(f"Saving {len(results)} results to CSV: {filename}")
        
        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                if not results:
                    return
                
                fieldnames = ['rank', 'company_name', 'scale', 'location', 'core_expertise', 
                             'industry_domain', 'organization_type', 'rrf_score', 'tfidf_score', 'dense_score']
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    writer.writerow({
                        'rank': result['rank'],
                        'company_name': result['company_name'],
                        'scale': result['scale'],
                        'location': result['location'],
                        'core_expertise': result['core_expertise'],
                        'industry_domain': result['industry_domain'],
                        'organization_type': result['organization_type'],
                        'rrf_score': result['scores']['rrf'],
                        'tfidf_score': result['scores']['tfidf'],
                        'dense_score': result['scores']['dense']
                    })
            
            logger.info(f"Results saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}")
            raise

def hybrid_search(query: str, top_k: int = 10, min_score: float = 0.0, 
                 filter_scale: str = None, filter_country: str = None,
                 filter_industry: str = None, expand_query: bool = True,
                 save_csv: str = None, config=None) -> List[Dict[str, Any]]:
    """
    Perform hybrid search with direct parameters (no argument parsing).
    
    Args:
        query: Search query string
        top_k: Number of results to return
        min_score: Minimum TF-IDF score threshold
        filter_scale: Filter by company scale
        filter_country: Filter by country
        filter_industry: Filter by industry
        expand_query: Whether to expand query terms
        save_csv: Path to save results as CSV (optional)
        config: Configuration object (if None, will load from default)
    
    Returns:
        List of search results
    """
    # Load configuration if not provided
    if config is None:
        config = load_config()
        if config is None:
            logger.error("Failed to load configuration")
            return []
    
    # Initialize hybrid search
    search_engine = CompanyHybridSearch(config)
    search_engine.initialize()
    
    # Prepare filters
    filters = {}
    if filter_scale:
        filters['scale'] = filter_scale
    if filter_country:
        filters['country'] = filter_country
    if filter_industry:
        filters['industry'] = filter_industry
    
    # Perform search
    results = search_engine.search(
        query=query,
        top_k=top_k,
        min_score=min_score,
        expand_query=expand_query,
        filters=filters
    )
    
    # Save to CSV if requested
    if save_csv and results:
        search_engine.save_results_to_csv(results, save_csv)
    
    return results

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Hybrid Search for Company Data")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--min-score", "-s", type=float, default=0.0, help="Minimum TF-IDF score")
    parser.add_argument("--filter-scale", help="Filter by company scale")
    parser.add_argument("--filter-country", help="Filter by country")
    parser.add_argument("--filter-industry", help="Filter by industry")
    parser.add_argument("--no-expand", action="store_true", help="Disable query expansion")
    parser.add_argument("--save-csv", help="Save results to CSV file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    if config is None:
        logger.error("Failed to load configuration")
        return
    
    # Initialize hybrid search
    search_engine = CompanyHybridSearch(config)
    search_engine.initialize()
    
    # Prepare filters
    filters = {}
    if args.filter_scale:
        filters['scale'] = args.filter_scale
    if args.filter_country:
        filters['country'] = args.filter_country
    if args.filter_industry:
        filters['industry'] = args.filter_industry
    
    # Perform search
    results = search_engine.search(
        query=args.query,
        top_k=args.top_k,
        min_score=args.min_score,
        expand_query=not args.no_expand,
        filters=filters
    )
    
    # Display results
    if not results:
        logger.info("No results found matching your criteria")
        return
    
    logger.info(f"Found {len(results)} results:")
    print(f"\nSearch Results for: '{args.query}'")
    print("=" * 80)
    
    for result in results:
        print(f"\n{result['rank']}. {result['company_name']}")
        print(f"   Scale: {result['scale']} | Location: {result['location']}")
        print(f"   Core Expertise: {result['core_expertise']}")
        print(f"   Industry: {result['industry_domain']}")
        print(f"   Scores → RRF: {result['scores']['rrf']:.4f}, "
              f"TF-IDF: {result['scores']['tfidf']:.4f}, "
              f"Dense: {result['scores']['dense']:.4f}")
        
        if result['summary']:
            print(f"   Summary: {result['summary'][:200]}...")
    
    # Save to CSV if requested
    if args.save_csv:
        search_engine.save_results_to_csv(results, args.save_csv)

if __name__ == "__main__":
    main()

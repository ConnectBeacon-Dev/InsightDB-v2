#!/usr/bin/env python3
"""
Company Search API

This module provides a unified API for company search functionality with three main operations:
1. Create - Build all search indices from raw data
2. Search - Perform hybrid search queries
3. Cleanup - Clean up generated indices and temporary files

Usage:
    from src.company_index.company_search_api import CompanySearchAPI
    
    api = CompanySearchAPI()
    
    # Create indices
    api.create()
    
    # Search
    results = api.search("software development", top_k=10)
    
    # Cleanup
    api.cleanup()
"""

import os
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.load_config import (
    load_config,
    get_company_mapped_data_processed_data_store,
    get_company_mapped_data_tfidf_search_store,
    get_company_mapped_data_dense_index_store,
    get_domain_mapped_csv_store
)

# Import the individual components
from src.company_index.process_company_data import process_company_data
from src.company_index.generate_company_cin_files import generate_company_cin_files
from src.company_index.create_tfidf_search_index import create_search_index
from src.company_index.build_dense_index import build_dense_index
from src.company_index.hybrid_search import hybrid_search

# Import query planning components
try:
    from src.query_engine.build_query_domain import IndustrialQueryPlanner
    from src.company_index.query_plan_executor import QueryPlanExecutor
    QUERY_PLANNING_AVAILABLE = True
except ImportError as e:
    print(f"Query planning not available: {e}")
    QUERY_PLANNING_AVAILABLE = False

class CompanySearchAPI:
    """
    Unified API for company search functionality.
    
    Provides three main operations:
    - create(): Build all search indices from raw data
    - search(): Perform hybrid search queries
    - cleanup(): Clean up generated indices and temporary files
    """
    
    def __init__(self, config_path: str = None, enable_query_planning: bool = True):
        """
        Initialize the Company Search API.
        
        Args:
            config_path: Path to configuration file (optional)
            enable_query_planning: Whether to enable intelligent query planning (default: True)
        """
        (config, logger) = load_config(config_path) if config_path else load_config()
        if config is None or logger is None:
            raise RuntimeError("Failed to load configuration")
        
        self.config = config
        self.logger = logger

        # Initialize query planning if available and enabled
        self.query_planner = None
        self.query_planning_enabled = False
        
        if enable_query_planning and QUERY_PLANNING_AVAILABLE:
            try:
                # Pass the config object instead of path string
                self.query_planner = IndustrialQueryPlanner(self.config)
                self.query_planning_enabled = True
                self.logger.info("Query planning enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize query planner: {e}")
                self.query_planner = None
                self.query_planning_enabled = False
        elif enable_query_planning:
            self.logger.warning("Query planning requested but not available")
        
        self.logger.info(f"Company Search API initialized (Query Planning: {'✓' if self.query_planning_enabled else '✗'})")
    
    def create(self, force_rebuild: bool = False) -> bool:
        """
        Create all search indices from raw data.
        
        This method orchestrates the complete pipeline:
        1. Process company data from CSV files
        2. Generate company CIN files
        3. Create TF-IDF search index
        4. Build dense vector index
        
        Args:
            force_rebuild: If True, rebuild even if indices already exist
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Starting search index creation pipeline...")
        
        try:
            # Check if indices already exist and force_rebuild is False
            if not force_rebuild and self._indices_exist():
                self.logger.info("Search indices already exist. Use force_rebuild=True to recreate.")
                return True
            
            # Step 1: Process company data
            self.logger.info("Step 1/4: Processing company data...")
            try:
                process_company_data()  # This function doesn't take parameters
                self.logger.info("✓ Company data processing completed")
            except Exception as e:
                self.logger.error(f"Failed to process company data: {e}")
                return False
            
            # Step 2: Generate company CIN files
            self.logger.info("Step 2/4: Generating company CIN files...")
            try:
                generate_company_cin_files()  # This function doesn't take parameters
                self.logger.info("✓ Company CIN files generation completed")
            except Exception as e:
                self.logger.error(f"Failed to generate company CIN files: {e}")
                return False
            
            # Step 3: Create TF-IDF search index
            self.logger.info("Step 3/4: Creating TF-IDF search index...")
            try:
                create_search_index(self.config, self.logger)
                self.logger.info("✓ TF-IDF search index creation completed")
            except Exception as e:
                self.logger.error(f"Failed to create TF-IDF search index: {e}")
                return False
            
            # Step 4: Build dense vector index
            self.logger.info("Step 4/4: Building dense vector index...")
            try:
                build_dense_index(self.config)
                self.logger.info("✓ Dense vector index creation completed")
            except Exception as e:
                self.logger.error(f"Failed to build dense vector index: {e}")
                return False
            
            self.logger.info("🎉 Search index creation pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in create pipeline: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10, min_score: float = 0.0,
               filter_scale: str = None, filter_country: str = None,
               filter_industry: str = None, expand_query: bool = True,
               save_csv: str = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search on company data.
        
        Args:
            query: Search query string
            top_k: Number of results to return (default: 10)
            min_score: Minimum TF-IDF score threshold (default: 0.0)
            filter_scale: Filter by company scale (e.g., "Small", "Medium", "Large")
            filter_country: Filter by country name
            filter_industry: Filter by industry domain
            expand_query: Whether to expand query terms with synonyms (default: True)
            save_csv: Path to save results as CSV (optional)
            
        Returns:
            List[Dict]: List of search results with company information and scores
            
        Raises:
            RuntimeError: If search indices are not available
        """
        self.logger.info(f"Performing search for query: '{query}'")
        
        # Check if indices exist
        if not self._indices_exist():
            raise RuntimeError("Search indices not found. Please run create() first.")
        
        try:
            results = hybrid_search(
                query=query,
                top_k=top_k,
                min_score=min_score,
                filter_scale=filter_scale,
                filter_country=filter_country,
                filter_industry=filter_industry,
                expand_query=expand_query,
                save_csv=save_csv,
                config=self.config
            )
            
            self.logger.info(f"Search completed. Found {len(results)} results.")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            raise
    
    def intelligent_search(self, query: str, use_planning: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Perform intelligent search with optional query planning and decomposition.
        
        This method can automatically break down complex queries into multiple steps,
        execute them with proper dependency management, and aggregate results intelligently.
        
        Args:
            query: Search query string
            use_planning: Whether to use query planning (default: True)
            **kwargs: Additional search parameters (top_k, filters, etc.)
            
        Returns:
            Dict containing:
            - 'results': Final aggregated results
            - 'query_plan': The execution plan used (if planning was used)
            - 'execution_steps': Step-by-step execution details
            - 'metadata': Performance and confidence metrics
            
        Raises:
            RuntimeError: If search indices are not available
        """
        self.logger.info(f"Performing intelligent search for query: '{query}'")
        
        # Check if indices exist
        if not self._indices_exist():
            raise RuntimeError("Search indices not found. Please run create() first.")
        
        # If planning is disabled or not available, fall back to regular search
        if not use_planning or not self.query_planning_enabled:
            self.logger.info("Using direct search (planning disabled or unavailable)")
            try:
                results = self.search(query, **kwargs)
                return {
                    'results': results,
                    'query_plan': None,
                    'execution_steps': [{'step': 'direct_search', 'results_count': len(results)}],
                    'metadata': {
                        'planning_used': False,
                        'confidence': 1.0,
                        'search_type': 'direct',
                        'total_steps': 1
                    }
                }
            except Exception as e:
                self.logger.error(f"Error in direct search: {e}")
                raise
        
        try:
            # Generate query plan
            self.logger.info("Generating query plan...")
            query_plan = self.query_planner.plan_query(query, use_smart_fallback=True)
            self.logger.info(f"Generated {query_plan.plan_type.value} plan with {len(query_plan.sub_queries)} steps")
            
            # Execute query plan
            executor = QueryPlanExecutor(self)
            execution_result = executor.execute_plan(query_plan)
            
            # Apply final filtering and limiting based on original parameters
            final_results = execution_result['results']
            
            # Apply top_k limit if specified
            top_k = kwargs.get('top_k', 10)
            if top_k and len(final_results) > top_k:
                final_results = final_results[:top_k]
                execution_result['results'] = final_results
                execution_result['metadata']['final_results_count'] = len(final_results)
                execution_result['metadata']['limited_by_top_k'] = True
            
            # Apply min_score filter if specified
            min_score = kwargs.get('min_score', 0.0)
            if min_score > 0.0:
                filtered_results = []
                for result in final_results:
                    # Check if any score meets the minimum threshold
                    scores = result.get('scores', {})
                    if any(score >= min_score for score in scores.values() if isinstance(score, (int, float))):
                        filtered_results.append(result)
                
                if len(filtered_results) != len(final_results):
                    execution_result['results'] = filtered_results
                    execution_result['metadata']['final_results_count'] = len(filtered_results)
                    execution_result['metadata']['filtered_by_min_score'] = True
            
            # Save to CSV if requested
            save_csv = kwargs.get('save_csv')
            if save_csv and execution_result['results']:
                try:
                    import pandas as pd
                    df = pd.DataFrame(execution_result['results'])
                    df.to_csv(save_csv, index=False)
                    execution_result['metadata']['saved_to_csv'] = save_csv
                    self.logger.info(f"Results saved to CSV: {save_csv}")
                except Exception as e:
                    self.logger.warning(f"Failed to save results to CSV: {e}")
            
            self.logger.info(f"Intelligent search completed: {len(execution_result['results'])} final results")
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Error in intelligent search: {e}")
            # Fall back to direct search on error
            self.logger.info("Falling back to direct search due to error")
            try:
                results = self.search(query, **kwargs)
                return {
                    'results': results,
                    'query_plan': None,
                    'execution_steps': [{'step': 'fallback_search', 'results_count': len(results)}],
                    'metadata': {
                        'planning_used': False,
                        'confidence': 0.5,
                        'search_type': 'fallback',
                        'error': str(e),
                        'total_steps': 1
                    }
                }
            except Exception as fallback_error:
                self.logger.error(f"Fallback search also failed: {fallback_error}")
                raise RuntimeError(f"Both intelligent and fallback search failed: {e}")
    
    def cleanup(self) -> bool:
        """
        Clean up generated indices and temporary files.
        
        This will remove:
        - TF-IDF search index files
        - Dense vector index files
        - Company CIN files
        - Processed data files
        
        Args:
            confirm: Must be True to actually perform cleanup (safety measure)
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Starting cleanup of search indices and generated files...")
        
        try:
            cleanup_paths = []
            
            # Get paths to clean up
            try:
                tfidf_store = get_company_mapped_data_tfidf_search_store(self.config)
                cleanup_paths.append(("TF-IDF search store", tfidf_store))
            except Exception:
                pass
            
            try:
                dense_store = get_company_mapped_data_dense_index_store(self.config)
                cleanup_paths.append(("Dense index store", dense_store))
            except Exception:
                pass
            
            try:
                company_store = get_company_mapped_data_processed_data_store(self.config)
                cleanup_paths.append(("Company processed data store", company_store))
            except Exception:
                pass
            
            try:
                domain_store = get_domain_mapped_csv_store(self.config)
                cleanup_paths.append(("Domain mapped CSV store", domain_store))
            except Exception:
                pass
            
            # Perform cleanup
            cleaned_count = 0
            for name, path in cleanup_paths:
                if path and Path(path).exists():
                    try:
                        if Path(path).is_dir():
                            shutil.rmtree(path)
                        else:
                            Path(path).unlink()
                        self.logger.info(f"✓ Cleaned up {name}: {path}")
                        cleaned_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to clean up {name} at {path}: {e}")
                else:
                    self.logger.debug(f"Path not found or already clean: {name}")
            
            self.logger.info(f"🧹 Cleanup completed. Removed {cleaned_count} directories/files.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return False
    
    def status(self) -> Dict[str, Any]:
        """
        Get the status of search indices and data files.
        
        Returns:
            Dict: Status information about various components
        """
        self.logger.info("Checking status of search indices...")
        
        status = {
            "indices_ready": False,
            "components": {},
            "config_loaded": self.config is not None
        }
        
        try:
            # Check TF-IDF index
            tfidf_store = get_company_mapped_data_tfidf_search_store(self.config)
            tfidf_index_file = Path(tfidf_store) / "tfidf_search_index.pkl"
            status["components"]["tfidf_index"] = {
                "exists": tfidf_index_file.exists(),
                "path": str(tfidf_index_file)
            }
            
            # Check dense index
            dense_store = get_company_mapped_data_dense_index_store(self.config)
            faiss_index_file = Path(dense_store) / "faiss.index"
            status["components"]["dense_index"] = {
                "exists": faiss_index_file.exists(),
                "path": str(faiss_index_file)
            }
            
            # Check company data
            company_store = get_company_mapped_data_processed_data_store(self.config)
            company_files = list(Path(company_store).glob("*_INFO.json")) if Path(company_store).exists() else []
            status["components"]["company_data"] = {
                "exists": len(company_files) > 0,
                "count": len(company_files),
                "path": str(company_store)
            }
            
            # Check domain mapped data
            domain_store = get_domain_mapped_csv_store(self.config)
            domain_files = list(Path(domain_store).glob("*.csv")) if Path(domain_store).exists() else []
            status["components"]["domain_data"] = {
                "exists": len(domain_files) > 0,
                "count": len(domain_files),
                "path": str(domain_store)
            }
            
            # Overall readiness
            status["indices_ready"] = (
                status["components"]["tfidf_index"]["exists"] and
                status["components"]["dense_index"]["exists"] and
                status["components"]["company_data"]["exists"]
            )
            
        except Exception as e:
            self.logger.error(f"Error checking status: {e}")
            status["error"] = str(e)
        
        return status
    
    def _indices_exist(self) -> bool:
        """Check if the required search indices exist."""
        try:
            # Check TF-IDF index
            tfidf_store = get_company_mapped_data_tfidf_search_store(self.config)
            tfidf_index_file = Path(tfidf_store) / "tfidf_search_index.pkl"
            
            # Check dense index
            dense_store = get_company_mapped_data_dense_index_store(self.config)
            faiss_index_file = Path(dense_store) / "faiss.index"
            
            return tfidf_index_file.exists() and faiss_index_file.exists()
            
        except Exception:
            return False

# Convenience functions for direct usage
def create_indices(config_path: str = None, force_rebuild: bool = False) -> bool:
    """
    Convenience function to create search indices.
    
    Args:
        config_path: Path to configuration file (optional)
        force_rebuild: If True, rebuild even if indices already exist
        
    Returns:
        bool: True if successful, False otherwise
    """
    api = CompanySearchAPI(config_path)
    return api.create(force_rebuild)

def search_companies(query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function to search companies.
    
    Args:
        query: Search query string
        top_k: Number of results to return
        **kwargs: Additional search parameters
        
    Returns:
        List[Dict]: Search results
    """
    api = CompanySearchAPI()
    return api.search(query, top_k, **kwargs)

def cleanup_indices(confirm: bool = False) -> bool:
    """
    Convenience function to cleanup search indices.
    
    Args:
        confirm: Must be True to actually perform cleanup
        
    Returns:
        bool: True if successful, False otherwise
    """
    api = CompanySearchAPI()
    return api.cleanup(confirm)

def get_status() -> Dict[str, Any]:
    """
    Convenience function to get status of search indices.
    
    Returns:
        Dict: Status information
    """
    api = CompanySearchAPI()
    return api.status()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Company Search API")
    parser.add_argument("action", choices=["create", "search", "cleanup", "status"],
                       help="Action to perform")
    parser.add_argument("--query", "-q", help="Search query (for search action)")
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of results")
    parser.add_argument("--force", action="store_true", help="Force rebuild (for create)")
    parser.add_argument("--confirm", action="store_true", help="Confirm cleanup")
    parser.add_argument("--filter-scale", help="Filter by scale")
    parser.add_argument("--filter-country", help="Filter by country")
    parser.add_argument("--filter-industry", help="Filter by industry")
    parser.add_argument("--save-csv", help="Save results to CSV")
    
    args = parser.parse_args()
    
    api = CompanySearchAPI()
    
    if args.action == "create":
        success = api.create(force_rebuild=args.force)
        exit(0 if success else 1)
        
    elif args.action == "search":
        if not args.query:
            print("Error: --query is required for search action")
            exit(1)
        
        results = api.search(
            query=args.query,
            top_k=args.top_k,
            filter_scale=args.filter_scale,
            filter_country=args.filter_country,
            filter_industry=args.filter_industry,
            save_csv=args.save_csv
        )
        
        print(f"\nFound {len(results)} results:")
        for result in results:
            print(f"{result['rank']}. {result['company_name']} - {result['core_expertise']}")
            
    elif args.action == "cleanup":
        success = api.cleanup()
        exit(0 if success else 1)
        
    elif args.action == "status":
        status = api.status()
        print(f"Indices Ready: {status['indices_ready']}")
        print(f"Config Loaded: {status['config_loaded']}")
        print("\nComponents:")
        for name, info in status.get('components', {}).items():
            print(f"  {name}: {'✓' if info['exists'] else '✗'}")

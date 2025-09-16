#!/usr/bin/env python3
"""
Embedding-Based Query Decomposer

A simpler, faster approach that:
1. Breaks down user queries into semantic components
2. Identifies dependencies between components
3. Uses embeddings for semantic search instead of complex LLM planning
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np

from src.load_config import load_config, get_logger

# Import helper methods
from src.query_engine.embedding_query_decomposer_helpers import (
    build_certification_search_query,
    build_testing_search_query,
    build_rd_search_query,
    build_defence_search_query,
    build_tech_area_search_query,
    build_export_search_query,
    filter_certifications_by_companies,
    filter_testing_by_companies,
    filter_rd_by_companies,
    filter_defence_by_companies,
    filter_tech_areas_by_companies,
    filter_exports_by_companies
)

class ComponentType(Enum):
    ENTITY = "entity"           # Companies, products, etc.
    ATTRIBUTE = "attribute"     # Scale, location, expertise, etc.
    RELATIONSHIP = "relationship"  # "from", "with", "having", etc.
    ACTION = "action"          # List, show, find, etc.

class DependencyType(Enum):
    SEQUENTIAL = "sequential"   # B depends on A
    PARALLEL = "parallel"      # A and B can run together
    FILTER = "filter"          # A filters B

@dataclass
class QueryComponent:
    id: str
    text: str
    type: ComponentType
    keywords: List[str]
    embedding: Optional[np.ndarray] = None
    domain_hint: Optional[str] = None

@dataclass
class ComponentDependency:
    source: str
    target: str
    type: DependencyType
    reason: str

@dataclass
class DecomposedQuery:
    original_query: str
    components: List[QueryComponent]
    dependencies: List[ComponentDependency]
    execution_order: List[str]
    search_strategy: str

class EmbeddingQueryDecomposer:
    """
    Fast query decomposer using embeddings and pattern matching.
    Much simpler and faster than LLM-based approaches.
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.logger.info("Initializing EmbeddingQueryDecomposer...")
        
        # Load domain patterns for quick matching
        self.domain_patterns = self._load_domain_patterns()
        
        # Initialize sentence transformer for embeddings
        self.embedder = None
        self._init_embedder()
        
        self.logger.info("EmbeddingQueryDecomposer initialized successfully")
    
    def _init_embedder(self):
        """Initialize sentence transformer for embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            import os
            from pathlib import Path
            
            # Try to get model from config, fallback to default
            model_name = self.config.get('sentence_transformer_model_name', 'all-MiniLM-L6-v2')
            
            # Check if it's a local path that doesn't exist
            if os.path.sep in model_name or model_name.startswith('models'):
                model_path = Path(model_name)
                if not model_path.exists():
                    self.logger.warning(f"⚠️ Local model path not found: {model_name}")
                    self.logger.info("🔄 Attempting to download model from Hugging Face...")
                    
                    # Extract the actual model name from the path
                    if 'sentence-transformers_all-mpnet-base-v2' in model_name:
                        model_name = 'sentence-transformers/all-mpnet-base-v2'
                    elif 'all-MiniLM-L6-v2' in model_name:
                        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
                    else:
                        # Fallback to a reliable model
                        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
                        self.logger.info(f"🔄 Using fallback model: {model_name}")
            
            # Load the model (will download automatically if not cached)
            self.logger.info(f"🔄 Loading sentence transformer: {model_name}")
            self.embedder = SentenceTransformer(model_name)
            self.logger.info(f"✅ Sentence transformer loaded successfully: {model_name}")
            
        except ImportError:
            self.logger.warning("⚠️ sentence-transformers not available, using keyword matching only")
            self.embedder = None
        except Exception as e:
            self.logger.error(f"❌ Failed to load sentence transformer: {e}")
            self.logger.info("🔄 Trying fallback model: all-MiniLM-L6-v2")
            try:
                # Try a smaller, more reliable model as fallback
                self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                self.logger.info("✅ Fallback sentence transformer loaded successfully")
            except Exception as fallback_error:
                self.logger.error(f"❌ Fallback model also failed: {fallback_error}")
                self.logger.warning("⚠️ Continuing without embeddings - using keyword matching only")
                self.embedder = None
    
    def _load_domain_patterns(self) -> Dict:
        """Load simplified domain patterns for fast matching."""
        return {
            'entities': {
                'companies': ['companies', 'firms', 'organizations', 'businesses', 'manufacturers', 'suppliers'],
                'products': ['products', 'goods', 'items', 'materials', 'components', 'equipment'],
                'certifications': ['certificates', 'certifications', 'ISO', 'standards', 'accredited', 'ISO 9001', 'ISO 27001', 'NABL'],
                'testing_facilities': ['testing', 'test facilities', 'testing capabilities', 'test labs', 'high voltage test', 'environmental test', 'vibration test', 'EMI EMC'],
                'rd_facilities': ['R&D', 'research', 'development', 'R&D facilities', 'research facilities', 'development capabilities'],
                'defence_platforms': ['defence platform', 'platform', 'defence systems'],
                'tech_areas': ['PTA', 'tech area', 'technology area', 'platform technology area'],
                'exports': ['exports', 'exported items', 'exported products', 'export capabilities']
            },
            'attributes': {
                'scale': ['small scale', 'medium scale', 'large scale', 'SME', 'MSME', 'enterprise'],
                'location': ['from', 'in', 'located', 'based', 'state', 'city', 'region'],
                'expertise': ['expertise', 'specialization', 'capabilities', 'experience', 'skills', 'electrical', 'mechanical', 'aerospace', 'software', 'embedded systems', 'communications', 'RF', 'defence'],
                'industry': ['aerospace', 'defence', 'automotive', 'electronics', 'power', 'energy', 'oil & gas'],
                'certification_type': ['ISO 9001', 'ISO 27001', 'NABL', 'certification type'],
                'rd_category': ['electrical', 'mechanical', 'materials', 'embedded', 'RF', 'power systems', 'electronics'],
                'test_category': ['high voltage', 'environmental', 'vibration', 'EMI EMC', 'electrical testing']
            },
            'actions': {
                'list': ['list', 'show', 'display', 'get'],
                'find': ['find', 'search', 'locate', 'identify'],
                'filter': ['with', 'having', 'that have', 'which have']
            },
            'relationships': {
                'possession': ['have', 'has', 'with', 'having'],
                'origin': ['from', 'by', 'supplied by', 'manufactured by'],
                'location': ['in', 'at', 'located in', 'based in']
            }
        }
    
    def decompose_query(self, user_query: str) -> DecomposedQuery:
        """
        Main method to decompose query into components and dependencies.
        
        Args:
            user_query: The user's natural language query
            
        Returns:
            DecomposedQuery with components, dependencies, and execution order
        """
        self.logger.info(f"🔍 Decomposing query: '{user_query}'")
        
        # Step 1: Extract components using pattern matching and embeddings
        components = self._extract_components(user_query)
        
        # Step 2: Identify dependencies between components
        dependencies = self._identify_dependencies(components, user_query)
        
        # Step 3: Determine execution order
        execution_order = self._determine_execution_order(components, dependencies)
        
        # Step 4: Choose search strategy
        search_strategy = self._choose_search_strategy(components, dependencies)
        
        decomposed = DecomposedQuery(
            original_query=user_query,
            components=components,
            dependencies=dependencies,
            execution_order=execution_order,
            search_strategy=search_strategy
        )
        
        self.logger.info(f"✅ Query decomposed into {len(components)} components with {len(dependencies)} dependencies")
        return decomposed
    
    def _extract_components(self, query: str) -> List[QueryComponent]:
        """Extract semantic components from the query."""
        components = []
        query_lower = query.lower()
        
        # Extract entities - check for compound terms first, then individual terms
        for entity_type, keywords in self.domain_patterns['entities'].items():
            found_entity = False
            # Sort keywords by length (longest first) to match compound terms first
            sorted_keywords = sorted(keywords, key=len, reverse=True)
            
            for keyword in sorted_keywords:
                if keyword.lower() in query_lower:
                    component = QueryComponent(
                        id=f"entity_{entity_type}",
                        text=keyword,
                        type=ComponentType.ENTITY,
                        keywords=[keyword],
                        domain_hint=entity_type
                    )
                    if self.embedder:
                        component.embedding = self.embedder.encode([keyword])[0]
                    components.append(component)
                    found_entity = True
                    break  # Only add one per entity type
            
            # Special handling for certification detection
            if not found_entity and entity_type == 'certifications':
                # Check for "ISO certification" pattern
                if 'iso' in query_lower and 'certification' in query_lower:
                    component = QueryComponent(
                        id=f"entity_{entity_type}",
                        text="ISO certification",
                        type=ComponentType.ENTITY,
                        keywords=["ISO certification", "ISO", "certification"],
                        domain_hint=entity_type
                    )
                    if self.embedder:
                        component.embedding = self.embedder.encode(["ISO certification"])[0]
                    components.append(component)
        
        # Extract attributes (avoid duplicates)
        added_attributes = set()
        for attr_type, keywords in self.domain_patterns['attributes'].items():
            for keyword in keywords:
                if keyword in query_lower:
                    # Extract the specific value (e.g., "small" from "small scale")
                    value = self._extract_attribute_value(query_lower, keyword, attr_type)
                    component_text = value or keyword
                    
                    # Avoid duplicate attributes with same text
                    if component_text not in added_attributes:
                        component = QueryComponent(
                            id=f"attr_{attr_type}",
                            text=component_text,
                            type=ComponentType.ATTRIBUTE,
                            keywords=[keyword, value] if value else [keyword],
                            domain_hint=attr_type
                        )
                        if self.embedder:
                            component.embedding = self.embedder.encode([component.text])[0]
                        components.append(component)
                        added_attributes.add(component_text)
                    break
        
        # Extract actions
        for action_type, keywords in self.domain_patterns['actions'].items():
            for keyword in keywords:
                if keyword in query_lower:
                    component = QueryComponent(
                        id=f"action_{action_type}",
                        text=keyword,
                        type=ComponentType.ACTION,
                        keywords=[keyword],
                        domain_hint=action_type
                    )
                    components.append(component)
                    break
        
        # Extract relationships
        for rel_type, keywords in self.domain_patterns['relationships'].items():
            for keyword in keywords:
                if keyword in query_lower:
                    component = QueryComponent(
                        id=f"rel_{rel_type}",
                        text=keyword,
                        type=ComponentType.RELATIONSHIP,
                        keywords=[keyword],
                        domain_hint=rel_type
                    )
                    components.append(component)
                    break
        
        return components
    
    def _extract_attribute_value(self, query: str, keyword: str, attr_type: str) -> Optional[str]:
        """Extract specific attribute values from the query."""
        if attr_type == 'scale':
            if 'small' in query:
                return 'small'
            elif 'medium' in query:
                return 'medium'
            elif 'large' in query:
                return 'large'
        elif attr_type == 'location':
            # Extract state/city names after location keywords
            location_pattern = r'(?:from|in|located|based)\s+([A-Za-z\s]+?)(?:\s|$|with|having)'
            match = re.search(location_pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        elif attr_type == 'expertise':
            # Extract expertise areas
            expertise_pattern = r'(?:expertise|specialization|capabilities)\s+in\s+([A-Za-z\s]+?)(?:\s|$|and|with)'
            match = re.search(expertise_pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _identify_dependencies(self, components: List[QueryComponent], query: str) -> List[ComponentDependency]:
        """Identify dependencies between components."""
        dependencies = []
        
        # Find entity and attribute pairs for filtering
        entities = [c for c in components if c.type == ComponentType.ENTITY]
        attributes = [c for c in components if c.type == ComponentType.ATTRIBUTE]
        
        for entity in entities:
            for attr in attributes:
                # Check if attribute should filter entity
                if self._should_filter(entity, attr, query):
                    dep = ComponentDependency(
                        source=attr.id,
                        target=entity.id,
                        type=DependencyType.FILTER,
                        reason=f"{attr.text} filters {entity.text}"
                    )
                    dependencies.append(dep)
        
        # Identify sequential dependencies between entities
        company_comp = next((c for c in entities if 'companies' in c.keywords), None)
        
        # Companies -> Products dependency
        product_comp = next((c for c in entities if 'products' in c.keywords), None)
        if company_comp and product_comp:
            if any(rel in query.lower() for rel in ['from', 'by', 'supplied by']):
                dep = ComponentDependency(
                    source=company_comp.id,
                    target=product_comp.id,
                    type=DependencyType.SEQUENTIAL,
                    reason="Products need to be filtered by companies first"
                )
                dependencies.append(dep)
        
        # Companies -> Certifications dependency
        cert_comp = next((c for c in entities if 'certifications' in c.keywords), None)
        if company_comp and cert_comp:
            if any(rel in query.lower() for rel in ['with', 'having', 'certified']):
                dep = ComponentDependency(
                    source=company_comp.id,
                    target=cert_comp.id,
                    type=DependencyType.SEQUENTIAL,
                    reason="Certifications need to be filtered by companies first"
                )
                dependencies.append(dep)
        
        # Companies -> R&D Facilities dependency
        rd_comp = next((c for c in entities if 'rd_facilities' in c.keywords), None)
        if company_comp and rd_comp:
            if any(rel in query.lower() for rel in ['with', 'having', 'r&d', 'research']):
                dep = ComponentDependency(
                    source=company_comp.id,
                    target=rd_comp.id,
                    type=DependencyType.SEQUENTIAL,
                    reason="R&D facilities need to be filtered by companies first"
                )
                dependencies.append(dep)
        
        # Companies -> Testing Facilities dependency
        test_comp = next((c for c in entities if 'testing_facilities' in c.keywords), None)
        if company_comp and test_comp:
            if any(rel in query.lower() for rel in ['with', 'having', 'testing', 'test']):
                dep = ComponentDependency(
                    source=company_comp.id,
                    target=test_comp.id,
                    type=DependencyType.SEQUENTIAL,
                    reason="Testing facilities need to be filtered by companies first"
                )
                dependencies.append(dep)
        
        # Companies -> Defence Platforms dependency
        defence_comp = next((c for c in entities if 'defence_platforms' in c.keywords), None)
        if company_comp and defence_comp:
            if any(rel in query.lower() for rel in ['for', 'working on', 'platform']):
                dep = ComponentDependency(
                    source=company_comp.id,
                    target=defence_comp.id,
                    type=DependencyType.SEQUENTIAL,
                    reason="Defence platforms need to be filtered by companies first"
                )
                dependencies.append(dep)
        
        # Companies -> Tech Areas dependency
        tech_comp = next((c for c in entities if 'tech_areas' in c.keywords), None)
        if company_comp and tech_comp:
            if any(rel in query.lower() for rel in ['in', 'for', 'technology']):
                dep = ComponentDependency(
                    source=company_comp.id,
                    target=tech_comp.id,
                    type=DependencyType.SEQUENTIAL,
                    reason="Technology areas need to be filtered by companies first"
                )
                dependencies.append(dep)
        
        # Companies -> Exports dependency
        export_comp = next((c for c in entities if 'exports' in c.keywords), None)
        if company_comp and export_comp:
            if any(rel in query.lower() for rel in ['exporting', 'export', 'exported']):
                dep = ComponentDependency(
                    source=company_comp.id,
                    target=export_comp.id,
                    type=DependencyType.SEQUENTIAL,
                    reason="Export capabilities need to be filtered by companies first"
                )
                dependencies.append(dep)
        
        return dependencies
    
    def _should_filter(self, entity: QueryComponent, attr: QueryComponent, query: str) -> bool:
        """Determine if an attribute should filter an entity."""
        query_lower = query.lower()
        
        # Scale attributes filter companies
        if attr.domain_hint == 'scale' and 'companies' in entity.keywords:
            return True
        
        # Location attributes filter companies
        if attr.domain_hint == 'location' and 'companies' in entity.keywords:
            return True
        
        # Expertise attributes filter companies
        if attr.domain_hint == 'expertise' and 'companies' in entity.keywords:
            return True
        
        return False
    
    def _determine_execution_order(self, components: List[QueryComponent], dependencies: List[ComponentDependency]) -> List[str]:
        """Determine the optimal execution order based on dependencies."""
        # Simple topological sort
        in_degree = {comp.id: 0 for comp in components}
        graph = {comp.id: [] for comp in components}
        
        # Build dependency graph
        for dep in dependencies:
            graph[dep.source].append(dep.target)
            in_degree[dep.target] += 1
        
        # Topological sort
        queue = [comp_id for comp_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return execution_order
    
    def _choose_search_strategy(self, components: List[QueryComponent], dependencies: List[ComponentDependency]) -> str:
        """Choose the best search strategy based on components."""
        has_embeddings = self.embedder is not None
        has_dependencies = len(dependencies) > 0
        
        if has_embeddings and has_dependencies:
            return "hybrid_embedding_sequential"
        elif has_embeddings:
            return "embedding_parallel"
        elif has_dependencies:
            return "keyword_sequential"
        else:
            return "keyword_parallel"
    
    def execute_decomposed_query(self, decomposed: DecomposedQuery) -> Dict:
        """
        Execute the decomposed query using embeddings and search.
        
        Args:
            decomposed: The decomposed query structure
            
        Returns:
            Dictionary with search results
        """
        self.logger.info(f"🚀 Executing decomposed query with strategy: {decomposed.search_strategy}")
        self.logger.info(f"📋 Execution plan: {len(decomposed.execution_order)} steps")
        
        try:
            from src.company_index.company_search_api import CompanySearchAPI
            # Initialize without query planning to avoid loading LLM
            api = CompanySearchAPI(enable_query_planning=False)
            
            # Check if indices are ready
            status = api.status()
            if not status.get('indices_ready', False):
                self.logger.warning("⚠️ Search indices not ready. Creating indices...")
                if not api.create():
                    return {"error": "Failed to create search indices"}
                self.logger.info("✅ Search indices created successfully")
                
        except ImportError:
            self.logger.error("CompanySearchAPI not available")
            return {"error": "Search API not available"}
        except Exception as e:
            self.logger.error(f"Failed to initialize CompanySearchAPI: {e}")
            return {"error": f"Search API initialization failed: {e}"}
        
        intermediate_results = {}
        
        # Log execution start
        self.logger.info("\n" + "⚡ QUERY EXECUTION STEPS:")
        self.logger.info("-" * 40)
        
        # Execute components in order
        for step_num, comp_id in enumerate(decomposed.execution_order, 1):
            component = next(c for c in decomposed.components if c.id == comp_id)
            self.logger.info(f"\n🔄 STEP {step_num}/{len(decomposed.execution_order)}: Executing {comp_id}")
            self.logger.info(f"   📝 Component: {component.type.value.upper()} - '{component.text}'")
            self.logger.info(f"   🏷️ Domain: {component.domain_hint}")
            self.logger.info(f"   🔑 Keywords: {component.keywords}")
            
            try:
                if component.type == ComponentType.ENTITY:
                    if 'companies' in component.keywords:
                        # Build search query for companies
                        search_query = self._build_company_search_query(component, decomposed)
                        
                        # Apply filters from attributes
                        filters = self._extract_filters_for_component(comp_id, decomposed)
                        
                        # Search for companies with filters
                        search_results = api.search(
                            query=search_query,
                            top_k=100,  # Get more results for filtering
                            filter_scale=filters.get('scale'),
                            filter_country=filters.get('country'),
                            filter_industry=filters.get('industry')
                        )
                        
                        intermediate_results[comp_id] = search_results
                        self.logger.info(f"   ✅ Found {len(search_results)} companies")
                        
                        # Log sample results for visibility
                        if search_results:
                            self.logger.info(f"   📋 Sample results (showing first 3):")
                            for i, result in enumerate(search_results[:3], 1):
                                company_name = result.get('company_name', 'Unknown')
                                expertise = result.get('core_expertise', 'N/A')
                                location = result.get('state', result.get('location', 'N/A'))
                                self.logger.info(f"      {i}. {company_name} | {expertise} | {location}")
                        else:
                            self.logger.info(f"   ❌ No companies found for this query")
                        
                    elif 'products' in component.keywords:
                        # Build search query for products
                        search_query = self._build_product_search_query(component, decomposed)
                        
                        # Check if we need to filter by companies from previous results
                        company_filter = self._get_company_filter(comp_id, decomposed.dependencies, intermediate_results)
                        
                        if company_filter:
                            # Search for products and filter by companies
                            all_products = api.search(query=search_query, top_k=200)
                            filtered_products = self._filter_products_by_companies(all_products, company_filter)
                            intermediate_results[comp_id] = filtered_products
                            logger.info(f"   ✅ Found {len(filtered_products)} filtered products")
                        else:
                            # Direct product search
                            search_results = api.search(query=search_query, top_k=50)
                            intermediate_results[comp_id] = search_results
                            logger.info(f"   ✅ Found {len(search_results)} products")
                    
                    elif component.domain_hint == 'certifications':
                        # Use hybrid search engine for companies with certifications
                        self.logger.info(f"   🔍 Using hybrid search for certifications")
                        
                        # Import the hybrid search function
                        from src.query_engine.hybrid_search_engine import search_companies_with_certifications
                        
                        # Check if we need to filter by companies from previous results
                        company_filter = self._get_company_filter(comp_id, decomposed.dependencies, intermediate_results)
                        
                        if company_filter:
                            # Filter certifications by companies using hybrid search
                            filtered_certs = filter_certifications_by_companies(company_filter)
                            intermediate_results[comp_id] = filtered_certs
                            self.logger.info(f"   ✅ Found {len(filtered_certs)} filtered companies with certifications")
                        else:
                            # Direct certification search using hybrid engine
                            search_results = search_companies_with_certifications(top_k=50)
                            intermediate_results[comp_id] = search_results
                            self.logger.info(f"   ✅ Found {len(search_results)} companies with certifications")
                    
                    elif component.domain_hint == 'testing_facilities':
                        # Use hybrid search engine for companies with testing facilities
                        self.logger.info(f"   🔍 Using hybrid search for testing facilities")
                        
                        # Import the hybrid search function
                        from src.query_engine.hybrid_search_engine import search_companies_with_testing_facilities
                        
                        # Check if we need to filter by companies from previous results
                        company_filter = self._get_company_filter(comp_id, decomposed.dependencies, intermediate_results)
                        
                        if company_filter:
                            # Filter testing facilities by companies using hybrid search
                            filtered_testing = filter_testing_by_companies(company_filter)
                            intermediate_results[comp_id] = filtered_testing
                            self.logger.info(f"   ✅ Found {len(filtered_testing)} filtered companies with testing facilities")
                        else:
                            # Direct testing facilities search using hybrid engine
                            search_results = search_companies_with_testing_facilities(top_k=50)
                            intermediate_results[comp_id] = search_results
                            self.logger.info(f"   ✅ Found {len(search_results)} companies with testing facilities")
                    
                    elif 'rd_facilities' in component.keywords:
                        # Search for companies with R&D facilities
                        search_query = build_rd_search_query(component, decomposed)
                        
                        # Check if we need to filter by companies from previous results
                        company_filter = self._get_company_filter(comp_id, decomposed.dependencies, intermediate_results)
                        
                        if company_filter:
                            # Filter R&D facilities by companies
                            filtered_rd = filter_rd_by_companies(company_filter)
                            intermediate_results[comp_id] = filtered_rd
                            logger.info(f"   ✅ Found {len(filtered_rd)} filtered R&D facilities")
                        else:
                            # Direct R&D facilities search
                            search_results = api.search(query=search_query, top_k=50)
                            intermediate_results[comp_id] = search_results
                            logger.info(f"   ✅ Found {len(search_results)} companies with R&D facilities")
                    
                    elif 'defence_platforms' in component.keywords:
                        # Search for companies working on defence platforms
                        search_query = build_defence_search_query(component, decomposed)
                        
                        # Check if we need to filter by companies from previous results
                        company_filter = self._get_company_filter(comp_id, decomposed.dependencies, intermediate_results)
                        
                        if company_filter:
                            # Filter defence platforms by companies
                            filtered_defence = filter_defence_by_companies(company_filter)
                            intermediate_results[comp_id] = filtered_defence
                            logger.info(f"   ✅ Found {len(filtered_defence)} filtered defence platforms")
                        else:
                            # Direct defence platforms search
                            search_results = api.search(query=search_query, top_k=50)
                            intermediate_results[comp_id] = search_results
                            logger.info(f"   ✅ Found {len(search_results)} companies with defence platforms")
                    
                    elif 'tech_areas' in component.keywords:
                        # Search for companies in specific technology areas
                        search_query = build_tech_area_search_query(component, decomposed)
                        
                        # Check if we need to filter by companies from previous results
                        company_filter = self._get_company_filter(comp_id, decomposed.dependencies, intermediate_results)
                        
                        if company_filter:
                            # Filter tech areas by companies
                            filtered_tech = filter_tech_areas_by_companies(company_filter)
                            intermediate_results[comp_id] = filtered_tech
                            logger.info(f"   ✅ Found {len(filtered_tech)} filtered technology areas")
                        else:
                            # Direct tech areas search
                            search_results = api.search(query=search_query, top_k=50)
                            intermediate_results[comp_id] = search_results
                            logger.info(f"   ✅ Found {len(search_results)} companies in technology areas")
                    
                    elif 'exports' in component.keywords:
                        # Search for companies with export capabilities
                        search_query = build_export_search_query(component, decomposed)
                        
                        # Check if we need to filter by companies from previous results
                        company_filter = self._get_company_filter(comp_id, decomposed.dependencies, intermediate_results)
                        
                        if company_filter:
                            # Filter exports by companies
                            filtered_exports = filter_exports_by_companies(company_filter)
                            intermediate_results[comp_id] = filtered_exports
                            logger.info(f"   ✅ Found {len(filtered_exports)} filtered export capabilities")
                        else:
                            # Direct exports search
                            search_results = api.search(query=search_query, top_k=50)
                            intermediate_results[comp_id] = search_results
                            logger.info(f"   ✅ Found {len(search_results)} companies with export capabilities")
                
                elif component.type == ComponentType.ATTRIBUTE:
                    # Apply attribute filters to existing results
                    self._apply_attribute_filter(component, intermediate_results, decomposed.dependencies)
                    self.logger.info(f"   ✅ Applied {component.text} filter")
                
            except Exception as e:
                self.logger.error(f"   ❌ Component {comp_id} failed: {e}")
                intermediate_results[comp_id] = []
        
        # Compile final results
        final_results = self._compile_final_results(decomposed, intermediate_results)
        
        self.logger.info(f"✅ Query execution completed")
        return final_results
    
    def _get_company_filter(self, comp_id: str, dependencies: List[ComponentDependency], intermediate_results: Dict) -> Optional[List]:
        """Get company filter for product searches."""
        for dep in dependencies:
            if dep.target == comp_id and dep.type == DependencyType.SEQUENTIAL:
                return intermediate_results.get(dep.source, [])
        return None
    
    def _filter_products_by_companies(self, products: List[Dict], companies: List[Dict]) -> List[Dict]:
        """Filter products by company list."""
        company_names = {c.get('company_name', '').lower() for c in companies}
        filtered = []
        
        for product in products:
            product_company = product.get('company_name', '').lower()
            if product_company in company_names:
                filtered.append(product)
        
        return filtered
    
    def _apply_attribute_filter(self, attr_component: QueryComponent, intermediate_results: Dict, dependencies: List[ComponentDependency]):
        """Apply attribute filters to intermediate results."""
        for dep in dependencies:
            if dep.source == attr_component.id and dep.type == DependencyType.FILTER:
                target_results = intermediate_results.get(dep.target, [])
                if target_results:
                    filtered = self._filter_by_attribute(target_results, attr_component)
                    intermediate_results[dep.target] = filtered
    
    def _filter_by_attribute(self, results: List[Dict], attr_component: QueryComponent) -> List[Dict]:
        """Filter results by attribute value."""
        if attr_component.domain_hint == 'scale':
            scale_value = attr_component.text.title()  # "small" -> "Small"
            return [r for r in results if r.get('company_scale') == scale_value or r.get('scale') == scale_value]
        
        elif attr_component.domain_hint == 'location':
            location_value = attr_component.text.title()
            return [r for r in results if 
                   location_value.lower() in r.get('state', '').lower() or
                   location_value.lower() in r.get('city', '').lower() or
                   location_value.lower() in r.get('location', '').lower()]
        
        return results
    
    def _build_company_search_query(self, component: QueryComponent, decomposed: DecomposedQuery) -> str:
        """Build search query for company searches."""
        # Extract key terms for company search
        company_terms = ["companies", "manufacturers", "suppliers", "firms", "organizations"]
        
        # Add expertise/industry terms if present
        expertise_terms = []
        for comp in decomposed.components:
            if comp.type == ComponentType.ATTRIBUTE and comp.domain_hint in ['expertise', 'industry']:
                # Use the actual text, not just keywords
                if comp.text and comp.text not in expertise_terms:
                    expertise_terms.append(comp.text)
        
        # Build focused search query
        if expertise_terms:
            query = f"{' '.join(expertise_terms)} companies manufacturers"
            self.logger.info(f"   🔍 Built search query: '{query}'")
            return query
        else:
            query = "companies manufacturers suppliers"
            self.logger.info(f"   🔍 Built search query: '{query}'")
            return query
    
    def _build_product_search_query(self, component: QueryComponent, decomposed: DecomposedQuery) -> str:
        """Build search query for product searches."""
        # Start with the original query as base
        base_query = decomposed.original_query
        
        # Extract key terms for product search
        product_terms = ["products", "goods", "items", "materials", "components", "equipment"]
        
        # Add industry/domain terms if present
        domain_terms = []
        for comp in decomposed.components:
            if comp.type == ComponentType.ATTRIBUTE and comp.domain_hint in ['industry', 'expertise']:
                domain_terms.extend(comp.keywords)
        
        # Build focused search query
        if domain_terms:
            return f"{' '.join(domain_terms)} products components materials"
        else:
            return "products components materials equipment"
    
    def _extract_filters_for_component(self, comp_id: str, decomposed: DecomposedQuery) -> Dict[str, str]:
        """Extract filters that should be applied to a component."""
        filters = {}
        
        # Find attributes that filter this component
        for dep in decomposed.dependencies:
            if dep.target == comp_id and dep.type == DependencyType.FILTER:
                # Find the source attribute component
                attr_comp = next((c for c in decomposed.components if c.id == dep.source), None)
                if attr_comp:
                    if attr_comp.domain_hint == 'scale':
                        filters['scale'] = attr_comp.text.title()  # "small" -> "Small"
                    elif attr_comp.domain_hint == 'location':
                        # Map location to country if it's a state/region
                        location = attr_comp.text.title()
                        if location.lower() in ['karnataka', 'maharashtra', 'tamil nadu', 'gujarat', 'rajasthan']:
                            filters['country'] = 'India'  # Assume Indian states
                        else:
                            filters['country'] = location
                    elif attr_comp.domain_hint == 'industry':
                        filters['industry'] = attr_comp.text
        
        return filters
    
    def _compile_final_results(self, decomposed: DecomposedQuery, intermediate_results: Dict) -> Dict:
        """Compile final results from intermediate results."""
        companies = []
        products = []
        certifications = []
        facilities = []
        
        # Check if we have both certifications and testing facilities
        has_certifications = any(c.domain_hint == 'certifications' for c in decomposed.components if c.type == ComponentType.ENTITY)
        has_testing = any(c.domain_hint == 'testing_facilities' for c in decomposed.components if c.type == ComponentType.ENTITY)
        
        if has_certifications and has_testing:
            # Special handling for queries requiring BOTH certifications AND testing
            self.logger.info("🔍 Query requires BOTH certifications AND testing facilities - using intersection")
            
            from src.query_engine.hybrid_search_engine import search_companies_with_multiple_capabilities
            
            # Get companies with both capabilities
            both_companies = search_companies_with_multiple_capabilities(
                certifications=True,
                testing_facilities=True,
                top_k=50
            )
            
            companies.extend(both_companies)
            self.logger.info(f"✅ Found {len(both_companies)} companies with BOTH certifications AND testing facilities")
        else:
            # Regular processing for single capability queries
            for comp_id, results in intermediate_results.items():
                component = next(c for c in decomposed.components if c.id == comp_id)
                if component.type == ComponentType.ENTITY:
                    if 'companies' in component.keywords:
                        # Only add generic company results if no specific capability results exist
                        if not has_certifications and not has_testing:
                            companies.extend(results)
                    elif 'products' in component.keywords:
                        products.extend(results)
                    elif component.domain_hint == 'certifications':
                        companies.extend(results)  # Certification results are companies
                    elif component.domain_hint == 'testing_facilities':
                        companies.extend(results)  # Testing facility results are companies
        
        # Remove duplicates based on company_name or id
        companies = self._remove_duplicates(companies)
        products = self._remove_duplicates(products)
        
        # Limit results for display
        max_display = 10
        
        return {
            "query": decomposed.original_query,
            "strategy": decomposed.search_strategy,
            "components": len(decomposed.components),
            "dependencies": len(decomposed.dependencies),
            "results": {
                "companies_count": len(companies),
                "products_count": len(products),
                "certifications_count": len(certifications),
                "facilities_count": len(facilities),
                "companies": companies[:max_display],
                "products": products[:max_display],
                "certifications": certifications[:max_display] if certifications else [],
                "facilities": facilities[:max_display] if facilities else []
            },
            "execution_details": {
                "execution_order": decomposed.execution_order,
                "component_results": {k: len(v) if isinstance(v, list) else 1 for k, v in intermediate_results.items()}
            }
        }
    
    def _remove_duplicates(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on company_name or unique identifier."""
        seen = set()
        unique_results = []
        
        for result in results:
            # Use company_name as the unique identifier
            identifier = result.get('company_name', result.get('id', str(result)))
            if identifier not in seen:
                seen.add(identifier)
                unique_results.append(result)
        
        return unique_results

# Main execution function
def execute_embedding_query(user_query: str, config, logger) -> Dict:
    """
    Execute query using embedding-based decomposition.
    
    This is much faster and simpler than LLM-based approaches.
    """
    
    decomposer = EmbeddingQueryDecomposer(config, logger)
    decomposed = decomposer.decompose_query(user_query)
    logger.info("**************************")
    logger.info(decomposed)
    logger.info("=========================")
    return decomposer.execute_decomposed_query(decomposed)

# Enhanced execution with LLM validation
def execute_enhanced_embedding_query(user_query: str, config, logger, enable_llm_validation=True) -> Dict:
    """
    Execute query with enhanced LLM validation enabled.
    
    This approach:
    1. First tries fast embedding-based decomposition
    2. Calculates confidence based on component matches and dependencies
    3. If confidence is low, uses LLM to validate/improve the decomposition
    4. Executes the final decomposition and returns results
    
    Args:
        user_query: The user's natural language query
        config: Configuration dictionary
        enable_llm_validation: Enable micro-LLM validation for low-confidence queries
    
    Returns:
        Dictionary with search results and execution details
    """
    logger.info("=" * 80)
    logger.info("🚀 ENHANCED EMBEDDING QUERY EXECUTION STARTED")
    logger.info("=" * 80)
    
    # Log the original user query
    logger.info(f"📝 USER QUERY: '{user_query}'")
    logger.info(f"🔧 LLM Validation Enabled: {enable_llm_validation}")
    
    # Step 1: Fast embedding-based decomposition
    logger.info("\n" + "🔍 STEP 1: QUERY DECOMPOSITION")
    logger.info("-" * 50)
    decomposer = EmbeddingQueryDecomposer(config, logger)
    decomposed = decomposer.decompose_query(user_query)
    
    # Log detailed decomposition results
    _log_decomposition_details(decomposed, logger)
    
    # Step 2: Calculate confidence score
    logger.info("\n" + "📊 STEP 2: CONFIDENCE CALCULATION")
    logger.info("-" * 50)
    confidence = _calculate_decomposition_confidence(decomposed, user_query, logger)
    logger.info(f"📊 Final Decomposition Confidence: {confidence:.2f}")
    
    # Step 3: LLM validation for low-confidence queries
    logger.info("\n" + "🤖 STEP 3: LLM VALIDATION CHECK")
    logger.info("-" * 50)
    llm_validation_used = False
    llm_confidence_threshold = config.get('llm_confidence_threshold', 0.7)
    logger.info(f"🎯 LLM Confidence Threshold: {llm_confidence_threshold}")
    
    if enable_llm_validation and confidence < llm_confidence_threshold:
        logger.info(f"⚠️ Low confidence ({confidence:.2f} < {llm_confidence_threshold}), triggering LLM validation...")
        try:
            validated_decomposed = _validate_with_llm(decomposed, user_query, config, logger)
            if validated_decomposed:
                logger.info("🔄 Updating decomposition with LLM improvements...")
                _log_decomposition_comparison(decomposed, validated_decomposed, logger)
                decomposed = validated_decomposed
                llm_validation_used = True
                logger.info("✅ LLM validation completed successfully")
            else:
                logger.warning("⚠️ LLM validation failed, using original decomposition")
        except Exception as e:
            logger.error(f"❌ LLM validation error: {e}")
            logger.info("🔄 Continuing with original embedding-based decomposition")
    else:
        logger.info(f"✅ High confidence ({confidence:.2f}), skipping LLM validation")
    
    # Step 4: Execute the decomposed query
    logger.info("\n" + "🚀 STEP 4: QUERY EXECUTION")
    logger.info("-" * 50)
    result = decomposer.execute_decomposed_query(decomposed)
    
    # Step 5: Add metadata about the process
    result['confidence'] = confidence
    result['llm_validation_used'] = llm_validation_used
    result['llm_confidence_threshold'] = llm_confidence_threshold
    result['approach'] = 'enhanced_embedding_with_llm_validation'
    
    # Log final execution summary
    logger.info("\n" + "📋 EXECUTION SUMMARY")
    logger.info("-" * 50)
    logger.info(f"✅ Query Processing Complete")
    logger.info(f"📊 Final Confidence: {confidence:.2f}")
    logger.info(f"🤖 LLM Validation Used: {llm_validation_used}")
    logger.info(f"🔍 Search Strategy: {result.get('strategy', 'Unknown')}")
    logger.info(f"📈 Total Results: {result.get('results', {}).get('companies_count', 0)} companies, {result.get('results', {}).get('products_count', 0)} products")
    logger.info("=" * 80)
    
    return result

def _log_decomposition_details(decomposed: DecomposedQuery, logger) -> None:
    """Log detailed information about the query decomposition."""
    logger.info("🔍 QUERY DECOMPOSITION RESULTS:")
    logger.info(f"📝 Original Query: '{decomposed.original_query}'")
    logger.info(f"🔧 Search Strategy: {decomposed.search_strategy}")
    
    # Log components breakdown
    logger.info(f"\n📦 COMPONENTS FOUND ({len(decomposed.components)} total):")
    if decomposed.components:
        for i, comp in enumerate(decomposed.components, 1):
            logger.info(f"  {i}. {comp.type.value.upper()}: '{comp.text}' (ID: {comp.id})")
            logger.info(f"     - Domain: {comp.domain_hint}")
            logger.info(f"     - Keywords: {comp.keywords}")
            if comp.embedding is not None:
                logger.info(f"     - Embedding: Vector of {len(comp.embedding)} dimensions")
            else:
                logger.info(f"     - Embedding: None (keyword-only matching)")
    else:
        logger.info("  ❌ No components found")
    
    # Log dependencies
    logger.info(f"\n🔗 DEPENDENCIES FOUND ({len(decomposed.dependencies)} total):")
    if decomposed.dependencies:
        for i, dep in enumerate(decomposed.dependencies, 1):
            logger.info(f"  {i}. {dep.source} --[{dep.type.value}]--> {dep.target}")
            logger.info(f"     - Reason: {dep.reason}")
    else:
        logger.info("  ❌ No dependencies found")
    
    # Log execution order
    logger.info(f"\n⚡ EXECUTION ORDER ({len(decomposed.execution_order)} steps):")
    if decomposed.execution_order:
        for i, comp_id in enumerate(decomposed.execution_order, 1):
            comp = next((c for c in decomposed.components if c.id == comp_id), None)
            if comp:
                logger.info(f"  {i}. {comp_id} ({comp.type.value}: '{comp.text}')")
            else:
                logger.info(f"  {i}. {comp_id} (component not found)")
    else:
        logger.info("  ❌ No execution order determined")

def _log_decomposition_comparison(original: DecomposedQuery, improved: DecomposedQuery, logger) -> None:
    """Log comparison between original and LLM-improved decomposition."""
    logger.info("🔄 DECOMPOSITION COMPARISON (Original vs LLM-Improved):")
    
    # Compare components
    logger.info(f"📦 Components: {len(original.components)} → {len(improved.components)}")
    
    # Find new components
    original_ids = {c.id for c in original.components}
    improved_ids = {c.id for c in improved.components}
    new_components = improved_ids - original_ids
    removed_components = original_ids - improved_ids
    
    if new_components:
        logger.info("  ✅ Added components:")
        for comp_id in new_components:
            comp = next((c for c in improved.components if c.id == comp_id), None)
            if comp:
                logger.info(f"    + {comp.type.value}: '{comp.text}' ({comp_id})")
    
    if removed_components:
        logger.info("  ❌ Removed components:")
        for comp_id in removed_components:
            comp = next((c for c in original.components if c.id == comp_id), None)
            if comp:
                logger.info(f"    - {comp.type.value}: '{comp.text}' ({comp_id})")
    
    # Compare dependencies
    logger.info(f"🔗 Dependencies: {len(original.dependencies)} → {len(improved.dependencies)}")
    
    # Compare execution order
    logger.info(f"⚡ Execution Steps: {len(original.execution_order)} → {len(improved.execution_order)}")

def _calculate_decomposition_confidence(decomposed: DecomposedQuery, user_query: str, logger=None) -> float:
    """
    Calculate confidence score for the decomposition based on:
    - Number of components found vs query complexity
    - Quality of component matches
    - Presence of dependencies
    - Coverage of the original query
    """
    query_words = user_query.lower().split()
    query_length = len(query_words)
    
    # Base confidence from component coverage
    component_coverage = 0.0
    matched_words = set()
    
    for component in decomposed.components:
        for keyword in component.keywords:
            if keyword and keyword.lower() in user_query.lower():
                # Add words covered by this component
                keyword_words = keyword.lower().split()
                matched_words.update(keyword_words)
                component_coverage += len(keyword_words)
    
    # Coverage ratio (how much of the query is covered by components)
    coverage_ratio = min(1.0, component_coverage / max(1, query_length))
    
    # Component quality score
    component_score = 0.0
    if decomposed.components:
        # Bonus for having entities (main query targets)
        entity_count = sum(1 for c in decomposed.components if c.type == ComponentType.ENTITY)
        if entity_count > 0:
            component_score += 0.3
        
        # Bonus for having attributes (filters/qualifiers)
        attr_count = sum(1 for c in decomposed.components if c.type == ComponentType.ATTRIBUTE)
        if attr_count > 0:
            component_score += 0.2
        
        # Bonus for having actions (clear intent)
        action_count = sum(1 for c in decomposed.components if c.type == ComponentType.ACTION)
        if action_count > 0:
            component_score += 0.1
    
    # Dependency score (logical relationships found)
    dependency_score = 0.0
    if decomposed.dependencies:
        dependency_score = min(0.2, len(decomposed.dependencies) * 0.1)
    
    # Combine scores
    confidence = (coverage_ratio * 0.5) + (component_score * 0.3) + (dependency_score * 0.2)
    
    # Ensure confidence is between 0 and 1
    return max(0.0, min(1.0, confidence))

def _validate_with_llm(decomposed: DecomposedQuery, user_query: str, config: Dict, logger) -> Optional[DecomposedQuery]:
    """
    Use LLM to validate and potentially improve the query decomposition.
    
    This is a lightweight LLM usage focused on validation rather than full planning.
    """
    try:
        from src.query_engine.qwen_client import QwenClient
        
        # Initialize LLM client
        qwen_client = QwenClient(config)
        
        # Create validation prompt
        validation_prompt = _create_validation_prompt(decomposed, user_query)
        
        # Get LLM response
        logger.info("🤖 Querying LLM for decomposition validation...")
        llm_response = qwen_client.generate_response(validation_prompt, max_tokens=500)
        
        # Parse LLM response and improve decomposition
        improved_decomposed = _parse_llm_validation_response(llm_response, decomposed, user_query, logger)
        
        return improved_decomposed
        
    except ImportError:
        logger.warning("⚠️ QwenClient not available for LLM validation")
        return None
    except Exception as e:
        logger.error(f"❌ LLM validation failed: {e}")
        return None

def _create_validation_prompt(decomposed: DecomposedQuery, user_query: str) -> str:
    """Create a focused prompt for LLM validation of query decomposition."""
    
    components_text = []
    for comp in decomposed.components:
        components_text.append(f"- {comp.type.value}: {comp.text} (domain: {comp.domain_hint})")
    
    dependencies_text = []
    for dep in decomposed.dependencies:
        dependencies_text.append(f"- {dep.source} -> {dep.target} ({dep.type.value}): {dep.reason}")
    
    prompt = f"""You are validating a query decomposition for an industrial company search system.

Original Query: "{user_query}"

Current Decomposition:
Components Found:
{chr(10).join(components_text) if components_text else "None"}

Dependencies Found:
{chr(10).join(dependencies_text) if dependencies_text else "None"}

Task: Validate this decomposition and suggest improvements. Focus on:
1. Are important entities (companies, products, etc.) correctly identified?
2. Are filters/attributes (scale, location, expertise) properly extracted?
3. Are there missing components that should be included?
4. Are the dependencies logical?

Respond in this format:
VALIDATION: [GOOD/NEEDS_IMPROVEMENT]
MISSING_COMPONENTS: [list any missing components]
SUGGESTED_IMPROVEMENTS: [brief suggestions]
CONFIDENCE: [0.0-1.0]

Keep response concise and focused."""

    return prompt

def _parse_llm_validation_response(llm_response: str, original_decomposed: DecomposedQuery, user_query: str, logger) -> Optional[DecomposedQuery]:
    """Parse LLM validation response and create improved decomposition if needed."""
    
    try:
        # Simple parsing of LLM response
        lines = llm_response.strip().split('\n')
        validation_status = None
        missing_components = []
        confidence = 0.5
        
        for line in lines:
            line = line.strip()
            if line.startswith('VALIDATION:'):
                validation_status = line.split(':', 1)[1].strip()
            elif line.startswith('MISSING_COMPONENTS:'):
                missing_text = line.split(':', 1)[1].strip()
                if missing_text and missing_text.lower() != 'none':
                    missing_components = [c.strip() for c in missing_text.split(',')]
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                except:
                    confidence = 0.5
        
        # If LLM says decomposition is good, return original
        if validation_status and 'GOOD' in validation_status.upper():
            logger.info("✅ LLM validation: decomposition is good")
            return original_decomposed
        
        # If there are missing components, try to add them
        if missing_components:
            logger.info(f"🔧 LLM suggests adding components: {missing_components}")
            improved_decomposed = _add_missing_components(original_decomposed, missing_components, user_query)
            return improved_decomposed
        
        # Otherwise return original
        return original_decomposed
        
    except Exception as e:
        logger.error(f"❌ Failed to parse LLM validation response: {e}")
        return None

def _add_missing_components(decomposed: DecomposedQuery, missing_components: List[str], user_query: str) -> DecomposedQuery:
    """Add missing components suggested by LLM to the decomposition."""
    
    # This is a simplified implementation - in practice, you'd want more sophisticated component creation
    new_components = list(decomposed.components)
    
    for missing_comp in missing_components:
        missing_comp = missing_comp.lower().strip()
        
        # Try to identify what type of component this should be
        if any(word in missing_comp for word in ['company', 'companies', 'firm', 'organization']):
            if not any(c.domain_hint == 'companies' for c in new_components):
                new_component = QueryComponent(
                    id="entity_companies_llm",
                    text="companies",
                    type=ComponentType.ENTITY,
                    keywords=["companies"],
                    domain_hint="companies"
                )
                new_components.append(new_component)
        
        elif any(word in missing_comp for word in ['product', 'products', 'item', 'material']):
            if not any(c.domain_hint == 'products' for c in new_components):
                new_component = QueryComponent(
                    id="entity_products_llm",
                    text="products",
                    type=ComponentType.ENTITY,
                    keywords=["products"],
                    domain_hint="products"
                )
                new_components.append(new_component)
    
    # Create new decomposed query with additional components
    improved_decomposed = DecomposedQuery(
        original_query=decomposed.original_query,
        components=new_components,
        dependencies=decomposed.dependencies,  # Keep original dependencies for now
        execution_order=decomposed.execution_order,  # Recalculate if needed
        search_strategy=decomposed.search_strategy
    )
    
    return improved_decomposed

# Test function
def test_embedding_decomposer():
    """Test the embedding-based query decomposer."""
    logger.info("🧪 Testing Embedding Query Decomposer")
    logger.info("=" * 60)
    
    test_queries = [
        #"list all products which are supplied by small scale companies",
        "show companies from Karnataka with electrical expertise",
        #"find products from medium scale companies",
        #"companies having ISO certification"
    ]
    
    (config, logger) = load_config()
    
    for query in test_queries:
        logger.info(f"\n🔍 Testing: '{query}'")
        logger.info("-" * 40)
        
        try:
            result = execute_embedding_query(query, config, logger)
            
            logger.info(f"Strategy: {result.get('strategy', 'Unknown')}")
            logger.info(f"Components: {result.get('components', 0)}")
            logger.info(f"Dependencies: {result.get('dependencies', 0)}")
            
            if result.get('results'):
                results = result['results']
                logger.info("************************")
                logger.info(results)
                logger.info("------------------------")
                logger.info(f"Companies: {results.get('companies_count', 0)}")
                logger.info(f"Products: {results.get('products_count', 0)}")
            
            logger.info("✅ Test completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")

def test_enhanced_embedding_decomposer():
    """Test the enhanced embedding-based query decomposer with LLM validation."""

    (config, logger) = load_config()

    logger.info("🧪 Testing Enhanced Embedding Query Decomposer (with LLM validation)")
    logger.info("=" * 80)
    
    # Test queries with varying complexity
    test_queries = [
        # Simple query (high confidence expected)
        #"list companies",
        
        # Medium complexity (may trigger LLM validation)
        "small scale electrical companies",
        
        # Complex/ambiguous query (should trigger LLM validation)
        #"products with advanced capabilities for defense applications",
        
        # Edge case query (likely to trigger LLM validation)
        #"ISO certified manufacturers having R&D facilities"

        #"yzx capbilities"
    ]
    
    
    for query in test_queries:
        logger.info(f"\n🔍 Testing Enhanced: '{query}'")
        logger.info("-" * 50)
        
        try:
            # Test with LLM validation enabled
            result = execute_enhanced_embedding_query(query, config, logger, enable_llm_validation=True)
            
            logger.info(f"Strategy: {result.get('strategy', 'Unknown')}")
            logger.info(f"Components: {result.get('components', 0)}")
            logger.info(f"Dependencies: {result.get('dependencies', 0)}")
            logger.info(f"Confidence: {result.get('confidence', 0):.2f}")
            logger.info(f"LLM Validation Used: {result.get('llm_validation_used', False)}")
            
            if result.get('results'):
                results = result['results']
                logger.info(f"Companies: {results.get('companies_count', 0)}")
                logger.info(f"Products: {results.get('products_count', 0)}")
            
            logger.info("✅ Enhanced test completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Enhanced test failed: {e}")

if __name__ == "__main__":
    # Test both regular and enhanced versions
   #test_embedding_decomposer()
    (config, logger) = load_config()

    logger.info("\n" + "="*100)
    
    # Test enhanced version with LLM validation
    test_enhanced_embedding_decomposer()

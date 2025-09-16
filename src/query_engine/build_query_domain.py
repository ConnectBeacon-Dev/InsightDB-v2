import json
import re
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Import centralized configuration and logging
from src.load_config import load_config, get_logger

# Set up logging
logger = get_logger(__name__)

# Import Qwen model client
try:
    from src.query_engine.qwen_client import create_qwen_client
    QWEN_CLIENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Qwen client not available: {e}")
    QWEN_CLIENT_AVAILABLE = False

class QueryType(Enum):
    SIMPLE = "simple"
    COMPOUND = "compound"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"

class Priority(Enum):
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"

@dataclass
class SubQuery:
    id: str
    query: str
    type: QueryType
    priority: Priority
    dependencies: List[str]
    context_needed: List[str]
    expected_output: str
    reasoning: str
    domain_mapping: Dict[str, List[str]]  # New field for domain mapping

@dataclass
class QueryPlan:
    original_query: str
    plan_type: QueryType
    sub_queries: List[SubQuery]
    execution_order: List[str]
    reasoning: str
    confidence: float

class IndustrialQueryPlanner:
    def __init__(self, config):
        """
        Initialize QueryPlanner with configuration and domain mapping.
        
        Args:
            config: Configuration dictionary
        """
        logger.info("Initializing IndustrialQueryPlanner...")
        
        # Load configuration
        self.config = config
        if not self.config:
            logger.warning("No configuration loaded, using defaults")
            self.config = {}
        
        # Load domain mapping
        self.domain_mapping = self._load_domain_mapping(config.get("domain_mapping"))
        
        # Load table relations
        self.table_relations = self._load_table_relations(config.get("table_relations"))
        
        # Get model paths from config
        self.qwen_model_path = self.config.get('qwen_model_path')
        self.mistral_model_path = self.config.get('mistral_model_path')
        
        # Initialize Qwen model client as primary LLM
        self.qwen_client = None
        if QWEN_CLIENT_AVAILABLE:
            try:
                logger.info("🤖 Initializing Qwen model client...")
                self.qwen_client = create_qwen_client(self.config)
                if self.qwen_client:
                    logger.info("✅ Qwen model client initialized successfully")
                else:
                    logger.warning("⚠️ Qwen model client initialization failed")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Qwen client: {e}")
                self.qwen_client = None
        else:
            logger.warning("⚠️ Qwen client not available, will use smart fallback")
        
        # Enhanced industrial domain patterns with field mappings
        self.industrial_patterns = self._load_industrial_patterns()
        
        self.system_prompt = self._build_system_prompt()
        logger.info("IndustrialQueryPlanner initialized successfully")
    
    def _load_domain_mapping(self, domain_mapping_path: str) -> Dict:
        """Load the domain mapping configuration."""
        try:
            with open(domain_mapping_path, 'r') as f:
                mapping = json.load(f)
                logger.info("Domain mapping loaded successfully")
                return mapping
        except Exception as e:
            logger.error(f"Failed to load domain mapping: {e}")
            return {}
    
    def _load_table_relations(self, relations_path: str) -> List[Dict]:
        """Load table relations from relations.json file."""
        try:
            with open(relations_path, 'r') as f:
                relations = json.load(f)
                logger.info(f"Table relations loaded successfully: {len(relations)} relationships")
                return relations
        except Exception as e:
            logger.error(f"Failed to load table relations: {e}")
            return []
    
    def _load_industrial_patterns(self) -> Dict:
        """Load patterns specific to industrial/business queries with domain field mappings."""
        base_patterns = {
            'company_search_indicators': {
                'keywords': ['companies', 'firms', 'organizations', 'businesses', 'manufacturers', 'suppliers', 'vendors', 'enterprises'],
                'primary_domain': 'CompanyProfile.BasicInfo',
                'fields': ['CompanyName', 'CompanyStatus', 'CompanyClass']
            },
            'capability_indicators': {
                'keywords': ['expertise', 'specialization', 'capabilities', 'competencies', 'experience', 'proficiency', 'skills', 'knowledge'],
                'primary_domain': 'BusinessDomain.CoreExpertise',
                'fields': ['CoreExpertiseName', 'OtherCompanyCoreExpertise']
            },
            'certification_indicators': {
                'keywords': ['ISO', 'certificate', 'certified', 'accredited', 'compliance', 'standards', 'quality', 'certification'],
                'primary_domain': 'QualityAndCompliance.Certifications',
                'fields': ['Cert_Type', 'Certification_Type', 'Certificate_No']
            },
            'scale_indicators': {
                'keywords': ['small scale', 'medium scale', 'large scale', 'SME', 'MSME', 'enterprise', 'startup', 'multinational'],
                'primary_domain': 'CompanyProfile.Classification',
                'fields': ['CompanyScale', 'OtherScale']
            },
            'product_indicators': {
                'keywords': ['products', 'goods', 'items', 'materials', 'components', 'parts', 'equipment', 'machinery', 'tools'],
                'primary_domain': 'ProductsAndServices.Products',
                'fields': ['ProductName', 'ProductDesc', 'ProductTypeName']
            },
            'location_indicators': {
                'keywords': ['from state', 'in region', 'located in', 'based in', 'operating in', 'presence in'],
                'primary_domain': 'CompanyProfile.ContactInfo',
                'fields': ['State', 'CityName', 'District', 'CountryName']
            },
            'industry_indicators': {
                'keywords': ['industry', 'sector', 'domain', 'aerospace', 'defence', 'oil & gas', 'automotive', 'power', 'energy', 'electronics'],
                'primary_domain': 'BusinessDomain.Industry',
                'fields': ['IndustryDomainType', 'IndustrySubDomainName']
            },
            'rd_indicators': {
                'keywords': ['R&D', 'research', 'development', 'lab', 'laboratory', 'innovation'],
                'primary_domain': 'ResearchAndDevelopment.RDCapabilities',
                'fields': ['RDCategoryName', 'RDSubCategoryName', 'RD_Details']
            },
            'testing_indicators': {
                'keywords': ['test', 'testing', 'facility', 'lab', 'laboratory', 'quality assurance', 'validation'],
                'primary_domain': 'QualityAndCompliance.TestingCapabilities',
                'fields': ['CategoryName', 'SubCategoryName', 'TestDetails']
            },
            'financial_indicators': {
                'keywords': ['revenue', 'turnover', 'financial', 'sales', 'income'],
                'primary_domain': 'FinancialMetrics.Revenue',
                'fields': ['Amount', 'YearId']
            },
            'defence_indicators': {
                'keywords': ['defence', 'defense', 'military', 'platform', 'pta'],
                'primary_domain': 'ProductsAndServices.DefenceSector',
                'fields': ['Name_of_Defence_Platform', 'PTAName']
            }
        }
        
        # Add specific technical expertise patterns
        technical_patterns = {
            'electrical_indicators': {
                'keywords': ['electrical', 'power systems', 'power electronics', 'electronics', 'high voltage', 'hv'],
                'primary_domain': 'ResearchAndDevelopment.RDCapabilities',
                'fields': ['RDCategoryName', 'RDSubCategoryName'],
                'synonyms': ['electrical systems', 'power', 'voltage', 'electrical engineering']
            },
            'mechanical_indicators': {
                'keywords': ['mechanical', 'aerospace', 'manufacturing'],
                'primary_domain': 'ResearchAndDevelopment.RDCapabilities',
                'fields': ['RDCategoryName', 'RDSubCategoryName']
            }
        }
        
        base_patterns.update(technical_patterns)
        return base_patterns
    
    def _map_query_to_domains(self, query: str, patterns_found: Dict) -> Dict[str, List[str]]:
        """Map query patterns to specific domain fields based on domain_mapping.json"""
        domain_mappings = {}
        
        for pattern_type, indicators in patterns_found.items():
            if pattern_type in self.industrial_patterns:
                pattern_config = self.industrial_patterns[pattern_type]
                primary_domain = pattern_config['primary_domain']
                
                # Get the actual domain structure from domain_mapping
                domain_parts = primary_domain.split('.')
                if len(domain_parts) == 2:
                    domain_category, domain_subcategory = domain_parts
                    
                    if domain_category in self.domain_mapping:
                        if domain_subcategory in self.domain_mapping[domain_category]:
                            # Get the actual field mappings
                            field_configs = self.domain_mapping[domain_category][domain_subcategory]
                            
                            relevant_fields = []
                            for field_config in field_configs:
                                table = field_config.get('table', '')
                                field = field_config.get('field', '')
                                hints = field_config.get('hints', '')
                                synonyms = field_config.get('synonyms', [])
                                
                                # Check if any of the query indicators match field synonyms or hints
                                query_lower = query.lower()
                                field_relevant = False
                                
                                for indicator in indicators:
                                    if (indicator.lower() in hints.lower() or 
                                        indicator.lower() in str(synonyms).lower() or
                                        any(syn.lower() in query_lower for syn in synonyms)):
                                        field_relevant = True
                                        break
                                
                                if field_relevant or not synonyms:  # Include if relevant or no specific synonyms
                                    relevant_fields.append({
                                        'table': table,
                                        'field': field,
                                        'hints': hints,
                                        'is_categorical': field_config.get('is_categorical', False)
                                    })
                            
                            if relevant_fields:
                                domain_mappings[primary_domain] = relevant_fields
        
        return domain_mappings
    
    def _build_system_prompt(self) -> str:
        return """You are an expert query planner specialized in industrial and business intelligence queries. Your job is to:

1. ANALYZE business/industrial queries for complexity and data requirements
2. IDENTIFY specific database fields and tables needed based on domain mapping
3. CREATE structured execution plans considering data dependencies and field relationships
4. OUTPUT valid JSON following the exact schema with domain mappings

DOMAIN-AWARE QUERY ANALYSIS:
- Company Profile: BasicInfo, ContactInfo, Classification
- Business Domain: CoreExpertise, Industry sectors and subdomains
- Products & Services: Product details, Defence platforms, Export info
- Quality & Compliance: Certifications, Testing capabilities with NABL accreditation
- Research & Development: R&D categories, subcategories, and accreditation
- Financial Metrics: Revenue and turnover data

ENHANCED BUSINESS LOGIC PATTERNS:
- Scale-based queries: CompanyProfile.Classification → CompanyProfile.BasicInfo → ProductsAndServices.Products
- Expertise queries: BusinessDomain.CoreExpertise + ResearchAndDevelopment.RDCapabilities
- Certification queries: QualityAndCompliance.Certifications with date validation
- Location-based: CompanyProfile.ContactInfo → filter other domains
- Industry-specific: BusinessDomain.Industry → related capabilities and products

FIELD-LEVEL MAPPING REQUIREMENTS:
- Map query terms to specific table.field combinations
- Consider categorical fields for filtering (is_categorical: true)
- Account for synonyms and hints in field selection
- Plan for joins between related tables (CompanyMaster, ProductTypeMaster, etc.)

OUTPUT REQUIREMENTS:
- Include domain_mapping in each sub_query with specific table.field references
- Use business-meaningful IDs reflecting domain areas (company_profile, product_search, certification_check)
- Specify exact field dependencies for database query optimization
- Include both primary and secondary domain mappings where relevant

REMEMBER: Every sub-query must include domain_mapping with specific table and field references from the domain schema."""

    def _create_planning_prompt(self, user_query: str) -> str:
        # Analyze query for industrial patterns
        query_analysis = self._analyze_query_patterns(user_query)
        domain_mappings = self._map_query_to_domains(user_query, query_analysis['patterns_detected'])
        
        schema_example = {
            "original_query": "string",
            "plan_type": "simple|compound|sequential|parallel|conditional", 
            "sub_queries": [
                {
                    "id": "company_profile_search",
                    "query": "specific search criteria with field-level details",
                    "type": "simple|compound|sequential|parallel|conditional",
                    "priority": "high|medium|low",
                    "dependencies": ["scale_definition"],
                    "context_needed": ["company_database", "size_criteria"],
                    "expected_output": "filtered list of companies with specific fields",
                    "reasoning": "need to identify target companies before listing products",
                    "domain_mapping": {
                        "CompanyProfile.Classification": [
                            {"table": "ScaleMaster", "field": "CompanyScale", "hints": "MSME/Large/Small/Medium", "is_categorical": True}
                        ],
                        "CompanyProfile.BasicInfo": [
                            {"table": "CompanyMaster", "field": "CompanyName", "hints": "Official name of the company", "is_categorical": True}
                        ]
                    }
                }
            ],
            "execution_order": ["scale_definition", "company_profile_search", "product_list"],
            "reasoning": "overall business logic and data flow rationale with domain considerations",
            "confidence": 0.85
        }
        
        examples = self._get_domain_specific_examples()
        
        return f"""ANALYZE INDUSTRIAL/BUSINESS QUERY: "{user_query}"

Query Pattern Analysis:
{json.dumps(query_analysis, indent=2)}

Detected Domain Mappings:
{json.dumps(domain_mappings, indent=2)}

Available Domain Structure:
{json.dumps(self._get_relevant_domain_structure(query_analysis), indent=2)}

Follow this exact JSON schema with domain mappings:
{json.dumps(schema_example, indent=2)}

Domain-Specific Examples:
{examples}

Now analyze: "{user_query}"

Consider:
- Specific table and field mappings from domain_mapping.json
- Data relationships (CompanyMaster → CompanyProducts → ProductTypeMaster)
- Categorical vs non-categorical fields for filtering optimization
- NABL accreditation flags for testing and R&D facilities
- Date-based filtering for certifications and financial data
- Synonyms and hints for field matching

Return only valid JSON with complete domain mappings:"""
    
    def _get_concise_domain_context(self, query_analysis: Dict) -> Dict:
        """
        Get concise domain context focused on detected patterns.
        Avoids overwhelming the LLM with too much information.
        """
        patterns = query_analysis['patterns_detected']
        relevant_domains = {}
        
        # Map patterns to domain categories
        pattern_to_domain = {
            'company_search_indicators': 'CompanyProfile',
            'scale_indicators': 'CompanyProfile', 
            'location_indicators': 'CompanyProfile',
            'product_indicators': 'ProductsAndServices',
            'capability_indicators': 'BusinessDomain',
            'certification_indicators': 'QualityAndCompliance',
            'rd_indicators': 'ResearchAndDevelopment',
            'testing_indicators': 'QualityAndCompliance',
            'industry_indicators': 'BusinessDomain'
        }
        
        # Get only relevant domain categories
        relevant_categories = set()
        for pattern_type in patterns.keys():
            if pattern_type in pattern_to_domain:
                relevant_categories.add(pattern_to_domain[pattern_type])
        
        # Extract relevant domain structure
        for category in relevant_categories:
            if category in self.domain_mapping:
                relevant_domains[category] = {}
                # Get most important subcategories for this category
                important_subcategories = self._get_important_subcategories(category, patterns)
                for subcategory in important_subcategories:
                    if subcategory in self.domain_mapping[category]:
                        relevant_domains[category][subcategory] = self.domain_mapping[category][subcategory][:3]  # Limit to 3 fields
        
        return relevant_domains

    def _get_important_subcategories(self, category: str, patterns: Dict) -> List[str]:
        """Get the most important subcategories for a domain category based on detected patterns."""
        
        # Category-specific important subcategories
        important_map = {
            'CompanyProfile': ['BasicInfo', 'ContactInfo', 'Classification'],
            'ProductsAndServices': ['Products', 'DefenceSector'] if 'defence_indicators' in patterns else ['Products'],
            'BusinessDomain': ['CoreExpertise', 'Industry'],
            'QualityAndCompliance': ['Certifications', 'TestingCapabilities'],
            'ResearchAndDevelopment': ['RDCapabilities'],
            'FinancialMetrics': ['Revenue']
        }
        
        return important_map.get(category, [])

    def _format_domain_context(self, domains: Dict) -> str:
        """Format domain context in a concise, readable way for the LLM."""
        
        if not domains:
            return "Standard company and product domains available"
        
        formatted_lines = []
        for category, subcategories in domains.items():
            formatted_lines.append(f"• {category}:")
            for subcategory, fields in subcategories.items():
                field_names = [f"{field.get('table', '')}.{field.get('field', '')}" for field in fields[:2]]  # Show max 2 fields
                formatted_lines.append(f"  - {subcategory}: {', '.join(field_names)}")
        
        return "\n".join(formatted_lines)

    def _get_relevant_domain_structure(self, query_analysis: Dict) -> Dict:
        """Extract relevant domain structure based on detected patterns."""
        patterns = query_analysis['patterns_detected']
        relevant_domains = {}
        
        for pattern_type in patterns.keys():
            if pattern_type in self.industrial_patterns:
                domain_path = self.industrial_patterns[pattern_type]['primary_domain']
                domain_parts = domain_path.split('.')
                
                if len(domain_parts) == 2:
                    category, subcategory = domain_parts
                    if category in self.domain_mapping:
                        if category not in relevant_domains:
                            relevant_domains[category] = {}
                        if subcategory in self.domain_mapping[category]:
                            relevant_domains[category][subcategory] = self.domain_mapping[category][subcategory]
        
        return relevant_domains
    
    def _analyze_query_patterns(self, query: str) -> Dict:
        """Analyze query for industrial/business patterns with enhanced domain awareness."""
        query_lower = query.lower()
        
        patterns_found = {}
        for pattern_type, pattern_config in self.industrial_patterns.items():
            keywords = pattern_config['keywords']
            synonyms = pattern_config.get('synonyms', [])
            
            matches = [keyword for keyword in keywords if keyword in query_lower]
            synonym_matches = [synonym for synonym in synonyms if synonym in query_lower]
            
            all_matches = matches + synonym_matches
            if all_matches:
                patterns_found[pattern_type] = all_matches
        
        return {
            'patterns_detected': patterns_found,
            'complexity_indicators': {
                'multi_criteria': len(patterns_found) > 2,
                'scale_dependent': 'scale_indicators' in patterns_found,
                'location_dependent': 'location_indicators' in patterns_found,
                'capability_focused': any(x in patterns_found for x in ['capability_indicators', 'rd_indicators', 'testing_indicators']),
                'certification_required': 'certification_indicators' in patterns_found,
                'industry_specific': 'industry_indicators' in patterns_found,
                'technical_expertise': any(x in patterns_found for x in ['electrical_indicators', 'mechanical_indicators'])
            },
            'suggested_decomposition': self._suggest_decomposition(patterns_found),
            'domain_complexity': self._assess_domain_complexity(patterns_found)
        }
    
    def _assess_domain_complexity(self, patterns: Dict) -> str:
        """Assess the complexity based on number of domains involved."""
        domain_count = len(set(self.industrial_patterns[p]['primary_domain'].split('.')[0] 
                             for p in patterns.keys() 
                             if p in self.industrial_patterns))
        
        if domain_count > 3:
            return "high - multiple domain joins required"
        elif domain_count > 1:
            return "medium - cross-domain relationships needed"
        else:
            return "low - single domain query"
    
    def _suggest_decomposition(self, patterns: Dict) -> str:
        """Suggest decomposition strategy based on detected patterns."""
        if 'scale_indicators' in patterns and 'product_indicators' in patterns:
            return "sequential: CompanyProfile.Classification → CompanyProfile.BasicInfo → ProductsAndServices.Products"
        elif 'company_search_indicators' in patterns and any(x in patterns for x in ['capability_indicators', 'rd_indicators']):
            return "compound: CompanyProfile + BusinessDomain.CoreExpertise + ResearchAndDevelopment"
        elif 'certification_indicators' in patterns and 'testing_indicators' in patterns:
            return "parallel: QualityAndCompliance.Certifications + QualityAndCompliance.TestingCapabilities"
        elif len(patterns) > 2:
            return "sequential: multiple dependent domain filters needed"
        else:
            return "simple: direct domain query execution"
    
    def _get_domain_specific_examples(self) -> str:
        return """
DOMAIN-AWARE QUERY EXAMPLES:

SCALE-BASED PRODUCT LISTING (Sequential with Domain Mapping):
Query: "List all products supplied by medium scale companies"
→ q1: company_scale_definition 
   Domain: CompanyProfile.Classification
   Fields: ScaleMaster.CompanyScale
→ q2: filter_companies_by_scale (depends on q1)
   Domain: CompanyProfile.BasicInfo
   Fields: CompanyMaster.CompanyName, CompanyMaster.CompanyStatus
→ q3: list_company_products (depends on q2)
   Domain: ProductsAndServices.Products
   Fields: CompanyProducts.ProductName, CompanyProducts.ProductDesc, ProductTypeMaster.ProductTypeName

EXPERTISE + CERTIFICATION (Compound with Multiple Domains):
Query: "Companies with electrical expertise and ISO 9001 certification"
→ q1: search_electrical_expertise
   Domain: BusinessDomain.CoreExpertise + ResearchAndDevelopment.RDCapabilities
   Fields: CompanyCoreExpertiseMaster.CoreExpertiseName, RDCategoryMaster.RDCategoryName
→ q2: filter_iso9001_certified
   Domain: QualityAndCompliance.Certifications
   Fields: CertificationTypeMaster.Cert_Type, CompanyCertificationDetail.Certification_Type
→ q3: intersect_results (depends on q1, q2)

LOCATION + TESTING FACILITIES (Sequential with Geographic Filter):
Query: "Companies from Karnataka with electrical testing facilities"
→ q1: filter_karnataka_companies
   Domain: CompanyProfile.ContactInfo
   Fields: CompanyMaster.State, CompanyMaster.CityName
→ q2: filter_electrical_testing (depends on q1)
   Domain: QualityAndCompliance.TestingCapabilities
   Fields: TestFacilityCategoryMaster.CategoryName, TestFacilitySubCategoryMaster.SubCategoryName
"""
    
    def _create_smart_fallback_plan(self, user_query: str) -> QueryPlan:
        """Create an intelligent fallback plan based on pattern analysis with domain mappings."""
        analysis = self._analyze_query_patterns(user_query)
        patterns = analysis['patterns_detected']
        complexity = analysis['complexity_indicators']
        domain_mappings = self._map_query_to_domains(user_query, patterns)
        
        # Determine if decomposition is needed
        if complexity['multi_criteria'] or (complexity['scale_dependent'] and 'product_indicators' in patterns):
            return self._create_decomposed_fallback(user_query, analysis, domain_mappings)
        else:
            return self._create_simple_fallback(user_query, analysis, domain_mappings)
    
    def _create_decomposed_fallback(self, user_query: str, analysis: Dict, domain_mappings: Dict) -> QueryPlan:
        """Create a decomposed fallback plan with proper domain mappings."""
        patterns = analysis['patterns_detected']
        sub_queries = []
        execution_order = []
        
        # Scale-dependent product listing pattern
        if 'scale_indicators' in patterns and 'product_indicators' in patterns:
            # Scale definition query
            scale_domain_mapping = {
                "CompanyProfile.Classification": [
                    {"table": "ScaleMaster", "field": "CompanyScale", "hints": "MSME/Large/Small/Medium", "is_categorical": True}
                ]
            }
            
            sub_queries.append(SubQuery(
                id="scale_definition",
                query="Define criteria for company scale classification (medium scale: employee count, revenue thresholds)",
                type=QueryType.SIMPLE,
                priority=Priority.HIGH,
                dependencies=[],
                context_needed=["company_scale_standards", "ScaleMaster"],
                expected_output="Clear criteria for medium scale company classification",
                reasoning="Establish scale criteria using ScaleMaster table before filtering companies",
                domain_mapping=scale_domain_mapping
            ))
            
            # Company filtering query
            company_domain_mapping = {
                "CompanyProfile.BasicInfo": [
                    {"table": "CompanyMaster", "field": "CompanyName", "hints": "Official name of the company", "is_categorical": True},
                    {"table": "CompanyMaster", "field": "CompanyStatus", "hints": "Active/Incorporated/Dormant/Struck off", "is_categorical": True}
                ],
                "CompanyProfile.Classification": [
                    {"table": "ScaleMaster", "field": "CompanyScale", "hints": "MSME/Large/Small/Medium", "is_categorical": True}
                ]
            }
            
            sub_queries.append(SubQuery(
                id="company_filter",
                query="Filter companies matching medium scale criteria from CompanyMaster joined with ScaleMaster",
                type=QueryType.SIMPLE,
                priority=Priority.HIGH,
                dependencies=["scale_definition"],
                context_needed=["CompanyMaster", "ScaleMaster", "scale_criteria"],
                expected_output="List of medium scale companies with CompanyRefNo and CompanyName",
                reasoning="Filter CompanyMaster records based on ScaleMaster.CompanyScale criteria",
                domain_mapping=company_domain_mapping
            ))
            
            # Product listing query
            product_domain_mapping = {
                "ProductsAndServices.Products": [
                    {"table": "CompanyProducts", "field": "ProductName", "is_categorical": True},
                    {"table": "CompanyProducts", "field": "ProductDesc"},
                    {"table": "ProductTypeMaster", "field": "ProductTypeName", "is_categorical": True}
                ]
            }
            
            sub_queries.append(SubQuery(
                id="product_listing",
                query="List all products from CompanyProducts table for filtered medium scale companies",
                type=QueryType.SIMPLE,
                priority=Priority.MEDIUM,
                dependencies=["company_filter"],
                context_needed=["CompanyProducts", "ProductTypeMaster", "filtered_company_list"],
                expected_output="Comprehensive product list with ProductName, ProductDesc, and ProductTypeName",
                reasoning="Join CompanyProducts with ProductTypeMaster for companies from previous step",
                domain_mapping=product_domain_mapping
            ))
            
            execution_order = ["scale_definition", "company_filter", "product_listing"]
            plan_type = QueryType.SEQUENTIAL
        
        # Multi-criteria company search with different domains
        elif len(patterns) > 2 and 'company_search_indicators' in patterns:
            base_id = 1
            
            for pattern_type, indicators in patterns.items():
                if pattern_type != 'company_search_indicators' and pattern_type in self.industrial_patterns:
                    pattern_config = self.industrial_patterns[pattern_type]
                    primary_domain = pattern_config['primary_domain']
                    
                    # Get domain mapping for this pattern
                    pattern_domain_mapping = {}
                    if primary_domain in domain_mappings:
                        pattern_domain_mapping[primary_domain] = domain_mappings[primary_domain]
                    else:
                        # Fallback to pattern configuration
                        domain_parts = primary_domain.split('.')
                        if len(domain_parts) == 2:
                            category, subcategory = domain_parts
                            if category in self.domain_mapping and subcategory in self.domain_mapping[category]:
                                pattern_domain_mapping[primary_domain] = self.domain_mapping[category][subcategory]
                    
                    sub_queries.append(SubQuery(
                        id=f"filter_{base_id}",
                        query=f"Filter companies by {pattern_type.replace('_indicators', '')} criteria: {indicators}",
                        type=QueryType.SIMPLE,
                        priority=Priority.HIGH,
                        dependencies=[],
                        context_needed=["company_database", f"{pattern_type}_criteria"],
                        expected_output=f"Companies matching {pattern_type} from relevant domain tables",
                        reasoning=f"Apply {pattern_type} filter using {primary_domain} domain tables",
                        domain_mapping=pattern_domain_mapping
                    ))
                    execution_order.append(f"filter_{base_id}")
                    base_id += 1
            
            # Add combination step
            combine_domain_mapping = {
                "CompanyProfile.BasicInfo": [
                    {"table": "CompanyMaster", "field": "CompanyRefNo", "hints": "Company Reference Number", "is_categorical": False},
                    {"table": "CompanyMaster", "field": "CompanyName", "hints": "Official name of the company", "is_categorical": True}
                ]
            }
            
            sub_queries.append(SubQuery(
                id="combine_results",
                query="Intersect company results from all filter criteria using CompanyMaster.CompanyRefNo",
                type=QueryType.SIMPLE,
                priority=Priority.MEDIUM,
                dependencies=execution_order.copy(),
                context_needed=["CompanyMaster", "filtered_company_lists"],
                expected_output="Final deduplicated list of companies meeting all criteria",
                reasoning="Combine multiple domain filter results using CompanyMaster as the base table",
                domain_mapping=combine_domain_mapping
            ))
            execution_order.append("combine_results")
            plan_type = QueryType.COMPOUND
        
        else:
            # Default to simple with enhanced context
            return self._create_simple_fallback(user_query, analysis, domain_mappings)
        
        return QueryPlan(
            original_query=user_query,
            plan_type=plan_type,
            sub_queries=sub_queries,
            execution_order=execution_order,
            reasoning=f"Smart fallback with domain mapping. Detected patterns: {list(patterns.keys())}. Domain complexity: {analysis.get('domain_complexity', 'unknown')}",
            confidence=0.75
        )
    
    def _create_simple_fallback(self, user_query: str, analysis: Dict, domain_mappings: Dict) -> QueryPlan:
        """Create a simple fallback plan with enhanced domain context."""
        patterns = analysis['patterns_detected']
        
        # Build comprehensive domain mapping based on patterns
        if not domain_mappings:
            domain_mappings = {}
            for pattern_type in patterns.keys():
                if pattern_type in self.industrial_patterns:
                    primary_domain = self.industrial_patterns[pattern_type]['primary_domain']
                    domain_parts = primary_domain.split('.')
                    
                    if len(domain_parts) == 2:
                        category, subcategory = domain_parts
                        if category in self.domain_mapping and subcategory in self.domain_mapping[category]:
                            domain_mappings[primary_domain] = self.domain_mapping[category][subcategory]
        
        # Determine context needed based on domain mappings
        context_needed = ["CompanyMaster"]  # Always need base company table
        for domain_fields in domain_mappings.values():
            for field_config in domain_fields:
                table = field_config.get('table', '')
                if table and table not in context_needed:
                    context_needed.append(table)
        
        sub_query = SubQuery(
            id="direct_domain_query",
            query=user_query,
            type=QueryType.SIMPLE,
            priority=Priority.HIGH,
            dependencies=[],
            context_needed=context_needed,
            expected_output="Direct answer using mapped domain tables and fields",
            reasoning=f"Direct execution with domain awareness. Patterns: {list(patterns.keys())}. Tables: {context_needed}",
            domain_mapping=domain_mappings
        )
        
        return QueryPlan(
            original_query=user_query,
            plan_type=QueryType.SIMPLE,
            sub_queries=[sub_query],
            execution_order=["direct_domain_query"],
            reasoning=f"Simple query with comprehensive domain mapping. Domain complexity: {analysis.get('domain_complexity', 'low')}",
            confidence=0.8
        )
    
    def plan_query(self, user_query: str, model_client=None, use_smart_fallback=True) -> QueryPlan:
        """
        Main method to plan and normalize a user query with domain mapping.
        
        Priority order:
        1. Use provided model_client (if given)
        2. Use built-in Qwen client (if available)
        3. Use smart fallback (if enabled)
        4. Use basic fallback
        
        Args:
            user_query: The original user query to analyze
            model_client: Optional external model client instance
            use_smart_fallback: Use intelligent fallback with domain mapping
            f
        Returns:
            QueryPlan object with structured breakdown and domain mappings
        """
        logger.info(f"Planning query: '{user_query}'")
        
        # Determine which model client to use
        active_client = None
        client_source = "none"
        
        if model_client:
            active_client = model_client
            client_source = "provided"
        elif self.qwen_client and self.qwen_client.is_available():
            active_client = self.qwen_client
            client_source = "qwen"
        
        # Try LLM approach first
        if active_client:
            try:
                logger.info(f"🤖 Using {client_source} LLM for query planning")
                
                # Create the full prompt
                full_prompt = f"{self.system_prompt}\n\n{self._create_planning_prompt(user_query)}"
                
                # Generate response from model
                response = active_client.generate(
                    prompt=full_prompt,
                    max_tokens=2048,
                    temperature=0.1,
                    top_p=0.9,
                    stop=["Human:", "Assistant:", "<|im_end|>"]
                )
                
                # Parse JSON response
                plan_data = self._parse_model_response(response)
                
                # Validate and create QueryPlan
                plan = self._create_query_plan(plan_data, user_query)
                logger.info(f"✅ LLM query planning successful (confidence: {plan.confidence})")
                return plan
                
            except Exception as e:
                logger.error(f"❌ LLM query planning failed: {e}")
                if not use_smart_fallback:
                    raise
        
        # Fallback to smart planning
        if use_smart_fallback:
            logger.info("🔧 Using smart fallback for query planning")
            plan = self._create_smart_fallback_plan(user_query)
            logger.info(f"✅ Smart fallback planning successful (confidence: {plan.confidence})")
            return plan
        else:
            logger.info("📝 Using basic fallback for query planning")
            plan = self._create_fallback_plan(user_query)
            logger.info(f"✅ Basic fallback planning successful (confidence: {plan.confidence})")
            return plan
    
    def _create_fallback_plan(self, user_query: str) -> QueryPlan:
        """Create a basic fallback plan when model fails."""
        # Basic domain mapping for fallback
        basic_domain_mapping = {
            "CompanyProfile.BasicInfo": [
                {"table": "CompanyMaster", "field": "CompanyName", "hints": "Official name of the company", "is_categorical": True}
            ]
        }
        
        sub_query = SubQuery(
            id="basic_query",
            query=user_query,
            type=QueryType.SIMPLE,
            priority=Priority.HIGH,
            dependencies=[],
            context_needed=["CompanyMaster"],
            expected_output="Direct answer to user query",
            reasoning="Basic fallback plan - treating as simple query with minimal domain mapping",
            domain_mapping=basic_domain_mapping
        )
        
        return QueryPlan(
            original_query=user_query,
            plan_type=QueryType.SIMPLE,
            sub_queries=[sub_query],
            execution_order=["basic_query"],
            reasoning="Basic fallback plan due to parsing error",
            confidence=0.3
        )
    
    def _parse_model_response(self, response: str) -> dict:
        """Extract and parse JSON from model response."""
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            cleaned = self._clean_json_string(json_str)
            return json.loads(cleaned)
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean up common JSON formatting issues."""
        json_str = re.sub(r'^[^{]*', '', json_str)
        json_str = re.sub(r'[^}]*$', '', json_str)
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        json_str = re.sub(r'(["\w])\s*\n\s*(["\w])', r'\1, \2', json_str)
        return json_str
    
    def _create_query_plan(self, plan_data: dict, original_query: str) -> QueryPlan:
        """Create QueryPlan object from parsed data with domain mappings."""
        sub_queries = []
        
        for sq_data in plan_data.get('sub_queries', []):
            # Parse domain mapping
            domain_mapping = sq_data.get('domain_mapping', {})
            
            sub_query = SubQuery(
                id=sq_data.get('id', f'q{len(sub_queries)+1}'),
                query=sq_data.get('query', ''),
                type=QueryType(sq_data.get('type', 'simple')),
                priority=Priority(sq_data.get('priority', 'medium')),
                dependencies=sq_data.get('dependencies', []),
                context_needed=sq_data.get('context_needed', []),
                expected_output=sq_data.get('expected_output', ''),
                reasoning=sq_data.get('reasoning', ''),
                domain_mapping=domain_mapping
            )
            sub_queries.append(sub_query)
        
        return QueryPlan(
            original_query=original_query,
            plan_type=QueryType(plan_data.get('plan_type', 'simple')),
            sub_queries=sub_queries,
            execution_order=plan_data.get('execution_order', [sq.id for sq in sub_queries]),
            reasoning=plan_data.get('reasoning', ''),
            confidence=float(plan_data.get('confidence', 0.5))
        )
    
    def format_plan_for_execution(self, plan: QueryPlan) -> Dict:
        """Format the plan for easy execution by downstream systems with domain info."""
        return {
            'queries_by_priority': self._group_by_priority(plan),
            'dependency_graph': self._build_dependency_graph(plan),
            'execution_batches': self._create_execution_batches(plan),
            'context_requirements': self._extract_context_requirements(plan),
            'data_sources_needed': self._identify_data_sources(plan),
            'domain_mappings': self._extract_domain_mappings(plan),
            'table_dependencies': self._analyze_table_dependencies(plan),
            'join_requirements': self._identify_join_requirements(plan)
        }
    
    def _extract_domain_mappings(self, plan: QueryPlan) -> Dict[str, Dict]:
        """Extract all domain mappings from the plan."""
        all_mappings = {}
        for sq in plan.sub_queries:
            for domain, fields in sq.domain_mapping.items():
                if domain not in all_mappings:
                    all_mappings[domain] = {}
                for field_config in fields:
                    table = field_config.get('table', '')
                    field = field_config.get('field', '')
                    if table not in all_mappings[domain]:
                        all_mappings[domain][table] = []
                    all_mappings[domain][table].append(field_config)
        return all_mappings
    
    def _analyze_table_dependencies(self, plan: QueryPlan) -> Dict[str, List[str]]:
        """Analyze table dependencies across all sub-queries."""
        table_deps = {}
        
        for sq in plan.sub_queries:
            tables_in_query = []
            for domain_fields in sq.domain_mapping.values():
                for field_config in domain_fields:
                    table = field_config.get('table', '')
                    if table and table not in tables_in_query:
                        tables_in_query.append(table)
            
            table_deps[sq.id] = tables_in_query
        
        return table_deps
    
    def _identify_join_requirements(self, plan: QueryPlan) -> List[Dict]:
        """Identify potential join requirements between tables using relations.json."""
        joins = []
        
        # Get all tables used in the plan
        all_tables = set()
        for sq in plan.sub_queries:
            for domain_fields in sq.domain_mapping.values():
                for field_config in domain_fields:
                    table = field_config.get('table', '')
                    if table:
                        all_tables.add(table)
        
        # Build join requirements from relations.json
        for relation in self.table_relations:
            from_table = relation['from_table']
            to_table = relation['to_table']
            from_column = relation['from_column']
            to_column = relation['to_column']
            
            # Check if both tables are used in the plan
            if from_table in all_tables and to_table in all_tables:
                # Determine join type based on relationship patterns
                join_type = self._determine_join_type(from_table, to_table, from_column)
                
                # Create description based on table names
                description = self._create_join_description(from_table, to_table, from_column, to_column)
                
                joins.append({
                    'tables': [from_table, to_table],
                    'from_column': from_column,
                    'to_column': to_column,
                    'join_type': join_type,
                    'description': description
                })
        
        return joins
    
    def _determine_join_type(self, from_table: str, to_table: str, from_column: str) -> str:
        """Determine the appropriate join type based on table relationship patterns."""
        # Master tables typically use LEFT JOIN when being joined to
        master_tables = ['CompanyMaster', 'ProductTypeMaster', 'CertificationTypeMaster', 
                        'ScaleMaster', 'OrganisationTypeMaster', 'CountryMaster',
                        'IndustryDomainMaster', 'IndustrySubdomainType', 'DefencePlatformMaster',
                        'PlatformTechAreaMaster', 'RDCategoryMaster', 'RDSubCategoryMaster',
                        'TestFacilityCategoryMaster', 'TestFacilitySubCategoryMaster',
                        'CompanyCoreExpertiseMaster']
        
        # If joining from a detail table to a master table, use LEFT JOIN
        if to_table in master_tables:
            return 'LEFT'
        
        # If joining from CompanyMaster to detail tables, use LEFT JOIN (optional relationships)
        if from_table == 'CompanyMaster':
            return 'LEFT'
        
        # For product relationships, use INNER JOIN (products must have a company)
        if from_table == 'CompanyProducts' and to_table == 'CompanyMaster':
            return 'INNER'
        
        # Default to LEFT JOIN for most relationships
        return 'LEFT'
    
    def _create_join_description(self, from_table: str, to_table: str, from_column: str, to_column: str) -> str:
        """Create a human-readable description of the join relationship."""
        # Map table names to more readable descriptions
        table_descriptions = {
            'CompanyMaster': 'Company',
            'CompanyProducts': 'Company Products',
            'CompanyCertificationDetail': 'Company Certifications',
            'CompanyRDFacility': 'Company R&D Facilities',
            'CompanyTestFacility': 'Company Test Facilities',
            'CompanyTurnOver': 'Company Turnover',
            'ProductTypeMaster': 'Product Types',
            'CertificationTypeMaster': 'Certification Types',
            'ScaleMaster': 'Company Scale',
            'OrganisationTypeMaster': 'Organization Types',
            'CountryMaster': 'Countries',
            'IndustryDomainMaster': 'Industry Domains',
            'IndustrySubdomainType': 'Industry Subdomains',
            'DefencePlatformMaster': 'Defence Platforms',
            'PlatformTechAreaMaster': 'Platform Technology Areas',
            'RDCategoryMaster': 'R&D Categories',
            'RDSubCategoryMaster': 'R&D Subcategories',
            'TestFacilityCategoryMaster': 'Test Facility Categories',
            'TestFacilitySubCategoryMaster': 'Test Facility Subcategories',
            'CompanyCoreExpertiseMaster': 'Core Expertise'
        }
        
        from_desc = table_descriptions.get(from_table, from_table)
        to_desc = table_descriptions.get(to_table, to_table)
        
        return f"{from_desc} to {to_desc} relationship via {from_column}->{to_column}"
    
    def _identify_data_sources(self, plan: QueryPlan) -> List[str]:
        """Identify all data sources needed for plan execution."""
        data_sources = set()
        for sq in plan.sub_queries:
            data_sources.update(sq.context_needed)
            # Add tables from domain mappings
            for domain_fields in sq.domain_mapping.values():
                for field_config in domain_fields:
                    table = field_config.get('table', '')
                    if table:
                        data_sources.add(table)
        return list(data_sources)
    
    def _group_by_priority(self, plan: QueryPlan) -> Dict[str, List[SubQuery]]:
        """Group sub-queries by priority level."""
        groups = {p.value: [] for p in Priority}
        for sq in plan.sub_queries:
            groups[sq.priority.value].append(sq)
        return groups
    
    def _build_dependency_graph(self, plan: QueryPlan) -> Dict[str, List[str]]:
        """Build a dependency graph for execution planning."""
        graph = {}
        for sq in plan.sub_queries:
            graph[sq.id] = sq.dependencies
        return graph
    
    def _create_execution_batches(self, plan: QueryPlan) -> List[List[str]]:
        """Create batches of queries that can be executed in parallel."""
        batches = []
        remaining = {sq.id: set(sq.dependencies) for sq in plan.sub_queries}
        completed = set()
        
        while remaining:
            ready = [qid for qid, deps in remaining.items() if not deps]
            if not ready:
                ready = list(remaining.keys())
            
            batches.append(ready)
            
            for qid in ready:
                completed.add(qid)
                del remaining[qid]
            
            for deps in remaining.values():
                deps -= completed
        
        return batches
    
    def _extract_context_requirements(self, plan: QueryPlan) -> Dict[str, List[str]]:
        """Extract context requirements for each query."""
        return {sq.id: sq.context_needed for sq in plan.sub_queries}
    
    def get_domain_summary(self, plan: QueryPlan) -> Dict:
        """Get a summary of domains and fields used in the plan."""
        domain_summary = {}
        
        for sq in plan.sub_queries:
            for domain, field_configs in sq.domain_mapping.items():
                if domain not in domain_summary:
                    domain_summary[domain] = {
                        'tables': set(),
                        'fields': set(),
                        'categorical_fields': set(),
                        'queries_using': []
                    }
                
                domain_summary[domain]['queries_using'].append(sq.id)
                
                for field_config in field_configs:
                    table = field_config.get('table', '')
                    field = field_config.get('field', '')
                    
                    if table:
                        domain_summary[domain]['tables'].add(table)
                    if field:
                        domain_summary[domain]['fields'].add(field)
                        if field_config.get('is_categorical', False):
                            domain_summary[domain]['categorical_fields'].add(field)
        
        # Convert sets to lists for JSON serialization
        for domain_info in domain_summary.values():
            domain_info['tables'] = list(domain_info['tables'])
            domain_info['fields'] = list(domain_info['fields'])
            domain_info['categorical_fields'] = list(domain_info['categorical_fields'])
        
        return domain_summary

# Test the enhanced planner with domain mapping
def test_enhanced_planner():
    """Test the enhanced industrial query planner with domain mapping."""
    logger.info("Starting Enhanced QueryPlanner with Domain Mapping test...")
    
    config, logger = load_config()

    planner = IndustrialQueryPlanner(config)
    
    test_queries = [
        #"give me list of all products which are supplied by medium scale companies",
        "give me list of all products which are supplied by small scale companies",
        #"Show me the companies who have expertise in electrical and also have ISO 9001 certificate",
        #"List companies having R&D in electrical and also having test facility on electrical",
        #"Companies from Karnataka with manufacturing capabilities and NABL accreditation",
        #"Show all defence platforms and their associated technology areas"
    ]
    
    for query in test_queries:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing query: {query}")
        logger.info(f"{'='*80}")
        
        # Test with smart fallback
        plan = planner.plan_query(query, use_smart_fallback=True)
        
        logger.info(f"Plan Type: {plan.plan_type.value}")
        logger.info(f"Confidence: {plan.confidence}")
        logger.info(f"Reasoning: {plan.reasoning}")
        logger.info(f"Sub-queries: {len(plan.sub_queries)}")
        
        for sq in plan.sub_queries:
            logger.info(f"\n--- Sub-query: {sq.id} ---")
            logger.info(f"Query: {sq.query}")
            logger.info(f"Type: {sq.type.value}, Priority: {sq.priority.value}")
            if sq.dependencies:
                logger.info(f"Dependencies: {sq.dependencies}")
            if sq.context_needed:
                logger.info(f"Context: {sq.context_needed}")
            
            # Show domain mappings
            if sq.domain_mapping:
                logger.info("Domain Mappings:")
                for domain, fields in sq.domain_mapping.items():
                    logger.info(f"  {domain}:")
                    for field_config in fields:
                        table = field_config.get('table', '')
                        field = field_config.get('field', '')
                        hints = field_config.get('hints', '')
                        is_cat = field_config.get('is_categorical', False)
                        logger.info(f"    {table}.{field} ({'categorical' if is_cat else 'non-categorical'}) - {hints}")
        
        # Show execution format
        execution_format = planner.format_plan_for_execution(plan)
        logger.info(f"\nExecution Batches: {execution_format['execution_batches']}")
        logger.info(f"Tables Needed: {execution_format['data_sources_needed']}")
        
        # Show domain summary
        domain_summary = planner.get_domain_summary(plan)
        logger.info("\nDomain Summary:")
        for domain, info in domain_summary.items():
            logger.info(f"  {domain}: {len(info['tables'])} tables, {len(info['fields'])} fields")
            logger.info(f"    Tables: {info['tables']}")
            logger.info(f"    Categorical fields: {info['categorical_fields']}")
        
        # Show join requirements
        if execution_format['join_requirements']:
            logger.info("\nJoin Requirements:")
            for join in execution_format['join_requirements']:
                logger.info(f"  {join['tables'][0]} {join['join_type']} JOIN {join['tables'][1]} ON {join['from_column']} = {join['to_column']}")
                logger.info(f"    Description: {join['description']}")
    
    logger.info("\nEnhanced QueryPlanner with Domain Mapping test completed successfully")

def test_query_to_api_integration():
    """Test integration between IndustrialQueryPlanner and CompanySearchAPI."""
    logger.info("\n🔗 Testing Query Planner to API Integration")
    logger.info("=" * 60)
    
    try:
        # Import CompanySearchAPI
        from src.company_index.company_search_api import CompanySearchAPI
        
        # Initialize components
        config, logger = load_config()
        planner = IndustrialQueryPlanner(config)
        api = CompanySearchAPI()
        
        # Test query
        user_query = "give me list of all products which are supplied by small scale companies"
        
        logger.info(f"Original Query: '{user_query}'")
        logger.info("-" * 60)
        
        # Method 1: Direct integration - automatically uses Qwen LLM first, then smart fallback
        logger.info("🔧 Method 1: Automatic Query Plan Execution (Qwen → Smart Fallback)")
        plan = planner.plan_query(user_query, use_smart_fallback=True)
        
        logger.info(f"Generated plan with {len(plan.sub_queries)} steps:")
        for i, sq in enumerate(plan.sub_queries, 1):
            logger.info(f"  Step {i}: {sq.id}")
            logger.info(f"    Query: {sq.query}")
            logger.info(f"    Dependencies: {sq.dependencies}")
            
            # For demonstration, show how you would execute each step
            if sq.id == "scale_definition":
                logger.info("    → Would define scale criteria (metadata query)")
            elif sq.id == "company_filter":
                logger.info("    → Would search for medium scale companies")
                # Example: results = api.search("medium scale companies", filter_scale="Medium")
            elif sq.id == "product_listing":
                logger.info("    → Would list products for filtered companies")
                # Example: results = api.search("products", company_ids=filtered_company_ids)
        
        # Method 2: Using intelligent_search (automatic execution)
        logger.info("\n🤖 Method 2: Automatic Intelligent Search")
        try:
            # Check if API has intelligent_search method
            if hasattr(api, 'intelligent_search'):
                logger.info("Using CompanySearchAPI.intelligent_search() - automatic query planning")
                # This would automatically use the query planner internally
                # intelligent_results = api.intelligent_search(user_query, top_k=5)
                logger.info("✓ intelligent_search method available")
            else:
                logger.info("intelligent_search method not available, using manual approach")
        except Exception as e:
            logger.warning(f"Error testing intelligent_search: {e}")
        
        # Method 3: Step-by-step execution example
        logger.info("\n📋 Method 3: Step-by-Step Execution Example")
        
        # Execute each sub-query based on its type and dependencies
        execution_results = {}
        
        for sq in plan.sub_queries:
            logger.info(f"\nExecuting: {sq.id}")
            
            if sq.id == "scale_definition":
                # This would be a metadata/configuration step
                execution_results[sq.id] = {
                    "type": "metadata",
                    "criteria": "Medium scale: 50-250 employees, 10M-100M revenue"
                }
                logger.info("  ✓ Scale criteria defined")
                
            elif sq.id == "company_filter":
                # Real search for companies matching scale criteria
                try:
                    # Dynamically determine scale from the original query
                    scale_filter = None
                    query_lower = user_query.lower()
                    
                    if "small scale" in query_lower:
                        scale_filter = "Small"
                        search_query = "small scale companies"
                    elif "medium scale" in query_lower:
                        scale_filter = "Medium"
                        search_query = "medium scale companies"
                    elif "large scale" in query_lower:
                        scale_filter = "Large"
                        search_query = "large scale companies"
                    else:
                        # Default fallback - try to extract scale from patterns
                        if any(word in query_lower for word in ["small", "sme", "msme"]):
                            scale_filter = "Small"
                            search_query = "small scale companies"
                        elif any(word in query_lower for word in ["medium", "mid-size"]):
                            scale_filter = "Medium"
                            search_query = "medium scale companies"
                        elif any(word in query_lower for word in ["large", "big", "enterprise", "multinational"]):
                            scale_filter = "Large"
                            search_query = "large scale companies"
                        else:
                            # No specific scale mentioned, search all companies
                            scale_filter = None
                            search_query = "companies"
                    
                    # Search for companies matching the detected scale criteria
                    results = api.search(search_query, filter_scale=scale_filter, top_k=50)
                    execution_results[sq.id] = {
                        "type": "company_search",
                        "query": search_query,
                        "scale_filter": scale_filter,
                        "results": results,
                        "count": len(results)
                    }
                    scale_desc = f"{scale_filter} scale" if scale_filter else "all scale"
                    logger.info(f"  ✓ Found {len(results)} {scale_desc} companies")
                    
                    # Show sample results
                    if results:
                        logger.info("  Sample companies:")
                        for i, company in enumerate(results[:3], 1):
                            company_name = company.get('company_name', 'Unknown')
                            scale = company.get('company_scale', 'Unknown')
                            logger.info(f"    {i}. {company_name} (Scale: {scale})")
                            
                except Exception as e:
                    logger.warning(f"  ⚠ Company search failed: {e}")
                    # Fallback to empty results with dynamic scale info
                    execution_results[sq.id] = {
                        "type": "company_search",
                        "query": search_query if 'search_query' in locals() else "companies",
                        "scale_filter": scale_filter if 'scale_filter' in locals() else None,
                        "results": [],
                        "count": 0,
                        "error": str(e)
                    }
                    
            elif sq.id == "product_listing":
                # Real search for products from filtered companies
                try:
                    # Get company results from previous step
                    company_results = execution_results.get("company_filter", {}).get("results", [])
                    
                    if company_results:
                        # Search for products from these companies
                        product_results = api.search("products", top_k=100)
                        
                        # Filter products by companies found in previous step (simplified approach)
                        company_names = {c.get('company_name', '').lower() for c in company_results}
                        filtered_products = []
                        
                        for product in product_results:
                            product_company = product.get('company_name', '').lower()
                            if product_company in company_names:
                                filtered_products.append(product)
                        
                        execution_results[sq.id] = {
                            "type": "product_search",
                            "query": "products from medium scale companies",
                            "results": filtered_products,
                            "count": len(filtered_products)
                        }
                        logger.info(f"  ✓ Found {len(filtered_products)} products from medium scale companies")
                        
                        # Show sample products
                        if filtered_products:
                            logger.info("  Sample products:")
                            for i, product in enumerate(filtered_products[:3], 1):
                                product_name = product.get('product_name', 'Unknown')
                                company_name = product.get('company_name', 'Unknown')
                                logger.info(f"    {i}. {product_name} (by {company_name})")
                    else:
                        logger.info("  ⚠ No companies found in previous step, skipping product search")
                        execution_results[sq.id] = {
                            "type": "product_search",
                            "query": "products from medium scale companies",
                            "results": [],
                            "count": 0,
                            "note": "No companies found in previous step"
                        }
                        
                except Exception as e:
                    logger.warning(f"  ⚠ Product search failed: {e}")
                    execution_results[sq.id] = {
                        "type": "product_search",
                        "query": "products from medium scale companies",
                        "results": [],
                        "count": 0,
                        "error": str(e)
                    }
        
        # Show final aggregated results
        logger.info("\n📊 Final Results Summary:")
        total_products = execution_results.get("product_listing", {}).get("count", 0)
        total_companies = execution_results.get("company_filter", {}).get("count", 0)
        
        logger.info(f"  Companies filtered: {total_companies}")
        logger.info(f"  Products found: {total_products}")
        logger.info(f"  Execution plan: {plan.plan_type.value}")
        logger.info(f"  Confidence: {plan.confidence}")
        
        # Show detailed results if available
        company_results = execution_results.get("company_filter", {}).get("results", [])
        product_results = execution_results.get("product_listing", {}).get("results", [])
        
        if company_results:
            logger.info(f"\n📋 Company Results ({len(company_results)} found):")
            for i, company in enumerate(company_results[:5], 1):  # Show top 5
                name = company.get('company_name', 'Unknown')
                scale = company.get('company_scale', 'Unknown')
                location = company.get('city', company.get('state', 'Unknown'))
                logger.info(f"  {i}. {name} | Scale: {scale} | Location: {location}")
            
            if len(company_results) > 5:
                logger.info(f"  ... and {len(company_results) - 5} more companies")
        
        if product_results:
            logger.info(f"\n🛠️ Product Results ({len(product_results)} found):")
            for i, product in enumerate(product_results[:5], 1):  # Show top 5
                name = product.get('product_name', 'Unknown')
                company = product.get('company_name', 'Unknown')
                product_type = product.get('product_type', 'Unknown')
                logger.info(f"  {i}. {name} | Company: {company} | Type: {product_type}")
            
            if len(product_results) > 5:
                logger.info(f"  ... and {len(product_results) - 5} more products")
        
        # Show how to format results for API response
        logger.info("\n🎯 API Response Format:")
        api_response = {
            "query": user_query,
            "plan_used": {
                "type": plan.plan_type.value,
                "steps": len(plan.sub_queries),
                "confidence": plan.confidence
            },
            "execution_steps": list(execution_results.keys()),
            "results": {
                "companies_count": total_companies,
                "products_count": total_products,
                "companies": company_results[:3] if company_results else [],  # Sample companies
                "products": product_results[:3] if product_results else [],   # Sample products
                "execution_time": "real_data"
            }
        }
        
        # Pretty format the API response for better readability
        try:
            if isinstance(api_response, dict):
                formatted_response = json.dumps(api_response, indent=2, ensure_ascii=False)
                logger.info(f"  Response:\n{formatted_response}")
            elif isinstance(api_response, str):
                try:
                    # Try to parse as JSON and pretty print
                    parsed_json = json.loads(api_response)
                    formatted_response = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                    logger.info(f"  Response:\n{formatted_response}")
                except json.JSONDecodeError:
                    # If not JSON, just log as is but with better formatting
                    logger.info(f"  Response: {api_response}")
            else:
                logger.info(f"  Response: {api_response}")
        except Exception as e:
            # Fallback to original format if anything goes wrong
            logger.info(f"  Response: {api_response}")
            logger.debug(f"Error formatting response: {e}")
        
        logger.info("\n✅ Query Planner to API Integration test completed successfully!")
        return True
        
    except ImportError as e:
        logger.warning(f"CompanySearchAPI not available for integration test: {e}")
        logger.info("This is normal if the company search API is not set up yet")
        return False
    except Exception as e:
        logger.error(f"Error in integration test: {e}")
        return False

def execute_query_with_steps(user_query: str):
    """
    Execute query following the 4-step process:
    Step-1: Use LLM (Qwen) to break user query into multiple parts
    Step-2: If LLM fails, use smart fallback
    Step-3: Output sub-queries and relations (sequential/parallel)
    Step-4: Execute in order and get results
    """
    logger.info("🚀 Starting 4-Step Query Execution Process")
    logger.info("=" * 80)
    logger.info(f"User Query: '{user_query}'")
    logger.info("=" * 80)
    
    try:
        # Initialize components
        from src.company_index.company_search_api import CompanySearchAPI
        config, logger = load_config()
        planner = IndustrialQueryPlanner(config)
        api = CompanySearchAPI()
        
        # STEP 1: Use LLM (Qwen) to break user query
        logger.info("\n📋 STEP 1: LLM-Based Query Decomposition")
        logger.info("-" * 50)
        
        plan = None
        llm_success = False
        
        # Try Qwen LLM first
        if planner.qwen_client and planner.qwen_client.is_available():
            try:
                logger.info("🤖 Using Qwen LLM to break down the query...")
                plan = planner.plan_query(user_query, use_smart_fallback=False)
                llm_success = True
                logger.info("✅ STEP 1 SUCCESS: LLM successfully broke down the query")
                logger.info(f"   Plan Type: {plan.plan_type.value}")
                logger.info(f"   Sub-queries: {len(plan.sub_queries)}")
                logger.info(f"   Confidence: {plan.confidence}")
            except Exception as e:
                logger.error(f"❌ STEP 1 FAILED: LLM decomposition failed - {e}")
                llm_success = False
        else:
            logger.warning("⚠️ STEP 1 SKIPPED: Qwen LLM not available")
            llm_success = False
        
        # STEP 2: If LLM fails, use smart fallback
        if not llm_success:
            logger.info("\n🔧 STEP 2: Smart Fallback Query Decomposition")
            logger.info("-" * 50)
            try:
                logger.info("🧠 Using smart fallback to break down the query...")
                plan = planner._create_smart_fallback_plan(user_query)
                logger.info("✅ STEP 2 SUCCESS: Smart fallback created query plan")
                logger.info(f"   Plan Type: {plan.plan_type.value}")
                logger.info(f"   Sub-queries: {len(plan.sub_queries)}")
                logger.info(f"   Confidence: {plan.confidence}")
            except Exception as e:
                logger.error(f"❌ STEP 2 FAILED: Smart fallback failed - {e}")
                raise
        else:
            logger.info("\n✅ STEP 2 SKIPPED: LLM was successful")
        
        # STEP 3: Output sub-queries and relations
        logger.info("\n📊 STEP 3: Query Plan Analysis")
        logger.info("-" * 50)
        
        logger.info(f"Query Decomposition Results:")
        logger.info(f"  Original Query: {plan.original_query}")
        logger.info(f"  Plan Type: {plan.plan_type.value}")
        logger.info(f"  Execution Strategy: {'Sequential' if plan.plan_type == QueryType.SEQUENTIAL else 'Parallel' if plan.plan_type == QueryType.PARALLEL else 'Simple'}")
        logger.info(f"  Total Sub-queries: {len(plan.sub_queries)}")
        logger.info(f"  Execution Order: {plan.execution_order}")
        logger.info(f"  Confidence Score: {plan.confidence}")
        
        logger.info("\nSub-query Breakdown:")
        for i, sq in enumerate(plan.sub_queries, 1):
            logger.info(f"  {i}. ID: {sq.id}")
            logger.info(f"     Query: {sq.query}")
            logger.info(f"     Type: {sq.type.value}")
            logger.info(f"     Priority: {sq.priority.value}")
            logger.info(f"     Dependencies: {sq.dependencies if sq.dependencies else 'None'}")
            logger.info(f"     Expected Output: {sq.expected_output}")
        
        logger.info("\nRelationship Analysis:")
        if plan.plan_type == QueryType.SEQUENTIAL:
            logger.info("  🔗 SEQUENTIAL: Sub-queries must be executed in order")
            for i, step_id in enumerate(plan.execution_order, 1):
                logger.info(f"     Step {i}: {step_id}")
        elif plan.plan_type == QueryType.PARALLEL:
            logger.info("  ⚡ PARALLEL: Sub-queries can be executed simultaneously")
        else:
            logger.info("  📝 SIMPLE: Single query execution")
        
        logger.info("✅ STEP 3 SUCCESS: Query plan analyzed and relationships identified")
        
        # STEP 4: Execute in order and get results
        logger.info("\n🎯 STEP 4: Sequential Execution and Results")
        logger.info("-" * 50)
        
        execution_results = {}
        total_companies = 0
        total_products = 0
        
        for step_num, sq in enumerate(plan.sub_queries, 1):
            logger.info(f"\n🔄 Executing Step {step_num}: {sq.id}")
            logger.info(f"   Query: {sq.query}")
            
            if sq.dependencies:
                logger.info(f"   Waiting for dependencies: {sq.dependencies}")
                # Check if dependencies are completed
                missing_deps = [dep for dep in sq.dependencies if dep not in execution_results]
                if missing_deps:
                    logger.warning(f"   ⚠️ Missing dependencies: {missing_deps}")
            
            try:
                if sq.id == "scale_definition":
                    logger.info("   📋 Defining scale criteria...")
                    execution_results[sq.id] = {
                        "type": "metadata",
                        "criteria": "Scale classification defined",
                        "status": "completed"
                    }
                    logger.info("   ✅ Scale criteria defined successfully")
                    
                elif sq.id == "company_filter":
                    logger.info("   🏢 Filtering companies by scale...")
                    
                    # Dynamic scale detection
                    scale_filter = None
                    query_lower = user_query.lower()
                    
                    if "small scale" in query_lower:
                        scale_filter = "Small"
                        search_query = "small scale companies"
                    elif "medium scale" in query_lower:
                        scale_filter = "Medium"
                        search_query = "medium scale companies"
                    elif "large scale" in query_lower:
                        scale_filter = "Large"
                        search_query = "large scale companies"
                    else:
                        if any(word in query_lower for word in ["small", "sme", "msme"]):
                            scale_filter = "Small"
                            search_query = "small scale companies"
                        elif any(word in query_lower for word in ["medium", "mid-size"]):
                            scale_filter = "Medium"
                            search_query = "medium scale companies"
                        elif any(word in query_lower for word in ["large", "big", "enterprise"]):
                            scale_filter = "Large"
                            search_query = "large scale companies"
                        else:
                            scale_filter = None
                            search_query = "companies"
                    
                    logger.info(f"   🎯 Detected scale: {scale_filter}")
                    logger.info(f"   🔍 Search query: {search_query}")
                    
                    # Execute company search
                    results = api.search(search_query, filter_scale=scale_filter, top_k=50)
                    total_companies = len(results)
                    
                    execution_results[sq.id] = {
                        "type": "company_search",
                        "query": search_query,
                        "scale_filter": scale_filter,
                        "results": results,
                        "count": total_companies,
                        "status": "completed"
                    }
                    
                    logger.info(f"   ✅ Found {total_companies} {scale_filter or 'all'} scale companies")
                    
                    # Show sample results
                    if results:
                        logger.info("   📋 Sample companies:")
                        for i, company in enumerate(results[:3], 1):
                            name = company.get('company_name', 'Unknown')
                            scale = company.get('scale', 'Unknown')
                            location = company.get('location', 'Unknown')
                            logger.info(f"      {i}. {name} | Scale: {scale} | Location: {location}")
                    
                elif sq.id == "product_listing":
                    logger.info("   🛠️ Listing products from filtered companies...")
                    
                    # Get company results from previous step
                    company_results = execution_results.get("company_filter", {}).get("results", [])
                    
                    if company_results:
                        # Search for products
                        product_results = api.search("products", top_k=100)
                        
                        # Filter products by companies (simplified approach)
                        company_names = {c.get('company_name', '').lower() for c in company_results}
                        filtered_products = []
                        
                        for product in product_results:
                            product_company = product.get('company_name', '').lower()
                            if product_company in company_names:
                                filtered_products.append(product)
                        
                        total_products = len(filtered_products)
                        
                        execution_results[sq.id] = {
                            "type": "product_search",
                            "results": filtered_products,
                            "count": total_products,
                            "status": "completed"
                        }
                        
                        logger.info(f"   ✅ Found {total_products} products from filtered companies")
                        
                        # Show sample products
                        if filtered_products:
                            logger.info("   📋 Sample products:")
                            for i, product in enumerate(filtered_products[:3], 1):
                                name = product.get('product_name', 'Unknown')
                                company = product.get('company_name', 'Unknown')
                                logger.info(f"      {i}. {name} | Company: {company}")
                    else:
                        logger.warning("   ⚠️ No companies found in previous step")
                        execution_results[sq.id] = {
                            "type": "product_search",
                            "results": [],
                            "count": 0,
                            "status": "skipped",
                            "reason": "No companies found"
                        }
                
                else:
                    # Generic execution for other query types
                    logger.info(f"   🔍 Executing generic query: {sq.query}")
                    results = api.search(sq.query, top_k=20)
                    execution_results[sq.id] = {
                        "type": "generic_search",
                        "results": results,
                        "count": len(results),
                        "status": "completed"
                    }
                    logger.info(f"   ✅ Found {len(results)} results")
                
            except Exception as e:
                logger.error(f"   ❌ Step {step_num} failed: {e}")
                execution_results[sq.id] = {
                    "type": "error",
                    "error": str(e),
                    "status": "failed"
                }
        
        # Final Results Summary
        logger.info("\n🎉 STEP 4 COMPLETED: Final Results Summary")
        logger.info("-" * 50)
        
        logger.info(f"Execution Summary:")
        logger.info(f"  Total Steps Executed: {len(execution_results)}")
        logger.info(f"  Companies Found: {total_companies}")
        logger.info(f"  Products Found: {total_products}")
        logger.info(f"  Plan Type: {plan.plan_type.value}")
        logger.info(f"  Overall Success: {'✅ Yes' if total_products > 0 or total_companies > 0 else '⚠️ Partial'}")
        
        # Create final API response
        final_response = {
            "query": user_query,
            "execution_method": "LLM" if llm_success else "Smart Fallback",
            "plan_used": {
                "type": plan.plan_type.value,
                "steps": len(plan.sub_queries),
                "confidence": plan.confidence
            },
            "execution_steps": list(execution_results.keys()),
            "results": {
                "companies_count": total_companies,
                "products_count": total_products,
                "companies": execution_results.get("company_filter", {}).get("results", [])[:5],
                "products": execution_results.get("product_listing", {}).get("results", [])[:5],
                "execution_time": "real_data"
            },
            "step_details": execution_results
        }
        
        # Pretty print final response
        logger.info("\n📄 Final API Response:")
        try:
            formatted_response = json.dumps(final_response, indent=2, ensure_ascii=False)
            logger.info(f"{formatted_response}")
        except Exception as e:
            logger.info(f"{final_response}")
            logger.debug(f"Error formatting response: {e}")
        
        logger.info("\n🏁 4-Step Query Execution Process Completed Successfully!")
        return final_response
        
    except ImportError as e:
        logger.error(f"❌ CompanySearchAPI not available: {e}")
        return {"error": "CompanySearchAPI not available", "details": str(e)}
    except Exception as e:
        logger.error(f"❌ Execution failed: {e}")
        return {"error": "Execution failed", "details": str(e)}

if __name__ == "__main__":
    # Test both original and optimized approaches
    test_query = "give me list of all products which are supplied by small scale companies"
    
    logger.info("🔄 Testing Original 4-Step Process:")
    logger.info("=" * 80)
    execute_query_with_steps(test_query)
    
    logger.info("\n\n🚀 Testing Optimized Execution:")
    logger.info("=" * 80)
    try:
        from src.optimized_query_executor import execute_optimized_query
        result = execute_optimized_query(test_query)
        
        logger.info("📊 OPTIMIZED EXECUTION COMPLETED:")
        logger.info(f"   Strategy: {result.get('strategy_used', 'Unknown')}")
        logger.info(f"   Method: {result.get('execution_method', 'Unknown')}")
        logger.info(f"   Time: {result.get('execution_time', 0):.2f}s")
        logger.info(f"   Success: {result.get('success', False)}")
        
        if result.get('results'):
            results = result['results']
            logger.info(f"   Companies: {results.get('companies_count', 0)}")
            logger.info(f"   Products: {results.get('products_count', 0)}")
            
    except ImportError:
        logger.warning("Optimized executor not available")
    except Exception as e:
        logger.error(f"Optimized execution failed: {e}")

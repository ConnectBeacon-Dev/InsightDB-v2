#!/usr/bin/env python3
"""
Local LLM Query Enhancer for InsightDB-v2
Uses locally installed Qwen2.5-14B-Instruct model for query understanding and enhancement
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Optional LLM dependencies
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

@dataclass
class QueryEnhancement:
    """Result of LLM query enhancement"""
    original_query: str
    enhanced_query: str
    extracted_entities: Dict[str, List[str]]
    suggested_synonyms: List[str]
    confidence: float
    reasoning: str
    fallback_used: bool = False

class LocalLLMQueryEnhancer:
    """
    Local LLM-powered query enhancement using Qwen2.5-14B-Instruct model
    Provides query understanding, entity extraction, and query expansion
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = False
        self.llm = None
        
        # Model path
        self.model_path = Path("models/Qwen2.5-14B-Instruct-Q4_K_M.gguf")
        
        # Initialize LLM
        self._initialize_llm()
        
        # Entity patterns for fallback
        self.entity_patterns = {
            'location': r'\b(?:in|at|from|near|located)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            'certification': r'\b(ISO\s*\d+|NABL|certified?|certification)\b',
            'scale': r'\b(small|medium|large|micro|enterprise)\s*(?:scale)?\b',
            'domain': r'\b(aerospace|manufacturing|telecom|electrical|software|mechanical)\b',
            'capability': r'\b(R&D|research|testing|development|capabilities?)\b'
        }
        
        # Business domain synonyms
        self.domain_synonyms = {
            'electrical': ['power', 'voltage', 'energy', 'electric'],
            'aerospace': ['aviation', 'aircraft', 'satellite', 'space'],
            'manufacturing': ['production', 'factory', 'industrial'],
            'software': ['IT', 'technology', 'digital', 'computing'],
            'mechanical': ['engineering', 'machinery', 'equipment'],
            'telecom': ['telecommunications', 'communication', '5G', 'network']
        }

    def _initialize_llm(self):
        """Initialize the local LLM model"""
        if not LLAMA_CPP_AVAILABLE:
            self.logger.warning("[LLM] llama-cpp-python not available. Install with: pip install llama-cpp-python")
            return
            
        if not self.model_path.exists():
            self.logger.warning(f"[LLM] Model not found at {self.model_path}")
            return
            
        try:
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=2048,  # Context window
                n_threads=4,  # CPU threads
                verbose=False
            )
            self.enabled = True
            self.logger.info(f"[LLM] Initialized Qwen2.5-14B-Instruct model from {self.model_path}")
        except Exception as e:
            self.logger.warning(f"[LLM] Failed to initialize model: {e}")
            self.enabled = False

    def enhance_query(self, query: str) -> QueryEnhancement:
        """
        Enhance a search query using LLM understanding
        Falls back to rule-based enhancement if LLM is unavailable
        """
        if self.enabled and self.llm:
            return self._llm_enhance_query(query)
        else:
            return self._fallback_enhance_query(query)

    def _llm_enhance_query(self, query: str) -> QueryEnhancement:
        """Use LLM to enhance the query"""
        try:
            prompt = self._build_enhancement_prompt(query)
            
            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.1,
                top_p=0.9,
                stop=["</response>", "\n\n"]
            )
            
            result_text = response['choices'][0]['text'].strip()
            return self._parse_llm_response(query, result_text)
            
        except Exception as e:
            self.logger.warning(f"[LLM] Enhancement failed: {e}, falling back to rule-based")
            return self._fallback_enhance_query(query)

    def _build_enhancement_prompt(self, query: str) -> str:
        """Build the prompt for query enhancement"""
        return f"""You are a business search query analyzer. Your task is to understand and enhance search queries for a company database.

The database contains companies with the following attributes:
- Company details: name, location (country, state, city), scale (small, medium, large, enterprise)
- Business domains: aerospace, manufacturing, telecom, electrical, software, mechanical
- Capabilities: R&D, testing facilities, certifications (ISO standards)
- Products and services

Analyze this query and provide enhancements:
Query: "{query}"

Provide your analysis in this JSON format:
{{
    "enhanced_query": "improved version of the query with better keywords",
    "entities": {{
        "location": ["extracted locations"],
        "domain": ["business domains"],
        "scale": ["company scales"],
        "certification": ["certifications mentioned"],
        "capability": ["capabilities mentioned"]
    }},
    "synonyms": ["alternative terms that could help find relevant companies"],
    "confidence": 0.8,
    "reasoning": "brief explanation of the enhancements made"
}}

Response:"""

    def _parse_llm_response(self, original_query: str, response_text: str) -> QueryEnhancement:
        """Parse LLM response into QueryEnhancement object"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                return QueryEnhancement(
                    original_query=original_query,
                    enhanced_query=result.get('enhanced_query', original_query),
                    extracted_entities=result.get('entities', {}),
                    suggested_synonyms=result.get('synonyms', []),
                    confidence=float(result.get('confidence', 0.5)),
                    reasoning=result.get('reasoning', 'LLM analysis completed'),
                    fallback_used=False
                )
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            self.logger.warning(f"[LLM] Failed to parse response: {e}")
            return self._fallback_enhance_query(original_query)

    def _fallback_enhance_query(self, query: str) -> QueryEnhancement:
        """Rule-based query enhancement when LLM is unavailable"""
        query_lower = query.lower()
        entities = {}
        synonyms = []
        enhanced_parts = [query]
        
        # Extract entities using regex patterns
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        
        # Add domain synonyms
        for domain, domain_synonyms in self.domain_synonyms.items():
            if domain in query_lower:
                synonyms.extend(domain_synonyms)
                # Add synonyms to enhanced query
                enhanced_parts.extend(domain_synonyms[:2])  # Add top 2 synonyms
        
        # Handle common query patterns
        if 'small' in query_lower and 'scale' not in query_lower:
            enhanced_parts.append('micro scale')
        
        if 'iso' in query_lower and 'certification' not in query_lower:
            enhanced_parts.append('certified companies')
        
        if 'testing' in query_lower and 'facilities' not in query_lower:
            enhanced_parts.append('test facilities laboratory')
        
        enhanced_query = ' '.join(enhanced_parts)
        
        return QueryEnhancement(
            original_query=query,
            enhanced_query=enhanced_query,
            extracted_entities=entities,
            suggested_synonyms=synonyms,
            confidence=0.6,  # Medium confidence for rule-based
            reasoning="Rule-based enhancement using domain patterns",
            fallback_used=True
        )

    def extract_search_intent(self, query: str) -> Dict[str, Any]:
        """Extract search intent from query"""
        enhancement = self.enhance_query(query)
        
        # Determine primary intent based on entities
        intents = []
        entities = enhancement.extracted_entities
        
        if entities.get('location'):
            intents.append('location')
        if entities.get('certification'):
            intents.append('certifications')
        if entities.get('capability'):
            if 'testing' in str(entities['capability']).lower():
                intents.append('testing_capabilities')
            if 'r&d' in str(entities['capability']).lower() or 'research' in str(entities['capability']).lower():
                intents.append('rd_capabilities')
        if entities.get('domain'):
            intents.append('business_domain')
        if entities.get('scale'):
            intents.append('company_scale')
        
        # Default intent if none detected
        if not intents:
            intents = ['basic_info', 'products_services']
        
        return {
            'intents': intents,
            'entities': entities,
            'enhanced_query': enhancement.enhanced_query,
            'confidence': enhancement.confidence,
            'reasoning': enhancement.reasoning,
            'llm_used': not enhancement.fallback_used
        }

    def is_available(self) -> bool:
        """Check if LLM enhancement is available"""
        return self.enabled

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the LLM enhancer"""
        return {
            'enabled': self.enabled,
            'model_path': str(self.model_path),
            'model_exists': self.model_path.exists(),
            'llama_cpp_available': LLAMA_CPP_AVAILABLE,
            'fallback_available': True
        }


# Factory function for easy integration
def create_llm_enhancer(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> LocalLLMQueryEnhancer:
    """Create and return an LLM query enhancer instance"""
    return LocalLLMQueryEnhancer(config, logger)


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test the enhancer
    enhancer = LocalLLMQueryEnhancer({}, logger)
    
    test_queries = [
        "small scale manufacturing companies",
        "companies with ISO certification",
        "electrical R&D capabilities in Sweden",
        "aerospace companies with testing facilities"
    ]
    
    print("LLM Query Enhancer Test Results:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nOriginal Query: {query}")
        enhancement = enhancer.enhance_query(query)
        print(f"Enhanced Query: {enhancement.enhanced_query}")
        print(f"Entities: {enhancement.extracted_entities}")
        print(f"Synonyms: {enhancement.suggested_synonyms}")
        print(f"Confidence: {enhancement.confidence:.2f}")
        print(f"Fallback Used: {enhancement.fallback_used}")
        print(f"Reasoning: {enhancement.reasoning}")

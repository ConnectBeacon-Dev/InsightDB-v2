#!/usr/bin/env python3
"""
Qwen Model Client for LLM Integration

This module provides a client for the Qwen model using llama-cpp-python
to load and run the Qwen2.5-14B-Instruct-Q4_K_M.gguf model for query planning.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List

# Set up logging
logger = logging.getLogger(__name__)

class QwenModelClient:
    """
    Client for Qwen model using llama-cpp-python.
    
    This client loads the Qwen2.5-14B-Instruct-Q4_K_M.gguf model and provides
    a generate method compatible with the IndustrialQueryPlanner.
    """
    
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = None):
        """
        Initialize the Qwen model client.
        
        Args:
            model_path: Path to the Qwen GGUF model file
            n_ctx: Context window size (default: 4096)
            n_threads: Number of threads to use (default: auto-detect)
        """
        self.model_path = model_path
        self.model = None
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count()
        
        logger.info(f"Initializing Qwen model client with model: {model_path}")
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen model using llama-cpp-python."""
        try:
            # Try to import llama-cpp-python
            from llama_cpp import Llama
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load the model
            logger.info(f"Loading Qwen model from: {self.model_path}")
            logger.info(f"Context window: {self.n_ctx}, Threads: {self.n_threads}")
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,  # Reduce llama.cpp verbosity
                n_gpu_layers=0  # Use CPU only for compatibility
            )
            
            logger.info("✅ Qwen model loaded successfully")
            
        except ImportError as e:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            raise ImportError("llama-cpp-python is required for Qwen model client") from e
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            raise RuntimeError(f"Failed to load Qwen model: {e}") from e
    
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.1, 
                 top_p: float = 0.9, stop: Optional[List[str]] = None) -> str:
        """
        Generate text using the Qwen model.
        
        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter
            stop: List of stop sequences
            
        Returns:
            Generated text response
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        try:
            logger.info(f"🤖 Qwen Generate: prompt_length={len(prompt)}, max_tokens={max_tokens}")
            
            # Format prompt for Qwen instruction format
            formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Generate response
            response = self.model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or ["<|im_end|>"],
                echo=False
            )
            
            # Extract the generated text
            generated_text = response['choices'][0]['text'].strip()
            
            logger.info(f"✅ Qwen generated {len(generated_text)} characters")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during Qwen generation: {e}")
            raise RuntimeError(f"Qwen generation failed: {e}") from e
    
    def is_available(self) -> bool:
        """Check if the model is loaded and available."""
        return self.model is not None
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "context_size": self.n_ctx,
            "threads": self.n_threads,
            "available": self.is_available()
        }

def create_qwen_client(config: dict) -> Optional[QwenModelClient]:
    """
    Factory function to create a Qwen model client from config.
    
    Args:
        config: Configuration dictionary containing qwen_model_path
        
    Returns:
        QwenModelClient instance or None if creation fails
    """
    try:
        qwen_model_path = config.get('qwen_model_path')
        if not qwen_model_path:
            logger.warning("No qwen_model_path found in config")
            return None
        
        # Create and return the client
        client = QwenModelClient(qwen_model_path)
        logger.info("Qwen model client created successfully")
        return client
        
    except Exception as e:
        logger.error(f"Failed to create Qwen client: {e}")
        return None

# Test function
def test_qwen_client():
    """Test the Qwen model client."""
    from src.load_config import load_config
    
    logger.info("Testing Qwen model client...")
    
    config = load_config()
    client = create_qwen_client(config)
    
    if client is None:
        logger.error("Failed to create Qwen client")
        return False
    
    # Test generation
    test_prompt = "What is artificial intelligence?"
    try:
        response = client.generate(test_prompt, max_tokens=100)
        logger.info(f"Test response: {response[:200]}...")
        return True
    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        return False

if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    test_qwen_client()

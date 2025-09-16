#!/usr/bin/env python3
"""
Dense Index Builder for Company Data

This script:
1. Loads the TF-IDF index created by create_tfidf_search_index.py
2. Encodes company documents using SentenceTransformer
3. Builds a FAISS index for dense vector search
4. Saves the FAISS index and metadata for later use

Usage: python build_dense_index.py
"""

import os
import pickle
import json
import logging
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.load_config import (
    load_config,
    get_company_mapped_data_tfidf_search_store,
    get_company_mapped_data_dense_index_store
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompanyDenseIndexBuilder:
    """Build and manage dense vector search index for company data."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.tfidf_store_dir = get_company_mapped_data_tfidf_search_store(config)
        self.dense_store_dir = get_company_mapped_data_dense_index_store(config)
        
        # Get model configuration from config
        self.embed_model_path = config.get('sentence_transformer_model_name', 'sentence-transformers/all-mpnet-base-v2')
        self.embed_model_from_net = config.get('sentence_transformer_model_from_net', 'sentence-transformers/all-mpnet-base-v2')
        self.batch_size = config.get('batch_size', 256)
        
        # Use local model if available, otherwise use from network
        if os.path.exists(self.embed_model_path):
            self.model_name = self.embed_model_path
            logger.debug(f"Using local model: {self.embed_model_path}")
        else:
            self.model_name = self.embed_model_from_net
            logger.debug(f"Using network model: {self.embed_model_from_net}")
    
    def l2_normalize(self, embeddings):
        """L2 normalize embeddings for cosine similarity via inner product."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms
    
    def load_tfidf_data(self):
        """Load company documents and metadata from TF-IDF index."""
        tfidf_index_path = Path(self.tfidf_store_dir) / "tfidf_search_index.pkl"
        
        if not tfidf_index_path.exists():
            logger.error(f"TF-IDF index not found: {tfidf_index_path}")
            raise FileNotFoundError(f"TF-IDF index not found: {tfidf_index_path}")
        
        logger.info(f"Loading TF-IDF data from: {tfidf_index_path}")
        
        try:
            with open(tfidf_index_path, "rb") as f:
                data = pickle.load(f)
            
            docs = data.get("company_documents", [])
            metas = data.get("company_metadata", [])
            
            if not docs or not metas:
                raise RuntimeError("TF-IDF index missing documents or metadata.")
            
            if len(docs) != len(metas):
                raise RuntimeError(f"Document/metadata length mismatch: {len(docs)} vs {len(metas)}")
            
            logger.info(f"Loaded {len(docs)} company documents and metadata")
            return docs, metas
            
        except Exception as e:
            logger.error(f"Error loading TF-IDF data: {e}")
            raise
    
    def create_embeddings(self, documents):
        """Create embeddings for company documents."""
        logger.info(f"Loading SentenceTransformer model: {self.model_name}")
        
        try:
            model = SentenceTransformer(self.model_name, device="cpu")
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        logger.info(f"Creating embeddings for {len(documents)} documents...")
        logger.info(f"Using batch size: {self.batch_size}")
        
        try:
            embeddings = model.encode(
                documents, 
                batch_size=self.batch_size, 
                show_progress_bar=True, 
                normalize_embeddings=False
            )
            
            embeddings = np.asarray(embeddings, dtype="float32")
            embeddings = self.l2_normalize(embeddings)  # Normalize for cosine similarity
            
            logger.info(f"Created embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def build_faiss_index(self, embeddings):
        """Build FAISS index from embeddings."""
        logger.info("Building FAISS index...")
        
        try:
            dim = embeddings.shape[1]
            # Use IndexFlatIP for cosine similarity via inner product (since embeddings are normalized)
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            
            logger.info(f"Built FAISS index with {index.ntotal} vectors, dimension {dim}")
            return index
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise
    
    def save_dense_index(self, index, embeddings, metadata):
        """Save the dense index and related data."""
        logger.info(f"Saving dense index to: {self.dense_store_dir}")
        
        try:
            # Ensure directory exists
            Path(self.dense_store_dir).mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss_path = Path(self.dense_store_dir) / "faiss.index"
            faiss.write_index(index, str(faiss_path))
            logger.info(f"Saved FAISS index to: {faiss_path}")
            
            # Save embeddings shape
            shape_path = Path(self.dense_store_dir) / "embeddings.shape.npy"
            np.save(str(shape_path), np.array(embeddings.shape, dtype=np.int64))
            logger.info(f"Saved embeddings shape to: {shape_path}")
            
            # Save metadata as JSONL
            metas_path = Path(self.dense_store_dir) / "metas.jsonl"
            with open(metas_path, "w", encoding="utf-8") as fh:
                for meta in metadata:
                    fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
            logger.info(f"Saved metadata to: {metas_path}")
            
            # Save configuration
            config_data = {
                "embed_model": self.model_name,
                "batch_size": self.batch_size,
                "embedding_dim": embeddings.shape[1],
                "num_documents": embeddings.shape[0],
                "index_type": "IndexFlatIP",
                "normalized": True
            }
            
            config_path = Path(self.dense_store_dir) / "config.json"
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(config_data, fh, ensure_ascii=False, indent=2)
            logger.info(f"Saved configuration to: {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving dense index: {e}")
            raise
    
    def build_index(self):
        """Main method to build the dense index."""
        logger.info("Starting dense index creation...")
        
        # Load TF-IDF data
        documents, metadata = self.load_tfidf_data()
        
        # Create embeddings
        embeddings = self.create_embeddings(documents)
        
        # Build FAISS index
        index = self.build_faiss_index(embeddings)
        
        # Save everything
        self.save_dense_index(index, embeddings, metadata)
        
        logger.info("Dense index creation complete!")
        logger.info(f"Total documents indexed: {len(documents)}")
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"Index saved to: {self.dense_store_dir}")

def build_dense_index(config):
    """Main function to build dense search index."""
    builder = CompanyDenseIndexBuilder(config)
    builder.build_index()

if __name__ == "__main__":
    (config, logger) = load_config()
    if config is None:
        logger.error("Failed to load configuration")
        exit(1)
    
    build_dense_index(config)

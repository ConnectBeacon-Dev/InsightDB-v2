import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"   # must be set before importing huggingface_hub

import sys
import shutil
import time
from pathlib import Path
from huggingface_hub import snapshot_download

# Fixed target directory and model name
TARGET_BASE = Path("sentence-transformers")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def get_fresh_model(model_name: str) -> str:
    """
    Always download a fresh copy of the model into:
    flat-file-approach/models/sentence-transformers/<name_with_underscores>
    Works on Windows without symlinks/admin.
    Includes retry mechanism to prevent empty directories.
    """
    TARGET_BASE.mkdir(parents=True, exist_ok=True)
    target_dir = TARGET_BASE / model_name.replace("/", "_")

    for attempt in range(MAX_RETRIES):
        try:
            print(f"[INFO] Attempt {attempt + 1}/{MAX_RETRIES} - Downloading {model_name}")
            
            # Clean up any existing directory
            if target_dir.exists():
                print(f"[INFO] Removing existing {target_dir}")
                shutil.rmtree(target_dir)

            # Use default HF cache (user cache dir) to avoid symlinks in your project tree
            snapshot_path = snapshot_download(
                repo_id=model_name,
                force_download=True,      # always fetch fresh
                local_files_only=False,   # allow network
                resume_download=True      # resume if interrupted
                # NOTE: no cache_dir here; we use default user cache to avoid symlinks in your project
            )

            print(f"[INFO] Copying from cache snapshot to target: {target_dir}")
            shutil.copytree(snapshot_path, target_dir)

            # Verify the model was downloaded completely
            if not _verify_model_complete(target_dir):
                raise Exception("Downloaded model is incomplete")

            # Additional validation - check directory is not empty
            if not _verify_directory_not_empty(target_dir):
                raise Exception("Target directory is empty after download")

            # Test model loading to ensure it's functional
            if not _test_model_loading(target_dir):
                raise Exception("Model cannot be loaded properly")

            # Optional: archive for offline reuse
            try:
                archive = shutil.make_archive(str(target_dir), "zip", target_dir)
                print(f"[INFO] Archived at: {archive}")
            except Exception as e:
                print(f"[WARN] Could not create archive: {e}")

            print(f"✅ Fresh model ready at: {target_dir}")
            return str(target_dir)
            
        except Exception as e:
            print(f"[ERROR] Attempt {attempt + 1} failed: {e}")
            
            # Clean up partial download
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            
            if attempt < MAX_RETRIES - 1:
                print(f"[INFO] Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"[ERROR] All {MAX_RETRIES} attempts failed for {model_name}")
                raise Exception(f"Failed to download model after {MAX_RETRIES} attempts: {e}")


def _verify_model_complete(model_path: Path) -> bool:
    """Verify that the downloaded model contains all necessary files."""
    required_files = ["config.json"]
    model_files = ["pytorch_model.bin", "model.safetensors"]
    
    # Check for config.json
    if not (model_path / "config.json").exists():
        print(f"[ERROR] Missing config.json in {model_path}")
        return False
    
    # Check for at least one model file
    has_model_file = any((model_path / f).exists() for f in model_files)
    if not has_model_file:
        print(f"[ERROR] Missing model files in {model_path}")
        return False
    
    print(f"[INFO] Model verification passed for {model_path}")
    return True


def _verify_directory_not_empty(model_path: Path) -> bool:
    """Verify that the model directory is not empty and contains files."""
    try:
        files = list(model_path.rglob("*"))
        file_count = len([f for f in files if f.is_file()])
        
        if file_count == 0:
            print(f"[ERROR] Directory {model_path} is empty (no files found)")
            return False
        
        print(f"[INFO] Directory contains {file_count} files")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to check directory contents: {e}")
        return False


def _test_model_loading(model_path: Path) -> bool:
    """Test that the model can be loaded with sentence-transformers."""
    try:
        # Import here to avoid issues if sentence-transformers is not installed
        from sentence_transformers import SentenceTransformer
        
        print(f"[INFO] Testing model loading from {model_path}")
        
        # Try to load the model
        model = SentenceTransformer(str(model_path))
        
        # Test encoding a simple sentence
        test_sentence = "This is a test sentence."
        embedding = model.encode([test_sentence])
        
        if embedding is None or len(embedding) == 0:
            print(f"[ERROR] Model loaded but failed to generate embeddings")
            return False
        
        print(f"[INFO] Model loading test passed - embedding shape: {embedding.shape}")
        return True
        
    except ImportError:
        print(f"[WARN] sentence-transformers not available for testing, skipping model load test")
        return True  # Skip test if library not available
    except Exception as e:
        print(f"[ERROR] Model loading test failed: {e}")
        return False


def download_default_model() -> str:
    """Download the default sentence transformer model"""
    print(f"[INFO] Downloading default model: {MODEL_NAME}")
    return get_fresh_model(MODEL_NAME)


# =====================
# Main execution
# =====================
if __name__ == "__main__":
    try:
        print(f"[INFO] Starting download of {MODEL_NAME}")
        print(f"[INFO] Target directory: {TARGET_BASE}")
        local_model_path = download_default_model()
        print(f"✅ Model ready at: {local_model_path}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        sys.exit(1)

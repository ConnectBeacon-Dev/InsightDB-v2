import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"   # must be set before importing huggingface_hub

import sys
import shutil
import time
from pathlib import Path
from huggingface_hub import snapshot_download

# -------- Settings --------
BASE_DIR     = Path("models")  # <-- put everything under ./models
TARGET_BASE  = BASE_DIR / "sentence-transformers"
MODEL_NAME   = "sentence-transformers/all-mpnet-base-v2"
TARGET_DIR   = TARGET_BASE / MODEL_NAME.replace("/", "_")  # models/sentence-transformers/sentence-transformers_all-mpnet-base-v2

MAX_RETRIES  = 3
RETRY_DELAY  = 5  # seconds

def main():
    try:
        print(f"[INFO] Desired target: {TARGET_DIR}")
        if _is_model_present(TARGET_DIR):
            print(f"✅ Model already present at: {TARGET_DIR}")
            sys.exit(0)

        # Ensure base dirs exist
        TARGET_BASE.mkdir(parents=True, exist_ok=True)

        # Download (with retries)
        local_model_path = _get_fresh_model(MODEL_NAME, TARGET_DIR)
        print(f"✅ Model ready at: {local_model_path}")
    except Exception as e:
        print(f"❌ Failed to prepare model: {e}")
        sys.exit(1)

def _is_model_present(path: Path) -> bool:
    """Return True if the target directory already exists and looks valid."""
    if not path.exists():
        return False
    # Minimal validity check: has config.json and at least one model file
    has_config = (path / "config.json").exists()
    has_model_bin = (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()
    if has_config and has_model_bin:
        print(f"[INFO] Existing directory looks valid: {path}")
        return True
    print(f"[WARN] Existing directory is incomplete/corrupt, will re-download: {path}")
    return False

def _get_fresh_model(model_name: str, target_dir: Path) -> str:
    """
    Download a fresh copy from HF cache snapshot into target_dir (no symlinks).
    Retries to avoid half-empty dirs.
    """
    for attempt in range(MAX_RETRIES):
        try:
            print(f"[INFO] Attempt {attempt + 1}/{MAX_RETRIES} - Downloading {model_name}")

            # Clean up any existing target_dir before copy
            if target_dir.exists():
                print(f"[INFO] Removing existing {target_dir}")
                shutil.rmtree(target_dir, ignore_errors=True)

            # Use default HF cache; then copy to our target_dir
            snapshot_path = snapshot_download(
                repo_id=model_name,
                force_download=True,      # ensure fresh files (cache still used under the hood)
                local_files_only=False,
                resume_download=True
            )

            print(f"[INFO] Copying snapshot -> {target_dir}")
            shutil.copytree(snapshot_path, target_dir)

            # Verify
            if not _verify_model_complete(target_dir):
                raise RuntimeError("Downloaded model is incomplete")

            if not _verify_directory_not_empty(target_dir):
                raise RuntimeError("Target directory is empty after download")

            # Optional: test loading (silently skipped if sentence-transformers not installed)
            if not _test_model_loading(target_dir):
                raise RuntimeError("Model cannot be loaded properly")

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
            # Clean partial
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            if attempt < MAX_RETRIES - 1:
                print(f"[INFO] Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                raise

def _verify_model_complete(model_path: Path) -> bool:
    """Verify minimal required files exist."""
    required_files = ["config.json"]
    model_files = ["pytorch_model.bin", "model.safetensors"]

    if not (model_path / "config.json").exists():
        print(f"[ERROR] Missing config.json in {model_path}")
        return False

    if not any((model_path / f).exists() for f in model_files):
        print(f"[ERROR] Missing model files in {model_path}")
        return False

    print(f"[INFO] Model verification passed for {model_path}")
    return True

def _verify_directory_not_empty(model_path: Path) -> bool:
    """Ensure there are actual files."""
    try:
        file_count = sum(1 for f in model_path.rglob("*") if f.is_file())
        if file_count == 0:
            print(f"[ERROR] Directory {model_path} is empty")
            return False
        print(f"[INFO] Directory contains {file_count} files")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to check directory contents: {e}")
        return False

def _test_model_loading(model_path: Path) -> bool:
    """Try loading with sentence-transformers (optional)."""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[INFO] Testing model loading from {model_path}")
        model = SentenceTransformer(str(model_path))
        emb = model.encode(["This is a test sentence."])
        if emb is None or len(emb) == 0:
            print(f"[ERROR] Model loaded but no embeddings returned")
            return False
        # emb is a numpy array; show shape if available
        try:
            print(f"[INFO] Model loading test passed - embedding shape: {emb.shape}")
        except Exception:
            print(f"[INFO] Model loading test passed")
        return True
    except ImportError:
        print(f"[WARN] sentence-transformers not installed; skipping load test")
        return True
    except Exception as e:
        print(f"[ERROR] Model loading test failed: {e}")
        return False

if __name__ == "__main__":
    main()

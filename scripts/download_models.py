"""Download models during Docker build to avoid runtime downloads."""

import os
import sys

# Set HuggingFace cache directory
os.environ["HF_HOME"] = "/app/models"
os.environ["TRANSFORMERS_CACHE"] = "/app/models"

def download_bge_m3():
    """Download BGE-M3 model for ColBERT embeddings."""
    print("Downloading BGE-M3 model...")
    from FlagEmbedding import BGEM3FlagModel

    # Initialize model to trigger download
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cpu")

    # Run a test embedding to ensure model is fully loaded
    result = model.encode(
        ["test"],
        return_dense=False,
        return_sparse=False,
        return_colbert_vecs=True
    )
    print(f"BGE-M3 model downloaded successfully. Test embedding shape: {result['colbert_vecs'][0].shape}")


if __name__ == "__main__":
    download_bge_m3()
    print("All models downloaded successfully!")

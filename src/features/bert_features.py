"""
BERT-based feature extraction for book descriptions.

- Uses HuggingFace Transformers (bert-base-multilingual-cased by default)
- Supports GPU/CPU, batching for scalability
- Pools output to a single vector per description (mean pooling)
- Provides fit/transform/save/load interface for pipeline integration

Example usage:
    from features.bert_features import compute_bert_embeddings
    book_embeddings = compute_bert_embeddings(book_descriptions_df)

Expected input: DataFrame with at least ["book_id", "description"] columns
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import joblib

def compute_bert_embeddings(
    descriptions_df: pd.DataFrame,
    model_name: str = "bert-base-multilingual-cased",
    description_col: str = "description",
    id_col: str = "book_id",
    device: Optional[str] = None,
    batch_size: int = 16,
    max_length: int = 256,
    output_path: Optional[Path] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Computes (and optionally saves) BERT-pooled embeddings for book descriptions.

    Args:
        descriptions_df: DataFrame with id_col and description_col.
        model_name: HF model name (default: bert-base-multilingual-cased)
        description_col: Column in descriptions_df with text to embed
        id_col: Key column matching books/users/etc.
        device: "cuda", "cpu", or None (auto)
        batch_size: Batch size for embedding
        max_length: Max token length for encoding
        output_path: If set, saves dictionary mapping id_col -> embedding
        force_recompute: If False and output_path exists, loads from disk
    Returns:
        DataFrame: id_col plus N-dim embedding columns
    """
    if output_path and output_path.exists() and not force_recompute:
        print(f"Loading cached BERT embeddings from {output_path}")
        emb_dict = joblib.load(output_path)
        ids = list(emb_dict.keys())
        arr = np.stack([emb_dict[_id] for _id in ids])
        cols = [f"bert_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame({id_col: ids, **{col: arr[:, i] for i, col in enumerate(cols)}})

    # Auto-detect device with fallback to CPU on errors
    if device is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Test GPU access if CUDA is available
            if device == "cuda":
                _ = torch.zeros(1).cuda()  # Test GPU access
        except Exception as e:
            print(f"Warning: GPU access failed ({e}), falling back to CPU")
            device = "cpu"
    
    # Force CPU mode for stability on Windows
    import os
    if os.name == 'nt':  # Windows
        device = "cpu"
        print("Windows detected: Using CPU mode for BERT (more stable)")
    
    print(f"Loading BERT ({model_name}) on {device}...")
    try:
        # Set CPU threads for better stability on Windows
        if device == "cpu":
            torch.set_num_threads(2)  # Limit threads to prevent crashes
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        if device == "cuda":
            print("Retrying on CPU...")
            device = "cpu"
            torch.set_num_threads(2)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.to(device)
            model.eval()
        else:
            raise

    ids = descriptions_df[id_col].tolist()
    texts = descriptions_df[description_col].fillna("").tolist()
    emb_list = []
    
    # Reduce batch size if on CPU or if memory is limited
    # Very small batch size on CPU/Windows to prevent crashes
    effective_batch_size = batch_size if device == "cuda" else min(batch_size, 2)
    
    try:
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), effective_batch_size), desc="Computing BERT embeddings"):
                batch = texts[i:i+effective_batch_size]
                batch_ids = ids[i:i+effective_batch_size]
                try:
                    enc = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}
                    outputs = model(**enc)
                    # Mean pooling
                    h = outputs.last_hidden_state
                    m = enc["attention_mask"].unsqueeze(-1)
                    h_pool = (h * m).sum(1) / m.sum(1)
                    emb_list.extend(h_pool.cpu().numpy())
                    # Clear GPU cache periodically
                    if device == "cuda" and i % (effective_batch_size * 10) == 0:
                        torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e) or "CUDA" in str(e):
                        print(f"Memory error at batch {i}, reducing batch size...")
                        torch.cuda.empty_cache() if device == "cuda" else None
                        effective_batch_size = max(1, effective_batch_size // 2)
                        # Retry with smaller batch
                        continue
                    else:
                        raise
    finally:
        # Cleanup
        if device == "cuda":
            torch.cuda.empty_cache()
        del model, tokenizer
    arr = np.stack(emb_list)
    cols = [f"bert_{i}" for i in range(arr.shape[1])]
    out_df = pd.DataFrame({id_col: ids, **{col: arr[:, i] for i, col in enumerate(cols)}})
    if output_path:
        joblib.dump(dict(zip(ids, arr)), output_path)
        print(f"Saved BERT embeddings to {output_path}")
    return out_df

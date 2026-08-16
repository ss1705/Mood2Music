"""
src/modal_clap.py

Modal serverless function for CLAP text encoding.

Why Modal:
    CLAP model (~900MB) is too large to load on Railway's memory budget
    alongside the sentence transformer and FAISS index. Modal runs CLAP
    on a T4 GPU serverless — only paying for the seconds it runs.

How it fits in:
    Railway (FastAPI) calls encode_text() via Modal's Python client.
    Modal returns a 512-dim vector. Railway searches its local CLAP
    FAISS index with that vector and reranks results.

Cold start: ~15s on first daily call (model download + load).
Warm calls: ~1-2s (model cached between calls in same session).
"""

import modal

app = modal.App("mood2music-clap")

# Image with all dependencies CLAP needs
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "librosa",
        "numpy",
        "faiss-cpu"
    )
)

@app.function(
    gpu="T4",
    image=image,
    timeout=120,
)
def encode_text(query_text: str) -> list[float]:
    """
    Encode a mood query using CLAP's text encoder.
    Returns a 512-dim normalized vector for FAISS search.

    Uses laion/clap-htsat-fused — confirmed working checkpoint.
    laion/larger_clap_music was found to produce near-zero similarities
    (broken checkpoint) during Phase 2B development.
    """
    import torch
    import numpy as np
    from transformers import ClapModel, ClapProcessor
    import faiss

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ClapModel.from_pretrained('laion/clap-htsat-fused').to(device)
    processor = ClapProcessor.from_pretrained('laion/clap-htsat-fused')
    model.eval()

    with torch.no_grad():
        inputs = processor(
            text=[query_text],
            return_tensors='pt',
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        embed = model.get_text_features(**inputs)
        if hasattr(embed, 'pooler_output'):
            embed = embed.pooler_output

    # Normalize for cosine similarity (same as FAISS index normalization)
    vec = embed.cpu().numpy()
    faiss.normalize_L2(vec)

    return vec[0].tolist()  # return as list — serializable over network
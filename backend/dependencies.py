"""
backend/dependencies.py

Model and artifact loading for FastAPI dependency injection.
All heavy objects (models, indexes, catalog) are loaded once
at startup and shared across requests via app.state.
"""
import os
import sys
import faiss
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure src/ is importable from the backend directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.retriever import (
    load_catalog,
    load_2a_artifacts,
    load_clap_artifacts,
)


def load_all_artifacts():
    import os
    use_clap = os.getenv("USE_CLAP", "0") == "1"

    print("Loading catalog...")
    catalog = load_catalog()
    print(f"  {len(catalog)} tracks loaded")

    print("Loading Phase 2A artifacts...")
    model_2a, index_2a, map_2a = load_2a_artifacts()
    print(f"  Index size: {index_2a.ntotal}")

    if use_clap:
        print("Loading CLAP FAISS index (model runs on Modal)...")
        index_clap = faiss.read_index(str(PROJECT_ROOT / 'data' / 'track_index_clap.faiss'))
        with open(PROJECT_ROOT / 'data' / 'track_id_map_clap.pkl', 'rb') as f:
            map_clap = pickle.load(f)
        model_clap = processor_clap = device_clap = None
        print(f"  CLAP index size: {index_clap.ntotal}")
    else:
        model_clap = processor_clap = device_clap = index_clap = map_clap = None

    return {
        'catalog':        catalog,
        'model_2a':       model_2a,
        'index_2a':       index_2a,
        'map_2a':         map_2a,
        'model_clap':     model_clap,
        'processor_clap': processor_clap,
        'device_clap':    device_clap,
        'index_clap':     index_clap,
        'map_clap':       map_clap,
    }
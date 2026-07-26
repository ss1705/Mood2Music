"""
src/retrieval/retriever.py

Three retrieval systems in one module:
  - search_2a:    sentence-transformer text retrieval (Phase 2A)
  - search_clap:  CLAP audio retrieval (Phase 2B)
  - search_hybrid: CLAP retrieval + valence reranking (Phase 2C)

Design:
  All three functions share the same interface — query text in,
  list of result dicts out — so they can be compared directly
  and swapped without changing calling code.
"""

import numpy as np
import faiss
import duckdb
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CATALOG_PATH = PROJECT_ROOT / 'data' / 'catalog.parquet'


# ── loaders ──────────────────────────────────────────────────────────────────

def load_catalog():
    return duckdb.query(f"""
        SELECT track_id, title, artist_name, genre_top,
               valence, energy, danceability, acousticness,
               instrumentalness, tempo_norm, valence_reliable
        FROM '{CATALOG_PATH}'
        WHERE valence_reliable = true
    """).df()


def load_2a_artifacts():
    """Load sentence-transformer model and Phase 2A FAISS index."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(str(PROJECT_ROOT / 'data' / 'track_index.faiss'))
    with open(PROJECT_ROOT / 'data' / 'track_id_map.pkl', 'rb') as f:
        track_id_map = pickle.load(f)
    return model, index, track_id_map


def load_clap_artifacts():
    """
    Load CLAP model and Phase 2B FAISS index.

    Import is deferred inside this function intentionally.
    transformers + torch are heavy — only pay the import cost
    when CLAP retrieval is actually needed, not on module load.
    """
    import torch
    from transformers import ClapModel, ClapProcessor

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ClapModel.from_pretrained('laion/clap-htsat-fused').to(device)
    processor = ClapProcessor.from_pretrained('laion/clap-htsat-fused')
    model.eval()

    index = faiss.read_index(str(PROJECT_ROOT / 'data' / 'track_index_clap.faiss'))
    with open(PROJECT_ROOT / 'data' / 'track_id_map_clap.pkl', 'rb') as f:
        track_id_map = pickle.load(f)

    return model, processor, device, index, track_id_map


# ── helpers ───────────────────────────────────────────────────────────────────

def _track_metadata(track_id, catalog):
    """Return a single track's metadata dict by track_id."""
    row = catalog[catalog['track_id'] == track_id].iloc[0]
    return {
        'track_id':  int(row['track_id']),
        'title':     row['title'],
        'artist':    row['artist_name'],
        'genre':     row['genre_top'],
        'valence':   round(float(row['valence']), 3),
        'energy':    round(float(row['energy']), 3),
    }


def _embed_text_clap(query_text, model, processor, device):
    """
    Embed a text query using CLAP's text encoder.

    pooler_output extraction needed for transformers >= 5.10:
    get_text_features() returns BaseModelOutputWithPooling
    rather than a plain tensor on newer versions.
    clap-htsat-fused confirmed working with this fix.
    """
    import torch
    with torch.no_grad():
        inputs = processor(
            text=[query_text], return_tensors='pt', padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        embed = model.get_text_features(**inputs)
        if hasattr(embed, 'pooler_output'):
            embed = embed.pooler_output
    return embed.cpu().numpy()


# ── retrieval functions ───────────────────────────────────────────────────────

def search_2a(query_text, model, index, track_id_map, catalog, k=5):
    """
    Phase 2A: sentence-transformer retrieval.

    Embeds the query as text, searches against text-description
    embeddings of tracks. Works well for valence-axis queries
    (melancholy, uplifting) but limited by description vocabulary.
    """
    query_vec = model.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    scores, positions = index.search(query_vec, k)

    results = []
    for score, pos in zip(scores[0], positions[0]):
        track_id = track_id_map[pos]
        meta = _track_metadata(track_id, catalog)
        meta['similarity'] = round(float(score), 3)
        meta['system'] = '2a'
        results.append(meta)
    return results


def search_clap(query_text, model, processor, device,
                index, track_id_map, catalog, k=5):
    """
    Phase 2B: CLAP audio retrieval.

    Embeds the query using CLAP's text encoder, searches against
    CLAP audio embeddings. Strong on arousal/energy axis,
    inconsistent on valence axis.
    """
    query_vec = _embed_text_clap(query_text, model, processor, device)
    faiss.normalize_L2(query_vec)
    scores, positions = index.search(query_vec, k)

    results = []
    for score, pos in zip(scores[0], positions[0]):
        track_id = track_id_map[pos]
        meta = _track_metadata(track_id, catalog)
        meta['similarity'] = round(float(score), 3)
        meta['system'] = 'clap'
        results.append(meta)
    return results


def search_hybrid(query_text,
                  model_2a, index_2a, map_2a,
                  model_clap, processor_clap, device_clap,
                  index_clap, map_clap,
                  catalog,
                  k=5,
                  candidate_pool=50,
                  w_clap=0.7,
                  w_valence=0.3,
                  w_energy=0.2,
                  n_valence_ref=5):
    """
    Two-stage retrieval with dual-axis reranking.

    Scoring:
        final_score = w_clap    × clap_similarity
                    - w_valence × |track_valence - target_valence|
                    - w_energy  × |track_energy  - target_energy|

    target_valence and target_energy both derived from Phase 2A top-n
    results — data-driven, no hand-coded mood mapping.

    w_clap=0.7, w_valence=0.3, w_energy=0.2:
        CLAP dominates (broad semantic match).
        Valence penalty stronger than energy because CLAP's valence
        weakness is more severe than its energy weakness.
        Energy penalty is lighter — CLAP already does reasonably
        well on energy, so we correct gently rather than override.
    """
    # Step 1 — derive targets from Phase 2A
    ref = search_2a(query_text, model_2a, index_2a, map_2a, catalog,
                    k=n_valence_ref)
    target_valence = np.mean([r['valence'] for r in ref])
    target_energy  = np.mean([r['energy']  for r in ref])

    # Step 2 — CLAP candidate pool
    query_vec = _embed_text_clap(
        query_text, model_clap, processor_clap, device_clap
    )
    faiss.normalize_L2(query_vec)
    scores, positions = index_clap.search(query_vec, candidate_pool)

    candidates = []
    for score, pos in zip(scores[0], positions[0]):
        meta = _track_metadata(map_clap[pos], catalog)
        meta['clap_similarity'] = round(float(score), 3)
        candidates.append(meta)

    # Step 3 — rerank with dual penalty
    for c in candidates:
        v_gap = abs(c['valence'] - target_valence)
        e_gap = abs(c['energy']  - target_energy)
        c['valence_gap']    = round(v_gap, 3)
        c['energy_gap']     = round(e_gap, 3)
        c['target_valence'] = round(target_valence, 3)
        c['target_energy']  = round(target_energy, 3)
        c['final_score']    = round(
            w_clap   * c['clap_similarity']
            - w_valence * v_gap
            - w_energy  * e_gap, 4
        )

    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    return candidates[:k]
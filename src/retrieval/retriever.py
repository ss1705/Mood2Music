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
import pandas as pd
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

def _get_va_coordinates(mood_text, model_2a, index_2a, map_2a, catalog, k=5):
    """
    Convert free-text mood to (valence, energy) coordinates.

    Three-layer strategy:
      1. Phrase overrides — compound expressions whose meaning differs
         from their component words (e.g. 'low energy' should map to
         low arousal, but word-level lookup gives moderate arousal
         because 'energy' is high-arousal in general English).
      2. VAD lexicon — word-level lookup with modifier handling.
      3. Phase 2A fallback — semantic retrieval for unrecognized input.
    """
    import re
    text_lower = mood_text.lower()

    # Layer 1 — phrase overrides
    phrase_matches = []
    for phrase, coords in PHRASE_OVERRIDES.items():
        if phrase in text_lower:
            phrase_matches.append(coords)

    # Layer 2 — word-level VAD with modifiers
    words = re.findall(r'[a-z]+', text_lower)
    words = [w for w in words if w not in STOPWORDS]
    word_matches = _apply_modifiers(words, VAD_LEXICON)

    all_matches = phrase_matches + word_matches

    if all_matches:
        valence = float(np.clip(np.mean([m[0] for m in all_matches]), 0, 1))
        energy  = float(np.clip(np.mean([m[1] for m in all_matches]), 0, 1))
        return round(valence, 3), round(energy, 3)

    # Layer 3 — Phase 2A fallback
    ref = search_2a(mood_text, model_2a, index_2a, map_2a, catalog, k=k)
    valence = float(np.mean([r['valence'] for r in ref]))
    energy  = float(np.mean([r['energy']  for r in ref]))
    return round(valence, 3), round(energy, 3)

def _tracks_near_coordinate(target_valence, target_energy, catalog, k=2, exclude_ids=None):
    """
    Find tracks whose Echonest valence/energy are nearest to a target
    coordinate in VA space, using Euclidean distance.

    Pure numeric search — no embeddings, no FAISS. Works directly on
    the catalog DataFrame.

    Args:
        exclude_ids: set of track_ids already used in earlier waypoints,
                     so the playlist doesn't repeat tracks.
    """
    if exclude_ids is None:
        exclude_ids = set()

    df = catalog[~catalog['track_id'].isin(exclude_ids)].copy()

    df['va_distance'] = np.sqrt(
        (df['valence'] - target_valence) ** 2 +
        (df['energy']  - target_energy)  ** 2
    )

    nearest = df.nsmallest(k, 'va_distance')

    results = []
    for _, row in nearest.iterrows():
        results.append({
            'track_id':   int(row['track_id']),
            'title':      row['title'],
            'artist':     row['artist_name'],
            'genre':      row['genre_top'],
            'valence':    round(float(row['valence']), 3),
            'energy':     round(float(row['energy']),  3),
            'va_distance': round(float(row['va_distance']), 3),
        })
    return results


def search_regulation(current_mood, target_mood,
                      model_2a, index_2a, map_2a,
                      catalog,
                      n_waypoints=5,
                      tracks_per_waypoint=2):
    """
    Mood regulation playlist: navigates from current mood to target mood
    through the valence-arousal space via linear interpolation.

    Design:
        Both mood inputs are converted to VA coordinates via Phase 2A —
        the same mechanism used in search_hybrid to derive penalty targets.
        Waypoints are evenly spaced along the straight line between them.
        Each waypoint retrieves tracks by Euclidean distance in VA space.

        Why straight-line interpolation:
        Linear interpolation is the simplest path between two points in
        a continuous space. More complex paths (curves, weighted steps)
        would require assumptions about how emotional transitions feel
        that we don't have data to justify. Start linear, refine later.

        Why exclude already-used tracks:
        A regulation playlist should feel like a journey — repeating a
        track breaks the sense of forward motion.

    Args:
        current_mood:        free text describing how the user feels now
        target_mood:         free text describing how the user wants to feel
        n_waypoints:         number of steps including start and end (default 5)
        tracks_per_waypoint: songs retrieved at each waypoint (default 2)

    Returns:
        dict with keys:
            'current_coords'  — (valence, energy) of current mood
            'target_coords'   — (valence, energy) of target mood
            'waypoints'       — list of waypoint dicts, each containing:
                'step'        — waypoint index (0 = current, n-1 = target)
                'valence'     — waypoint valence coordinate
                'energy'      — waypoint energy coordinate
                'tracks'      — list of track dicts nearest this coordinate
    """
    # Step 1 — convert mood text to VA coordinates
    current_v, current_e = _get_va_coordinates(
        current_mood, model_2a, index_2a, map_2a, catalog
    )
    target_v, target_e = _get_va_coordinates(
        target_mood, model_2a, index_2a, map_2a, catalog
    )

    # Step 2 — interpolate waypoints
    waypoint_coords = []
    for i in range(n_waypoints):
        t = i / (n_waypoints - 1)   # 0.0 at start, 1.0 at end
        wp_v = round(current_v + t * (target_v - current_v), 3)
        wp_e = round(current_e + t * (target_e - current_e), 3)
        waypoint_coords.append((wp_v, wp_e))

    # Step 3 — retrieve tracks at each waypoint
    used_ids = set()
    waypoints = []

    for i, (wp_v, wp_e) in enumerate(waypoint_coords):
        tracks = _tracks_near_coordinate(
            wp_v, wp_e, catalog,
            k=tracks_per_waypoint,
            exclude_ids=used_ids
        )
        used_ids.update(t['track_id'] for t in tracks)

        waypoints.append({
            'step':    i,
            'valence': wp_v,
            'energy':  wp_e,
            'tracks':  tracks
        })

    return {
        'current_mood':   current_mood,
        'target_mood':    target_mood,
        'current_coords': (current_v, current_e),
        'target_coords':  (target_v,  target_e),
        'waypoints':      waypoints
    }

# Phrase-level VA overrides — compound expressions whose meaning
# differs significantly from the average of their component words.
# Applied before word-level lookup.
# Values derived from Russell's circumplex model.
PHRASE_OVERRIDES = {
    'low energy':      (0.40, 0.15),
    'high energy':     (0.70, 0.90),
    'low mood':        (0.15, 0.20),
    'good mood':       (0.80, 0.60),
    'bad mood':        (0.10, 0.50),
    'burnt out':       (0.15, 0.10),
    'worn out':        (0.20, 0.10),
    'wiped out':       (0.15, 0.10),
    'fed up':          (0.10, 0.60),
    'fired up':        (0.75, 0.90),
    'wound up':        (0.20, 0.85),
    'on edge':         (0.15, 0.80),
    'at peace':        (0.80, 0.10),
    'at ease':         (0.75, 0.15),
    'ready to go':     (0.75, 0.85),
    'can\'t focus':    (0.25, 0.60),
}

NEGATIONS = {'not', 'no', "n't", 'never', 'without'}
INTENSIFIERS = {'very', 'extremely', 'really', 'so', 'super'}
DIMINISHERS = {'low', 'little', 'slightly', 'somewhat', 'kind', 'kinda', 'sort'}
STOPWORDS = {
    'i', 'feel', 'feeling', 'am', 'im', 'want', 'to', 'be', 'like',
    'a', 'an', 'the', 'and', 'or', 'but', 'so', 'my', 'me', 'myself',
    'get', 'go', 'just', 'really', 'that', 'this', 'it', 'is', 'at',
    'in', 'of', 'with', 'for', 'have', 'has', 'do', 'did', 'would',
    'could', 'should', 'bit', 'lot', 'way', 'right', 'now'
}

def _apply_modifiers(words, vad_lexicon):
    """
    Handle modifier words before VAD lookup.
    
    Three cases:
      Negation:    'not calm'    → flip valence and arousal around 0.5
      Intensifier: 'very calm'  → push values further from 0.5
      Diminisher:  'low energy' → pull values toward 0.5
    
    Words not preceded by a modifier are looked up normally.
    Modifiers themselves are consumed and not looked up.
    """
    matched = []
    skip_next = False
    
    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue
            
        # Check if this word is a modifier
        if word in NEGATIONS and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in vad_lexicon:
                v, a = vad_lexicon[next_word]
                # Flip around 0.5 — negate the affect
                matched.append((1.0 - v, 1.0 - a))
                skip_next = True
            continue
            
        if word in INTENSIFIERS and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in vad_lexicon:
                v, a = vad_lexicon[next_word]
                # Push further from 0.5
                matched.append((
                    0.5 + 1.3 * (v - 0.5),
                    0.5 + 1.3 * (a - 0.5)
                ))
                skip_next = True
            continue
            
        if word in DIMINISHERS and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in vad_lexicon:
                v, a = vad_lexicon[next_word]
                # Pull toward 0.5
                matched.append((
                    0.5 + 0.5 * (v - 0.5),
                    0.5 + 0.5 * (a - 0.5)
                ))
                skip_next = True
            continue
        
        # No modifier — look up normally
        if word in vad_lexicon:
            matched.append(vad_lexicon[word])
    
    return matched

def _load_vad_lexicon():
    """
    Load NRC VAD Lexicon as a word → (valence, arousal) dict.
    Dominance dimension is excluded — not used in this system.
    
    Source: Mohammad & Turney (2018), NRC VAD Lexicon Aug2018 release.
    Terms: research use only, not redistributed.
    """
    path = PROJECT_ROOT / 'data' / 'NRC-VAD-Lexicon.txt'
    df = pd.read_csv(path, sep='\t', header=0,
                     names=['word', 'valence', 'arousal', 'dominance'])
    return {
        row['word']: (float(row['valence']), float(row['arousal']))
        for _, row in df.iterrows()
    }

VAD_LEXICON = _load_vad_lexicon()
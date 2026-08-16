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

     # Generate YouTube search URL for this track
    import urllib.parse
    query = urllib.parse.quote(f"{row['title']} {row['artist_name']}")
    youtube_url = f"https://www.youtube.com/results?search_query={query}"

    return {
        'track_id':  int(row['track_id']),
        'title':     row['title'],
        'artist':    row['artist_name'],
        'genre':     row['genre_top'],
        'valence':   round(float(row['valence']), 3),
        'energy':    round(float(row['energy']), 3),
        'youtube_url': youtube_url,
        'description': generate_track_description(row)
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
    Phase 2A: sentence-transformer retrieval with query reranking.

    Two-stage:
      1. FAISS retrieves top k*3 candidates by description similarity
      2. Rerank by direct query-description cosine similarity
      3. Return top k after reranking

    Why retrieve k*3 then rerank to k:
        Reranking within only k results is too narrow — the best match
        for the specific query might just miss the top-k cutoff from
        FAISS. Retrieving a wider pool gives reranking room to work.
    """
    query_vec = model.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    scores, positions = index.search(query_vec, k * 3)  # wider pool

    results = []
    for score, pos in zip(scores[0], positions[0]):
        track_id = track_id_map[pos]
        meta = _track_metadata(track_id, catalog)
        meta['similarity'] = round(float(score), 3)
        results.append(meta)

    # Rerank by query-description semantic similarity
    results = _rerank_by_query(results, query_text, model)

    return results[:k]

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
    query_vec = _embed_text_clap_modal(query_text)
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
    import urllib.parse
    
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
        query = urllib.parse.quote(f"{row['title']} {row['artist_name']}")
        youtube_url = f"https://www.youtube.com/results?search_query={query}"
        results.append({
            'track_id':    int(row['track_id']),
            'title':       row['title'],
            'artist':      row['artist_name'],
            'genre':       row['genre_top'],
            'valence':     round(float(row['valence']), 3),
            'energy':      round(float(row['energy']),  3),
            'va_distance': round(float(row['va_distance']), 3),
            'youtube_url': youtube_url,
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

MOOD_VOCABULARY = {
    (1, 1): ["desolate", "hollow", "grief-stricken", "hopeless"],
    (1, 2): ["mournful", "sorrowful", "despairing", "forlorn"],
    (1, 3): ["anguished", "tortured", "harrowing", "bleak"],
    (1, 4): ["desperate", "agonizing", "raw", "gut-wrenching"],
    (2, 1): ["sad", "heavy", "subdued", "somber"],
    (2, 2): ["melancholy", "downcast", "wistful", "pensive"],
    (2, 3): ["troubled", "brooding", "dark", "unsettled"],
    (2, 4): ["tense", "anxious", "uneasy", "restless"],
    (3, 1): ["quiet", "withdrawn", "introspective", "dim"],
    (3, 2): ["reflective", "contemplative", "bittersweet", "longing"],
    (3, 3): ["stirring", "charged", "moody", "complex"],
    (3, 4): ["driving", "urgent", "intense", "forceful"],
    (4, 1): ["subdued", "understated", "restrained", "spare"],
    (4, 2): ["measured", "considered", "nuanced", "layered"],
    (4, 3): ["dynamic", "textured", "rich", "varied"],
    (4, 4): ["propulsive", "energetic", "vigorous", "bold"],
    (5, 1): ["gentle", "soft", "easy", "relaxed"],
    (5, 2): ["smooth", "balanced", "flowing", "steady"],
    (5, 3): ["lively", "engaging", "warm", "bright"],
    (5, 4): ["spirited", "animated", "vibrant", "punchy"],
    (6, 1): ["peaceful", "serene", "tender", "soothing"],
    (6, 2): ["content", "warm", "comfortable", "grounded"],
    (6, 3): ["upbeat", "cheerful", "buoyant", "light"],
    (6, 4): ["energized", "exciting", "invigorating", "alive"],
    (7, 1): ["blissful", "calm", "radiant", "glowing"],
    (7, 2): ["joyful", "happy", "bright", "sunny"],
    (7, 3): ["jubilant", "celebratory", "triumphant", "soaring"],
    (7, 4): ["exhilarating", "thrilling", "euphoric", "electric"],
    (8, 1): ["transcendent", "ethereal", "luminous", "sublime"],
    (8, 2): ["ecstatic", "elated", "overjoyed", "rapturous"],
    (8, 3): ["euphoric", "exuberant", "radiant", "glorious"],
    (8, 4): ["explosive", "unstoppable", "fierce", "electrifying"],
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

def _get_mood_words(valence: float, energy: float) -> list[str]:
    """
    Map valence and energy to mood vocabulary words.

    8 valence bands × 4 energy bands = 32 distinct emotional regions.
    Each region has 4 human-curated music mood words drawn from
    music description language — not general English affect words.

    Why hand-curated over NRC VAD Lexicon:
        The NRC VAD Lexicon covers all English words. Most high-valence
        words in it are domain-irrelevant ("moisturizer", "diploma").
        Music mood description requires domain-specific vocabulary that
        people actually use when describing how music feels.
    """
    # Map valence to 1-8 band
    v_band = min(8, max(1, int(valence * 8) + 1))
    if valence >= 1.0:
        v_band = 8

    # Map energy to 1-4 band
    e_band = min(4, max(1, int(energy * 4) + 1))
    if energy >= 1.0:
        e_band = 4

    return MOOD_VOCABULARY.get((v_band, e_band), ["atmospheric", "evocative"])

def generate_track_description(row):
    """
    Convert a track's Echonest features into a natural language description.

    Uses MOOD_VOCABULARY — 32 VA regions × 4 words each — to produce
    emotionally distinct descriptions across the valence-arousal space.
    Richer vocabulary means similar queries like "anxious" vs "sad"
    retrieve genuinely different tracks rather than collapsing to the
    same VA region.

    Thresholds:
        Acousticness: p50 split (0.574) — bimodal distribution
        Instrumentalness: fixed 0.5 — bimodal, domain knowledge
        Genre: included when available, omitted for Unknown
    """
    mood_words = _get_mood_words(row['valence'], row['energy'])
    mood_str = ", ".join(mood_words[:3])  # use 3 of the 4 words

    # Acousticness
    if row['acousticness'] > 0.574:
        texture = "acoustic"
    elif row['acousticness'] < 0.104:
        texture = "electronic"
    else:
        texture = "mixed"

    # Instrumentalness
    vocal = "instrumental" if row['instrumentalness'] > 0.5 else "vocal"

    # Genre
    genre = row['genre_top'] if row['genre_top'] != 'Unknown' else ""
    genre_str = f"{genre} " if genre else ""

    return (
        f"A {genre_str}track that feels {mood_str}. "
        f"The music is {texture} and {vocal}."
    )

def _rerank_by_query(results: list, query_text: str, model) -> list:
    """
    Rerank retrieved tracks by semantic similarity to the original query.

    Why this helps:
        FAISS retrieves tracks whose description vectors are closest to
        the query vector. But with a small candidate pool, tracks from
        the same VA region get similar scores even if their descriptions
        differ in emotional nuance. Reranking by direct query-description
        similarity promotes tracks whose specific vocabulary best matches
        the query's emotional language.

    Method:
        Embed the query and each track description independently,
        compute cosine similarity between query and each description,
        sort by that score descending.

    Cost:
        One model.encode() call per result (small — descriptions are
        short). Acceptable for k=5-10 results.
    """
    import numpy as np

    query_vec = model.encode([query_text], convert_to_numpy=True)
    desc_vecs = model.encode(
        [r['description'] for r in results],
        convert_to_numpy=True
    )

    # Cosine similarity
    query_norm = query_vec / np.linalg.norm(query_vec)
    desc_norms = desc_vecs / np.linalg.norm(desc_vecs, axis=1, keepdims=True)
    scores = (desc_norms @ query_norm.T).squeeze()

    # Attach rerank score and sort
    for i, r in enumerate(results):
        r['rerank_score'] = float(scores[i])

    return sorted(results, key=lambda x: x['rerank_score'], reverse=True)

def _embed_text_clap_modal(query_text: str) -> np.ndarray:
    """
    Encode query text via Modal serverless CLAP function.
    
    Replaces local CLAP model loading — Modal runs laion/clap-htsat-fused
    on a T4 GPU and returns a normalized 512-dim vector.
    Cold start: ~15s. Warm: ~1-2s.
    """
    import modal
    f = modal.Function.from_name("mood2music-clap", "encode_text")
    result = f.remote(query_text)
    return np.array(result, dtype=np.float32).reshape(1, -1)
# Mood2Music — Affective Music Retrieval System

> **Status: Active revamp** — rebuilding from prototype to production-grade architecture.

## What this is

A mood-based music recommendation system that retrieves songs based on how a user
feels, not just what genre they want. The core idea: music recommendation is usually
a collaborative filtering problem ("users like you also liked..."), but the more
interesting angle is reasoning about *why* someone wants a particular song in a
given moment — their current affective state.

## Architecture

This system is built around the **valence-arousal model** of emotion (Russell, 1980),
which represents any emotional state as a point in a 2D space:
- **Valence** — how positive or negative the feeling is
- **Arousal** — how activated or calm

### Retrieval pipeline (two-stage)

**Stage 1 — CLAP audio retrieval**
Tracks are embedded using `laion/clap-htsat-fused`, a multimodal model that places
audio and text in a shared vector space. A user's mood query is embedded with the
same model's text encoder. FAISS finds the top-50 most similar tracks.

**Stage 2 — Valence reranking**
CLAP retrieval is strong on the arousal axis but inconsistent on valence (musical
positiveness/negativity), because valence is an abstract signal not directly encoded
in the waveform. A valence penalty reranks CLAP's candidates using Echonest valence
scores — purpose-built numerical measurements of musical positiveness.

The target valence for a query is derived by running the same query through a
sentence-transformer system (Phase 2A) and reading the average valence of its
top-5 results. No hand-coded mood-to-valence mapping.

**Final score:**
final_score = 0.7 × clap_similarity − 0.3 × |track_valence − target_valence|

### Data

- **Catalog:** FMA (Free Music Archive) × Echonest subset — 4,816 tracks with
  both audio files and reliable valence/energy scores
- **Audio features:** Echonest (valence, energy, danceability, acousticness,
  instrumentalness, tempo)
- **Audio embeddings:** `laion/clap-htsat-fused` via Colab Pro (A100)
- **Text embeddings:** `all-MiniLM-L6-v2` (sentence-transformers)

### Why CLAP runs on Colab

Local environment (macOS 26.5 beta / PyTorch 2.12) has MPS backend instability
that crashes the kernel during audio inference. The embedding pipeline runs on
Colab Pro (A100 GPU) and saves artifacts to Google Drive. All retrieval and
serving logic runs locally.

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Data catalog | ✅ Done | FMA + Echonest → clean Parquet catalog, 11,868 tracks |
| 2A — Text retrieval | ✅ Done | Sentence-transformer + FAISS, text-description embeddings |
| 2B — CLAP retrieval | ✅ Done | Audio embeddings via laion/clap-htsat-fused (Colab A100) |
| 2C — Hybrid rerank | ✅ Done | CLAP candidates + dual-axis valence/energy penalty |
| 3 — Mood regulation | ✅ Done | Waypoint playlists navigating the valence-arousal space |
| 4 — Personalization | ⬜ Planned | Implicit feedback + collaborative filtering reranker |
| 5 — Frontend + API | ⬜ Planned | FastAPI backend, React/TypeScript UI |

## Mood Regulation

The mood regulation feature is the most differentiated part of this system. Rather than
matching a static mood, it navigates from a current emotional state to a target state
through the valence-arousal space via linear interpolation.

A user provides two free-text inputs — how they feel now and how they want to feel.
Both are converted to (valence, energy) coordinates using a three-layer mapping:

1. **Phrase overrides** — compound expressions like "low energy" or "at peace" that
   word-level lexicons systematically mishandle
2. **NRC VAD Lexicon** — 20,000 English words rated by humans on valence and arousal
   dimensions (Mohammad & Turney, 2018), with modifier handling for negation,
   intensifiers, and diminishers
3. **Phase 2A fallback** — semantic retrieval for words not covered by the lexicon

Five waypoints are interpolated along the straight line between the two coordinates.
For each waypoint, tracks are retrieved by Euclidean distance in VA space using
Echonest valence and energy scores — no embeddings needed at this stage.

The result is a 10-song playlist that traces a deliberate emotional arc.

## Stack

**ML/Data:** Python 3.11, PyTorch, transformers, sentence-transformers, FAISS,
librosa, DuckDB, Parquet

**Compute:** MacBook Air M5 (local dev), Google Colab Pro A100 (CLAP embedding)

**Planned:** FastAPI, React/TypeScript, MLflow

## Running locally

```bash
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notebooks run in order: `01_data_exploration` → `02_embeddings` → `03_clap_embeddings`
(03 requires Colab — see notebook header for instructions).

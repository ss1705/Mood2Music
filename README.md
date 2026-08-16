### Data

- **Catalog:** FMA (Free Music Archive) × Echonest subset — 11,868 tracks with
  reliable valence/energy scores; 4,816 with audio files for CLAP embedding
- **Audio features:** Echonest (valence, energy, danceability, acousticness,
  instrumentalness, tempo)
- **Audio embeddings:** `laion/clap-htsat-fused` via Colab Pro (A100)
- **Text embeddings:** `all-MiniLM-L6-v2` (sentence-transformers)
- **Track descriptions:** 32-region mood vocabulary (8 valence × 4 energy bands)
  with curated domain-specific affect language

### Why CLAP runs on Colab

Local environment (macOS 26.5 beta / PyTorch 2.12) has MPS backend instability
that crashes the kernel during audio inference. The embedding pipeline runs on
Colab Pro (A100 GPU) and saves artifacts to Google Drive. All retrieval and
serving logic runs on Railway.

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Data catalog | ✅ Done | FMA + Echonest → clean Parquet catalog, 11,868 tracks |
| 2A — Text retrieval | ✅ Done | Sentence-transformer + FAISS, 32-region mood vocabulary |
| 2B — CLAP retrieval | ✅ Done | Audio embeddings via laion/clap-htsat-fused (Colab A100) |
| 2C — Hybrid rerank | ✅ Done | CLAP candidates + dual-axis valence/energy penalty |
| 3 — Mood regulation | ✅ Done | Waypoint playlists navigating the valence-arousal space |
| 4 — Frontend + API | ✅ Done | FastAPI on Railway, React/TypeScript on Vercel |
| 5 — Personalization | ⬜ Planned | Implicit feedback + collaborative filtering reranker |

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
Echonest valence and energy scores.

The result is a 10-song playlist that traces a deliberate emotional arc.

## Known limitations

- **Catalog:** FMA is independent/experimental music. Tracks are linked to YouTube
  search for playback but won't be familiar to most listeners. Extending to a licensed
  mainstream catalog is the production path.
- **CLAP on macOS 26 beta:** MPS instability prevents local CLAP inference. Embeddings
  were generated on Colab Pro; production serving uses Phase 2A text retrieval with
  CLAP-informed reranking via precomputed indexes.
- **Mood regulation coordinates:** Word-level averaging ignores negation ("not happy"
  → treated as "happy"). Rare in practice for direct mood descriptions but worth noting.

## Stack

**ML/Data:** Python 3.11, PyTorch, transformers, sentence-transformers, FAISS,
librosa, DuckDB, Parquet, NRC VAD Lexicon

**Backend:** FastAPI, Railway

**Frontend:** React/TypeScript, Vite, Tailwind CSS, Vercel

**Compute:** MacBook Air M5 (local dev), Google Colab Pro A100 (CLAP embedding)

## Running locally

```bash
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Backend
uvicorn backend.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

Notebooks run in order:
- `01_data_exploration` — FMA + Echonest catalog exploration and cleaning
- `02_embeddings` — sentence-transformer embeddings and Phase 2A FAISS index
- `03_clap_embeddings` — CLAP audio embeddings (requires Colab Pro, see notebook header)
- `04_hybrid_evaluation` — side-by-side comparison of Phase 2A, CLAP, and hybrid retrieval
- `05_mood_regulation` — mood regulation pipeline testing and waypoint validation
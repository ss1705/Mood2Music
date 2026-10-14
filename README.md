# Mood2Music

**Mood-aware music recommendation with semantic retrieval, audio-text embeddings, and valence-energy features.**

[Live demo](https://mood2-music.vercel.app/)

Mood2Music is a music recommendation prototype that takes a listener's current mood into account. Instead of using listening history, it retrieves tracks from a description of how someone feels or builds a playlist that moves from a current mood toward a desired one.

I built it to explore the intersection of music information retrieval and affective computing, particularly how far pretrained representations and existing audio features can take a mood-aware recommender without user interaction data.

## Features

- **Mood Matching:** Enter a description such as "quiet, nostalgic music" to retrieve tracks using semantic text search or an optional CLAP-based hybrid pipeline.
- **Mood Regulation:** Enter how you feel now and how you'd like to feel. The system selects a 10-track playlist along a gradual path through valence-energy space.
- **Track-level context:** Results include metadata and feature-based information. The demo links to YouTube searches for discovery rather than streaming audio directly.

**Try it:** https://mood2-music.vercel.app/

## How it works

### Data

The project uses the [Free Music Archive (FMA)](https://github.com/mdeff/fma) dataset and its associated Echo Nest audio features. These include valence, energy, acousticness, instrumentalness, danceability, and tempo. Energy serves as a practical proxy for perceived musical arousal, not a measurement of a listener's emotional response.

The joined Parquet catalog contains **13,129 tracks**. The backend uses **11,868 tracks** after applying a heuristic filter to extreme valence estimates. Of those, **4,816 tracks** had audio available for CLAP embedding generation. The different index sizes matter when comparing retrieval approaches.

### Mood matching

The retrieval pipeline developed in three stages:

**1. Text-based retrieval (Phase 2A).** A rule-based generator turns track features into short descriptions using a curated vocabulary across 32 valence-energy regions (8 valence bands by 4 energy bands), plus genre, acousticness, and instrumentalness. [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) encodes these descriptions into 384-dimensional vectors stored in a FAISS index. The same encoder embeds user queries for similarity search.

**2. Audio-text retrieval (Phase 2B).** [`laion/clap-htsat-fused`](https://huggingface.co/laion/clap-htsat-fused) embeds recordings and text in a shared 512-dimensional space. Audio embeddings were generated offline on a Colab Pro A100 and indexed with FAISS. This makes it possible to search using information from the recordings themselves, rather than only feature-derived descriptions.

**3. Hybrid reranking (Phase 2C).** In hybrid mode, MiniLM's top five results provide an estimated target valence and energy for the query. CLAP retrieves 50 audio candidates, which are then reranked using a weighted combination of audio-text similarity and distance from those target coordinates:

```text
score = 0.7 * clap_similarity
      - 0.3 * abs(track_valence - target_valence)
      - 0.2 * abs(track_energy  - target_energy)
```

The weights are heuristic. This approach combines CLAP's audio-text representations with explicit affect-related constraints, but the current experiments do not establish that it consistently outperforms text-only retrieval.

### Mood regulation

Mood Regulation maps the starting and desired mood descriptions to valence-energy coordinates using phrase rules, the [NRC VAD Lexicon](https://saifmohammad.com/WebPages/nrc-vad.html), and a text-retrieval fallback. It then interpolates five waypoints and selects two nearby tracks at each waypoint, without repeating tracks.

The resulting playlist follows an intended **trajectory through musical features**. It has not been tested for actual changes in listeners' emotional states.

## System architecture

```text
                        OFFLINE
FMA + Echo Nest features               FMA audio recordings
          |                                      |
Rule-based track descriptions            CLAP audio encoder
          |                                      |
     MiniLM embeddings                    Audio embeddings
          |                                      |
     FAISS text index                     FAISS audio index
          |                                      |
          +------------------+-------------------+
                             |
                         ONLINE
                      User mood input
                             |
            +----------------+----------------+
            |                                 |
       Mood Matching                    Mood Regulation
            |                                 |
  MiniLM text search                 Mood-to-VA mapping
  or CLAP text query                 and interpolation
  (Modal) + reranking                       |
            |                         Waypoint track search
       Ranked tracks                          |
                                      10-track playlist
```

The API is built with **FastAPI** and hosted on **Railway**. The frontend uses **React, TypeScript, and Vite** and is deployed on **Vercel**. Track embeddings are precomputed; optional hybrid search calls **Modal** for query-time CLAP text inference. The backend defaults to text-only search when `USE_CLAP=0`. The live deployment's active setting should be checked separately.

## Evaluation

The saved Phase 2A evaluation uses eight curated mood queries. For each query, it checks whether the top five retrieved tracks fall within manually defined valence and energy ranges.

| Metric | Phase 2A result |
|---|---:|
| Mean valence-energy range alignment at 5 | **57.5%** |
| Mean valence-energy distance | **0.228** |

These are preliminary **feature-level** measures, not human-rated relevance scores. Because the descriptions and evaluation criteria both rely on Echo Nest features, the results do not independently validate emotional perception. The hybrid notebook contains qualitative comparisons, but not a controlled quantitative result establishing a lift over the baselines.

See [`notebooks/06_evaluation.ipynb`](notebooks/06_evaluation.ipynb) and [`notebooks/04_hybrid_evaluation.ipynb`](notebooks/04_hybrid_evaluation.ipynb) for the saved experiments.

## Limitations and next steps

- **Subjective ground truth:** Estimated musical valence and energy are not equivalent to how an individual listener feels. Listener studies would be needed to evaluate mood relevance and regulation outcomes.
- **Catalog coverage:** The FMA subset is limited, and the CLAP index covers fewer tracks than the text index. Retrieval methods should be compared on a common candidate pool.
- **Heuristic decisions:** The valence reliability cutoff, description vocabulary, and reranking weights were not independently optimized or validated. Ablations and sensitivity tests would help.
- **Playlist coherence:** Regulation currently follows a path in valence-energy space; it does not optimize musical transitions, genre continuity, or individual preferences.
- **Personalization:** A useful next step is combining immediate mood signals with longer-term listening behavior.

## Running locally

Use Python 3.11 and Node.js/npm. The repository includes the prepared catalog and indexes, but not the original FMA audio recordings.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Set the frontend API endpoint in `frontend/src/api/client.ts` for your local backend. By default, the backend uses Phase 2A (`USE_CLAP=0`). Hybrid mode requires `USE_CLAP=1` and access to the deployed Modal CLAP text encoder. Rebuilding the audio index requires obtaining the FMA audio files separately.

## Repository guide

| Location | Purpose |
|---|---|
| [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) | FMA/Echo Nest join, exploration, preprocessing |
| [`notebooks/02_embeddings.ipynb`](notebooks/02_embeddings.ipynb) | Track descriptions and MiniLM index |
| [`notebooks/03_clap_embeddings.ipynb`](notebooks/03_clap_embeddings.ipynb) | CLAP audio embedding generation |
| [`notebooks/04_hybrid_evaluation.ipynb`](notebooks/04_hybrid_evaluation.ipynb) | Hybrid retrieval experiments |
| [`notebooks/05_mood_regulation.ipynb`](notebooks/05_mood_regulation.ipynb) | Mood mapping and waypoint playlists |
| [`notebooks/06_evaluation.ipynb`](notebooks/06_evaluation.ipynb) | Phase 2A evaluation |
| [`src/retrieval/retriever.py`](src/retrieval/retriever.py) | Search and mood-regulation logic |
| [`src/modal_clap.py`](src/modal_clap.py) | CLAP text inference on Modal |
| [`backend/`](backend/) | FastAPI service |
| [`frontend/`](frontend/) | React application |

## References

- Russell, J. A. (1980). *A Circumplex Model of Affect.*
- Defferrard et al. (2017). *FMA: A Dataset for Music Analysis.* [Dataset and code](https://github.com/mdeff/fma)
- [NRC Valence, Arousal, and Dominance Lexicon](https://saifmohammad.com/WebPages/nrc-vad.html)
- [Sentence Transformers: all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [LAION CLAP: clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused)
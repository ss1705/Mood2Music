# Mood2Music

**Mood-aware music discovery and emotion-guided playlist sequencing.**

[**Try the live demo**](https://mood2-music.vercel.app/) · [**Explore the source**](https://github.com/ss1705/mood2music)

Most recommenders learn from *what you usually listen to*. Mood2Music explores a complementary question: **what kind of music fits how you feel right now—and where you'd like your mood to go?** Built as an exploratory music information retrieval project, it combines natural-language queries, pretrained audio/text representations, and interpretable valence–energy features.

## What it does

**Mood matching.** Describe a mood in natural language (for example, “something gentle and nostalgic”) to retrieve tracks using semantic text retrieval or, when enabled, hybrid audio–text retrieval with affect-related reranking.

**Mood regulation.** Enter how you feel now and how you want to feel. The system constructs a **10-track sequence by default**, selecting two tracks at each of five interpolated waypoints in a two-dimensional valence–energy space. This is an *emotion-guided playlist trajectory*, **not** a demonstrated clinical or psychological intervention.

> **Try it:** [mood2-music.vercel.app](https://mood2-music.vercel.app/)  
> The demo depends on the availability and configuration of the deployed backend. Track playback links are YouTube searches, not embedded or guaranteed licensed streams.

## How it works

```text
                        Free Music Archive + Echonest
                                     |
                           Catalog preprocessing
                                     |
                       Parquet + filtered track metadata
                                     |
              +----------------------+----------------------+
              |                                             |
    Rule-based track descriptions                   Available FMA audio
              |                                             |
     MiniLM text embeddings                          CLAP audio embeddings
      (384 dimensions)                                (512 dimensions)
              |                                             |
      FAISS text index                                 FAISS audio index
       (11,868 tracks)                                  (4,816 tracks)
              |                                             |
              +-------------------+-------------------------+
                                  |
                     User's natural-language mood
                                  |
                 +----------------+----------------+
                 |                                 |
            Mood matching                     Mood regulation
                 |                                 |
       MiniLM query embedding               Phrase rules / NRC VAD
                 |                            / text fallback
       Text-only search OR                         |
       CLAP text query on Modal               Start + target VA
                 |                                 |
       CLAP top-50 candidates               Five interpolated points
                 |                                 |
       Valence/energy reranking             Nearest tracks per point
                 |                                 |
         Ranked track results                  Ordered playlist
```

### 1. Catalog and affect-related features

The [FMA research dataset](https://github.com/mdeff/fma) supplies music metadata and available recordings. Its associated Echo Nest table provides precomputed descriptors such as **valence, energy, acousticness, instrumentalness, danceability, and tempo**. The joined Parquet catalog contains **13,129 tracks**; the serving catalog filters to **11,868** using a heuristic flag that excludes valence estimates outside **[0.05, 0.95]**. This threshold is *not* a validated measure of label confidence.

The project uses **musical energy as a proxy for emotional arousal**, not as a direct measure of a listener's physiological or experienced emotion. Similarly, Echo Nest valence estimates are not human emotion annotations.

### 2. Phase 2A — Semantic text retrieval

A rule-based generator converts track features and genre metadata into descriptions using a curated **32-region vocabulary** (eight valence bands × four energy bands), plus acousticness/instrumentalness labels. `sentence-transformers/all-MiniLM-L6-v2` embeds descriptions and user queries into a shared **384-dimensional text space**. L2-normalized embeddings are searched using an exact FAISS inner-product index (`IndexFlatIP`), equivalent to cosine similarity for normalized vectors.

This makes feature-derived descriptions searchable through paraphrases, but it cannot recover acoustic details omitted by the handcrafted descriptions.

### 3. Phase 2B — CLAP audio–text retrieval

`laion/clap-htsat-fused` encodes available FMA recordings into **512-dimensional audio embeddings**. These were precomputed offline on a **Google Colab Pro A100** and stored in a separate FAISS index covering **4,816 tracks** with available audio. At query time, the optional hybrid path encodes the user's text with CLAP via a **Modal GPU endpoint**; the backend searches the precomputed audio index.

The text and audio indexes have different catalog coverage. Comparisons between them therefore require a shared candidate set to be methodologically fair.

### 4. Phase 2C — Hybrid retrieve and rerank

The hybrid path uses **CLAP to retrieve 50 audio candidates** and **Phase 2A's top five results** to estimate the query's target valence and energy (their mean values). It then reranks the CLAP candidates with:

```text
score = 0.7 * clap_similarity
      - 0.3 * abs(track_valence - target_valence)
      - 0.2 * abs(track_energy  - target_energy)
```

This combines audio–text semantic similarity with explicit emotional-coordinate constraints. The weights are **heuristic**, not established as optimal through a documented validation sweep. Reranking cannot recover tracks outside the initial CLAP candidate pool.

**Serving behavior:** `/search` uses text-only retrieval when `USE_CLAP=0` (the code default) and hybrid retrieval when `USE_CLAP=1`. The live deployment's current flag value cannot be established from source code alone.

### 5. Mood regulation

The `/regulate` route maps both the starting and desired moods to valence–energy coordinates through **phrase overrides**, the **NRC VAD Lexicon** with word/modifier heuristics, and a **Phase 2A fallback**. It linearly interpolates five waypoints, then greedily selects nearby tracks by Euclidean distance in the filtered catalog, avoiding duplicate track IDs.

The resulting ordering follows an intended *feature-space* transition; it has **not** been evaluated for actual changes in listeners' emotional states. It does not jointly optimize transitions for harmony, genre, or musical continuity.

## Evaluation and findings

The repository includes six notebooks. The saved **Phase 2A evaluation** tests eight manually specified mood prompts, using top-five agreement with hand-selected valence–energy ranges. Its reported mean **VA-range alignment is 0.575 (57.5%)** and mean VA distance is **0.228**. These are *proxy, feature-based metrics*, not independent human-rated relevance scores.

The hybrid notebook contains qualitative comparisons of text-only, CLAP-only, and hybrid outputs on four example prompts. **It does not establish a quantitative, controlled improvement for hybrid over either baseline.** In particular, results should not be compared as though the 11,868-track text index and 4,816-track audio index cover the same candidate universe.

**What the project demonstrates:** a functioning prototype for natural-language mood retrieval, complementary content representations, an interpretable reranking strategy, and affect-guided playlist sequencing. **What it does not demonstrate:** validated emotion induction, production-scale recommendation gains, or statistically significant hybrid superiority.

## Stack and deployment

| Layer | Technologies |
|---|---|
| Data | Python, pandas, Parquet, DuckDB, FMA/Echonest |
| Text representation | Sentence Transformers, `all-MiniLM-L6-v2` |
| Audio representation | PyTorch, Hugging Face Transformers, `laion/clap-htsat-fused` |
| Retrieval | FAISS exact inner-product indexes, feature-based reranking |
| Mood mapping | NRC VAD Lexicon, phrase heuristics, VA interpolation |
| API | FastAPI on Railway |
| Query-time CLAP inference | Modal serverless GPU, when hybrid mode is enabled |
| Web app | React, TypeScript, Vite, Vercel |
| Offline audio processing | Google Colab Pro A100 |

**Compute placement:** track descriptions and catalog embeddings are prepared offline. The serving backend loads saved indexes; user-specific text embeddings are computed at request time. Hybrid queries depend on remote CLAP text inference and can experience GPU cold-start latency. Exact deployed latency and active feature flags should be verified using service logs.

## Repository guide

| Location | Purpose |
|---|---|
| `notebooks/01_data_exploration.ipynb` | FMA/Echonest join, distributions, reliability flag, Parquet |
| `notebooks/02_embeddings.ipynb` | Feature-to-text descriptions, MiniLM embeddings, FAISS |
| `notebooks/03_clap_embeddings.ipynb` | Audio preprocessing and CLAP indexing |
| `notebooks/04_hybrid_evaluation.ipynb` | Qualitative retrieval comparisons and hybrid scoring |
| `notebooks/05_mood_regulation.ipynb` | Mood-to-coordinate mapping and waypoint examples |
| `notebooks/06_evaluation.ipynb` | Phase 2A feature-based evaluation |
| `src/retrieval/retriever.py` | Retrieval, reranking, mood mapping, playlist generation |
| `src/modal_clap.py` | Remote CLAP text-embedding endpoint |
| `backend/` | FastAPI endpoints, schemas, dependency loading |
| `frontend/` | React application and visualizations |
| `data/` | Saved catalog, indexes, ID maps, and lexicon |

## Run locally

Use **Python 3.11** and Node.js/npm. The included `requirements.txt` is a development-environment snapshot; some dependencies may be platform-specific.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend's API base URL is configured in `frontend/src/api/client.ts`; point it to the backend you intend to use. For hybrid retrieval, configure `USE_CLAP=1` **and** deploy/configure the Modal CLAP function with the necessary access; without that setup, leave `USE_CLAP=0` for text-only mood matching. The `/regulate` endpoint is separate from the CLAP flag. Reproducing the CLAP audio index from scratch requires the original FMA audio files, which are not included in this repository snapshot.

## Limitations and next steps

- **Subjective emotion:** feature-based valence/energy alignment is not the same as perceived or induced emotion; human listening studies would be needed for stronger claims.
- **Coverage:** the audio-covered CLAP subset is smaller than the text catalog. Future experiments should use matched candidate sets and report coverage explicitly.
- **Evaluation:** add held-out mood prompts, human relevance ratings, uncertainty estimates, and ablations of reranking weights and coordinate mapping.
- **Mood mapping:** heuristic phrase handling, negation, and lexicon averaging can fail on complex expressions.
- **Serving:** test cold-start latency, fallback behavior, caching, and monitoring before treating hybrid inference as production-ready.
- **Personalization:** incorporate listening history and feedback to balance immediate mood with longer-term taste.
- **Catalog/playback:** FMA is a research catalog; YouTube search links do not guarantee playback availability or rights.

## Research foundations

- Russell, J. A. (1980). *A Circumplex Model of Affect.*
- Defferrard et al. (2017). *FMA: A Dataset for Music Analysis.* [Dataset/repository](https://github.com/mdeff/fma)
- [LAION CLAP / `laion/clap-htsat-fused`](https://huggingface.co/laion/clap-htsat-fused)
- [Sentence Transformers / `all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [NRC Valence, Arousal, and Dominance Lexicon](https://saifmohammad.com/WebPages/nrc-vad.html)

---

*An exploratory research prototype at the intersection of music information retrieval, representation learning, and affect-aware recommendation.*
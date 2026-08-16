"""
backend/main.py

FastAPI application — three endpoints:
  POST /search    — mood matching via hybrid CLAP reranker
  POST /regulate  — mood regulation via VA waypoint playlists
  GET  /health    — startup status check

All ML logic lives in src/retrieval/retriever.py.
This file handles HTTP, validation, and error responses only.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.models import (
    SearchRequest, SearchResponse, Track, SearchMeta,
    RegulateRequest, RegulateResponse, Waypoint, RegulationMeta
)
from backend.dependencies import load_all_artifacts
from src.retrieval.retriever import search_hybrid, search_regulation
import os
import traceback

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup, store in app.state."""
    print("Starting Mood2Music API...")
    app.state.artifacts = load_all_artifacts()
    print("All artifacts loaded. Ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Mood2Music API",
    description="Affective music retrieval and mood regulation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",  # Vite dev server
                   "https://mood2-music.vercel.app"],  # production
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_artifacts(request: Request) -> dict:
    return request.app.state.artifacts


@app.get("/health")
def health(request: Request):
    artifacts = get_artifacts(request)
    return {
        "status": "ok",
        "models_loaded": artifacts is not None,
        "catalog_size": len(artifacts['catalog'])
    }

# Feature flag — set CLAP=1 in production environment (Modal)
USE_CLAP = os.getenv("USE_CLAP", "0") == "1"

# @app.post("/search", response_model=SearchResponse)
# def search(body: SearchRequest, request: Request):
#     a = get_artifacts(request)

#     try:
#         if USE_CLAP:
#             results = search_hybrid(
#                 query_text=body.query,
#                 model_2a=a['model_2a'],
#                 index_2a=a['index_2a'],
#                 map_2a=a['map_2a'],
#                 model_clap=a['model_clap'],
#                 processor_clap=a['processor_clap'],
#                 device_clap=a['device_clap'],
#                 index_clap=a['index_clap'],
#                 map_clap=a['map_clap'],
#                 catalog=a['catalog'],
#                 k=body.k
#             )
#         else:
#             # Phase 2A only — used during local development
#             # CLAP hybrid enabled in production via USE_CLAP=1 env var
#             from src.retrieval.retriever import search_2a
#             results = search_2a(
#                 query_text=body.query,
#                 model=a['model_2a'],
#                 index=a['index_2a'],
#                 track_id_map=a['map_2a'],
#                 catalog=a['catalog'],
#                 k=body.k
#             )
#             # Add dummy target coords for response schema compatibility
#             for r in results:
#                 r['target_valence'] = round(
#                     float(a['catalog']['valence'].mean()), 3)
#                 r['target_energy'] = round(
#                     float(a['catalog']['energy'].mean()), 3)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     tracks = [Track(**{k: v for k, v in r.items()
#                        if k in Track.model_fields}) for r in results]

#     meta = SearchMeta(
#         system="hybrid_clap" if USE_CLAP else "2a_text",
#         target_valence=results[0]['target_valence'],
#         target_energy=results[0]['target_energy'],
#         query=body.query
#     )

#     return SearchResponse(tracks=tracks, meta=meta)

@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request):
    a = get_artifacts(request)

    try:
        if USE_CLAP:
            results = search_hybrid(
                query_text=body.query,
                model_2a=a['model_2a'],
                index_2a=a['index_2a'],
                map_2a=a['map_2a'],
                model_clap=None,
                processor_clap=None,
                device_clap=None,
                index_clap=a['index_clap'],
                map_clap=a['map_clap'],
                catalog=a['catalog'],
                k=body.k
            )
        else:
            from src.retrieval.retriever import search_2a
            results = search_2a(
                query_text=body.query,
                model=a['model_2a'],
                index=a['index_2a'],
                track_id_map=a['map_2a'],
                catalog=a['catalog'],
                k=body.k
            )
            for r in results:
                r['target_valence'] = round(float(a['catalog']['valence'].mean()), 3)
                r['target_energy'] = round(float(a['catalog']['energy'].mean()), 3)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    tracks = [Track(**{k: v for k, v in r.items()
                       if k in Track.model_fields}) for r in results]

    meta = SearchMeta(
        system='hybrid_clap' if USE_CLAP else '2a_text',
        target_valence=results[0]['target_valence'],
        target_energy=results[0]['target_energy'],
        query=body.query
    )

    return SearchResponse(tracks=tracks, meta=meta)

@app.post("/regulate", response_model=RegulateResponse)
def regulate(body: RegulateRequest, request: Request):
    """
    Mood regulation via VA waypoint playlists.

    Converts current and target mood text to VA coordinates using
    the three-layer mapping (phrase overrides → VAD lexicon →
    Phase 2A fallback), interpolates waypoints, retrieves tracks
    by Euclidean distance at each step.
    """
    a = get_artifacts(request)

    try:
        result = search_regulation(
            current_mood=body.current_mood,
            target_mood=body.target_mood,
            model_2a=a['model_2a'],
            index_2a=a['index_2a'],
            map_2a=a['map_2a'],
            catalog=a['catalog'],
            n_waypoints=body.n_waypoints,
            tracks_per_waypoint=body.tracks_per_waypoint
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    waypoints = [
        Waypoint(
            step=wp['step'],
            valence=wp['valence'],
            energy=wp['energy'],
            tracks=[Track(**{k: v for k, v in t.items()
                             if k in Track.model_fields})
                    for t in wp['tracks']]
        )
        for wp in result['waypoints']
    ]

    meta = RegulationMeta(
        current_mood=result['current_mood'],
        target_mood=result['target_mood'],
        current_coords=result['current_coords'],
        target_coords=result['target_coords'],
        n_waypoints=body.n_waypoints
    )

    return RegulateResponse(waypoints=waypoints, meta=meta)
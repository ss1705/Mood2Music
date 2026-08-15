"""
backend/models.py

Pydantic schemas for request validation and response serialization.
Keeping these separate from main.py means the API contract is explicit
and documented independently of the endpoint logic.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Request schemas ───────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Free-text mood description",
        examples=["I feel melancholy and tired"]
    )
    k: int = Field(default=5, ge=1, le=20)


class RegulateRequest(BaseModel):
    current_mood: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="How the user feels now",
        examples=["I feel stressed and overwhelmed"]
    )
    target_mood: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="How the user wants to feel",
        examples=["I want to feel calm and at peace"]
    )
    n_waypoints: int = Field(default=5, ge=3, le=7)
    tracks_per_waypoint: int = Field(default=2, ge=1, le=3)


# ── Response schemas ──────────────────────────────────────────────────────────

class Track(BaseModel):
    track_id:   int
    title:      str
    artist:     str
    genre:      str
    valence:    float
    energy:     float
    youtube_url: str


class SearchMeta(BaseModel):
    system:         str
    target_valence: float
    target_energy:  float
    query:          str


class SearchResponse(BaseModel):
    tracks: list[Track]
    meta:   SearchMeta


class Waypoint(BaseModel):
    step:    int
    valence: float
    energy:  float
    tracks:  list[Track]


class RegulationMeta(BaseModel):
    current_mood:   str
    target_mood:    str
    current_coords: tuple[float, float]
    target_coords:  tuple[float, float]
    n_waypoints:    int


class RegulateResponse(BaseModel):
    waypoints: list[Waypoint]
    meta:      RegulationMeta
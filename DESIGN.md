# Mood2Music — Design Document

## 1. Problem Framing

This project explores **affect-aware music recommendation** by mapping free-text mood descriptions into a continuous **valence–arousal space** and retrieving Spotify tracks whose audio features best **match or regulate** the user’s emotional state.

Unlike traditional recommender systems that predict user preferences or optimize for engagement, Mood2Music frames recommendation as a **human-centered affective alignment problem**. The system aims to understand *how a user feels* and recommend music that either reinforces or gently shifts that mood.

---

## 2. Key Assumptions

* **Mood is continuous, not categorical**: Emotional states cannot be reliably represented using discrete labels such as “happy” or “sad”.
* **Text can express mixed emotions**: Users often describe overlapping or ambiguous emotional states (e.g., “nostalgic but hopeful”).
* **Audio features approximate affect**: Spotify’s audio features (e.g., valence, energy) provide meaningful proxies for emotional qualities of music.
* **Recommendation is retrieval, not prediction**: The task is framed as retrieving songs close to a target affective state rather than predicting what the user will like.

These assumptions align with research in affective computing, music information retrieval (MIR), and Spotify’s public research on emotion-aware personalization.

---

## 3. Mood Representation Choice

### 3.1 Valence–Arousal Model

Mood2Music adopts **Russell’s Circumplex Model of Affect**, which represents emotion along two continuous dimensions:

* **Valence**: positive ↔ negative emotional tone
* **Arousal**: low ↔ high physiological activation

This representation is preferred over categorical emotion labels because:

* It captures emotional intensity and ambiguity
* It supports smooth transitions between emotional states
* It aligns naturally with Spotify’s `valence` and `energy` audio features

### 3.2 Alignment with Spotify Audio Features

Spotify provides per-track audio features including:

* `valence` → emotional positivity
* `energy` → perceived intensity
* `tempo`, `acousticness`, `danceability` → secondary affective cues

Mapping mood into valence–arousal space allows direct comparison between inferred user mood and track-level affective representations.

---

## 4. System Overview

```
Text Input
   ↓
Multi-label Emotion Classification
   ↓
Emotion → Valence/Arousal Projection
   ↓
Affective-Space Retrieval (KNN)
   ↓
Ranked Track Recommendations
```

---

## 5. Model & Methodology Choices

### 5.1 Text → Emotion Inference

* **Model**: `j-hartmann/emotion-english-distilroberta-base`
* **Rationale**:

  * Trained on emotion-specific data (GoEmotions)
  * Supports multi-label outputs
  * Lightweight and suitable for rapid prototyping

Rather than predicting a single dominant emotion, the model outputs a distribution over multiple emotions, preserving ambiguity and emotional nuance.

### 5.2 Emotion → Affect Projection

Emotion probabilities are projected into valence–arousal space using a **linear weighted mapping**. Each emotion contributes proportionally to the final affective vector.

This approach is:

* Interpretable
* Consistent with affective psychology literature
* Easy to extend or learn via optimization

### 5.3 Affective Retrieval

* **Representation**: Each song is embedded as a vector of normalized audio features (initially valence and energy).
* **Retrieval Method**: Cosine similarity / K-nearest neighbors

Recommendation is framed as finding songs closest to the target affective vector in continuous space.

### 5.4 Use of JAX

JAX is used to:

* Experiment with **differentiable emotion-to-affect mappings**
* Optimize emotion weights via gradient-based loss functions

This positions JAX as a research tool for affective modeling rather than a general ML framework replacement.

---

## 6. Mood Regulation vs Mood Matching

Mood2Music supports two conceptual modes:

* **Mood Matching**: Retrieve songs close to the user’s current affective state
* **Mood Regulation**: Retrieve songs that gradually shift affect (e.g., low arousal → moderate arousal)

This distinction is motivated by research in music psychology and aligns with Spotify’s interest in wellness, focus, and context-aware listening.

---

## 7. Limitations

* Emotion-to-affect mapping is heuristic and not learned from user feedback
* No personalization based on listening history
* Dataset size is limited and static
* Cultural and individual differences in emotional perception are not modeled

These limitations are accepted to prioritize interpretability and rapid prototyping.

---

## 8. Future Extensions

* Learn emotion–affect mappings from mood-labeled playlists
* Incorporate audio embeddings instead of hand-crafted features
* Add user-level personalization and feedback loops
* Extend to podcasts and spoken content
* Deploy a TypeScript-based frontend for interactive exploration

---

## 9. Summary

Mood2Music demonstrates a research-oriented approach to music recommendation by treating emotion as a continuous, interpretable signal and framing recommendation as affective retrieval. The project emphasizes clarity of assumptions, alignment with Spotify’s audio representations, and human-centered AI design over predictive optimization.

# Mood2Music

**Mood2Music** is a basic full-stack prototype that recommends music based on your current mood.  
It combines **emotion analysis** from text input with **audio feature mapping** to suggest a mood-aligned song.  

> This is a prototype and under active development. Many features can be improved or expanded.

## Features (Prototype)

- Text-based emotion detection using a HuggingFace model
- Mapping detected emotions to simple audio features (valence, energy, tempo)
- FastAPI backend serving emotion analysis and music recommendation data
- React + TypeScript frontend to submit moods and display results
- Visual representation of audio features (radar chart)

## Project Structure
```
mood2music/
├── backend/ # FastAPI backend
│ ├── app.py
│ ├── emotion_classifier.py
│ ├── emo_2_aud.py
│ └── requirements.txt
├── frontend/ # React + TypeScript frontend
│ ├── src/
│ │ ├── App.tsx
│ │ └── components/
│ │ └── MoodForm.tsx
│ └── package.json
├── README.md
└── DESIGN.md
```

## Usage

1. Open the frontend in a browser.
2. Enter a mood or a short description of your day.
3. The app will detect emotions, map them to audio features, and display a simple visual.

Currently, this is a basic prototype, so the music recommendation is simplified. Full features (actual song recommendations, personalized playlists, multi-language support, etc.) can be added in future development.

## Notes
- The project uses a pre-trained HuggingFace model for emotion detection.
- Audio feature mapping is a simplified demonstration, not real song playback.
- This project is a proof-of-concept to explore AI + frontend + backend integration.

## Future Improvements
- Integrate actual music recommendation API (Spotify / other services)
- Enhance emotion analysis with context and multiple sentences
- Improve frontend UI/UX
- Add user authentication and session tracking
- Deploy as a full-stack cloud application

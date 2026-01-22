from fastapi import FastAPI
from pydantic import BaseModel
from emotion_classifier import infer_emotions
from emo_2_aud import map_emo_2_aud
from fastapi.middleware.cors import CORSMiddleware

class TextInput(BaseModel):
    text: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Mood2Music API running!"}

@app.post("/mood")
def mood_endpoint(input: TextInput):
    emotions = infer_emotions(input.text)
    audio_features = map_emo_2_aud(emotions)
    return {
        "emotions": emotions,
        "audio_features": audio_features
    }
from typing import Dict

EMOTION_TO_AUDIO = {
    "joy":        {"valence": 0.9,  "energy": 0.7, "tempo": 120},
    "excitement": {"valence": 0.95, "energy": 0.9, "tempo": 140},
    "sadness":    {"valence": 0.2,  "energy": 0.3, "tempo": 80},
    "fear":       {"valence": 0.3,  "energy": 0.6, "tempo": 100},
    "anger":      {"valence": 0.4,  "energy": 0.8, "tempo": 130},
    "neutral":    {"valence": 0.5,  "energy": 0.5, "tempo": 100},
    "surprise":   {"valence": 0.8,  "energy": 0.6, "tempo": 110},
    "disgust":    {"valence": 0.3,  "energy": 0.4, "tempo": 90},
}

def map_emo_2_aud(emotions:Dict[str,float]) -> Dict[str,float]:
    valence = 0.0
    energy = 0.0
    tempo = 0.0
    total = sum(emotions.values())

    if total==0:
        return {"valence": 0.5, "energy": 0.5, "tempo": 100}
    
    for emotion,score in emotions.items():
        weight = score/total
        mapping = EMOTION_TO_AUDIO.get(emotion,EMOTION_TO_AUDIO["neutral"])
        valence += mapping["valence"]*weight
        energy += mapping["energy"]*weight
        tempo += mapping["tempo"]*weight

    return {"valence":valence, "energy":energy, "tempo": tempo}

if __name__ == "__main__":
    sample = {'joy': 0.8549, 'sadness': 0.0844, 'fear': 0.0607}
    audio_features = map_emo_2_aud(sample)
    print(audio_features)
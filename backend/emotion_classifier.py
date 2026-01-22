from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    framework="pt",
    top_k=None
)

def infer_emotions(text:str, threshold:float=0.05):
    raw_ops = classifier(text)[0]
    emotions = {
        item["label"]: item["score"]
        for item in raw_ops if item["score"]>=threshold
    }

    total = sum(emotions.values())
    if total>0:
        emotions = {k: v / total for k, v in emotions.items()}

    return emotions

if __name__=="__main__":
    text = "Today was stressful but exciting"
    emotions = infer_emotions(text)
    print(emotions)

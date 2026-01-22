import requests

url = "http://127.0.0.1:8000/mood"
data = {"text": "Today was stressful but exciting"}

response = requests.post(url, json=data)
print(response.json())

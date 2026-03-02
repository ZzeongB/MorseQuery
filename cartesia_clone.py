import requests

url = "https://api.cartesia.ai/voices/clone"

files = {
    "clip": (
        "live/logs/audio/t6BSGLLizDhxwzwhAAAB/summary_0.mp3",
        open("live/logs/audio/t6BSGLLizDhxwzwhAAAB/summary_0.mp3", "rb"),
    )
}
payload = {
    "name": "Me",
    "description": "My cloned voice",
    "language": "en",
}
headers = {
    "Cartesia-Version": "2025-04-16",
    "Authorization": "Bearer sk_car_nmPGF6jF7Mo7bBba5r5kgF",
}

response = requests.post(url, data=payload, files=files, headers=headers)

print(response.text)

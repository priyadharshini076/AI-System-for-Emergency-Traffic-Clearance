import requests

url = 'https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.1/yolov3.pt'
response = requests.get(url)
with open('yolov3.pt', 'wb') as f:
    f.write(response.content)
print("Downloaded yolov3.pt")

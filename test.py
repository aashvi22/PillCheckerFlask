import requests
import json

resp = requests.post("http://localhost:5000/", files={'file': open('website/testimage_pill6.jpg', 'rb')})

print(resp.json())
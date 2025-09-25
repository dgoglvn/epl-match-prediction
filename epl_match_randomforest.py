from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import requests
from dotenv import load_dotenv
import os

load_dotenv()

url = "https://fbrapi.com/league-standings"
params = {"league_id": "9"}
headers = {"X-API-Key": os.getenv("FBR_API_KEY")}

response = requests.get(url, params=params, headers=headers)
print(response.json())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from mangum import Mangum
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or replace "*" with specific origins like ['https://yourfrontend.com']
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, or restrict to specific methods like ['GET', 'POST']
    allow_headers=["*"],  # Allow all headers
)
# Load data
df2 = pd.read_csv("app/final.csv")
makeup = pd.read_csv("app/makeup_final.csv")
features = ['normal', 'dry', 'oily', 'combination', 'acne', 'sensitive', 'fine lines', 'wrinkles', 
            'redness', 'dull', 'pore', 'pigmentation', 'blackheads', 'whiteheads', 'blemishes', 
            'dark circles', 'eye bags', 'dark spots']

# Preprocess: One-hot encoding
entries = len(df2)
one_hot_encodings = np.zeros([entries, len(features)])
for i in range(entries):
    for j in range(5):
        target = features[j]
        sk_type = df2.iloc[i]['skin type']
        if sk_type == 'all':
            one_hot_encodings[i][0:5] = 1
        elif target == sk_type:
            one_hot_encodings[i][j] = 1
for i in range(entries):
    for j in range(5, len(features)):
        feature = features[j]
        if feature in df2.iloc[i]['concern']:
            one_hot_encodings[i][j] = 1

# Models
class RecommendationRequest(BaseModel):
    vector: list
    skin_tone: str = None
    skin_type: str = None

# Helper function
def wrap(info_arr):
    return {
        "brand": info_arr[0],
        "name": info_arr[1],
        "price": info_arr[2],
        "url": info_arr[3],
        "img": info_arr[4],
        "skin type": info_arr[5],
        "concern": str(info_arr[6]).split(','),
    }

# API endpoints
@app.post("/essentials")
async def recommend_essentials(req: RecommendationRequest):
    vector = np.array(req.vector)
    cs_values = cosine_similarity([vector], one_hot_encodings)
    df2['cs'] = cs_values[0]
    recommendations = df2.sort_values('cs', ascending=False).head(5)
    return [wrap(row) for _, row in recommendations.iterrows()]

@app.post("/makeup")
async def recommend_makeup(req: RecommendationRequest):
    skin_tone, skin_type = req.skin_tone, req.skin_type
    dfs_to_concat = [
        makeup[(makeup['skin tone'] == skin_tone) & (makeup['skin type'] == skin_type) & (makeup['label'] == 'foundation')].head(2),
        makeup[(makeup['skin tone'] == skin_tone) & (makeup['skin type'] == skin_type) & (makeup['label'] == 'concealer')].head(2),
        makeup[(makeup['skin tone'] == skin_tone) & (makeup['skin type'] == skin_type) & (makeup['label'] == 'primer')].head(2)
    ]
    dff = pd.concat(dfs_to_concat, ignore_index=True).sample(frac=1)
    return dff.to_dict(orient='records')

handler = Mangum(app)

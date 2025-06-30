# segment.py
import requests
from PIL import Image
import base64
import json
import os
from dotenv import load_dotenv

SAM2_MODEL_VERSION = "fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83"
SAM2_API_URL = "https://api.replicate.com/v1/predictions"

load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
SAM2_ENDPOINT = "https://api.replicate.com/v1/predictions"

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def get_mask_sam2(image_path, boxes):
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }

    image_base64 = image_to_base64(image_path)
    prompt = {
        "image": f"data:image/jpeg;base64,{image_base64}",
        "boxes": [{"x": float(box[1][0]), "y": float(box[1][1]), "width": float(box[1][2]-box[1][0]), "height": float(box[1][3]-box[1][1])} for box in boxes]
    }

    payload = {
        "version": "fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
        "input": prompt
    }

    response = requests.post(SAM2_ENDPOINT, headers=headers, json=payload)
    return response.json()

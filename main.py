import base64
import io
import os
import re

import openai
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



extractor = AutoFeatureExtractor.from_pretrained("kmewhort/resnet34-sketch-classifier")
model = AutoModelForImageClassification.from_pretrained("kmewhort/resnet34-sketch-classifier")

@app.post("/recognize")
async def recognize_sketch(image_data_url: str = Query(..., description="Data URL of the image from the canvas")):
    try:
        print("Received image_data_url:", image_data_url)
        
        image_data = re.sub('^data:image/.+;base64,', '', image_data_url)
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        inputs = extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits

        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

        return {"predicted_class": predicted_class}
    except Exception as e:
        return {"error": "An error occurred during classification."}
    
@app.post("/recognizee")
async def recognize_sketch(image_data_url: str = Query(..., description="Data URL of the image from the canvas")):
    try:
        print("Received image_data_url:", image_data_url)

        image_data = re.sub('^data:image/.+;base64,', '', image_data_url)
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        inputs = extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits

        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

        return {"predicted_class": predicted_class}
    except Exception as e:
        return {"error": "An error occurred during classification."}


# extractor = AutoFeatureExtractor.from_pretrained("kmewhort/beit-sketch-classifier")
# model = AutoModelForImageClassification.from_pretrained("kmewhort/beit-sketch-classifier")

# @app.post("/recognize")
# async def recognize_sketch(image_data_url: str = Query(..., description="Data URL of the image from the canvas")):
    
#     try:
#         print("Received image_data_url:", image_data_url)
#         # Convert the data URL to an image
#         image_data = re.sub('^data:image/.+;base64,', '', image_data_url)
#         image = Image.open(io.BytesIO(base64.b64decode(image_data)))

#         # Perform image classification
#         inputs = extractor(images=image, return_tensors="pt")
#         outputs = model(**inputs)
#         logits = outputs.logits
#         # model predicts one of the 21,841 ImageNet-22k classes
#         predicted_class_idx = logits.argmax(-1).item()
#         predicted_class = model.config.id2label[predicted_class_idx]

#         return {"predicted_class": predicted_class}
#     except Exception as e:
#         return {"error": "An error occurred during classification."}
    
# @app.post("/recognizee")
# async def recognize_sketch(image_data_url: str = Query(..., description="Data URL of the image from the canvas")):
    
#     try:
#         print("Received image_data_url:", image_data_url)
#         # Convert the data URL to an image
#         image_data = re.sub('^data:image/.+;base64,', '', image_data_url)
#         image = Image.open(io.BytesIO(base64.b64decode(image_data)))

#         # Perform image classification
#         inputs = extractor(images=image, return_tensors="pt")
#         outputs = model(**inputs)
#         logits = outputs.logits
#         # model predicts one of the 21,841 ImageNet-22k classes
#         predicted_class_idx = logits.argmax(-1).item()
#         predicted_class = model.config.id2label[predicted_class_idx]

#         return {"predicted_class": predicted_class}
#     except Exception as e:
#         return {"error": "An error occurred during classification."}
    

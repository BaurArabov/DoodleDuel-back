import base64
import io
import json
import os
import random
import re

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from matplotlib.pyplot import imshow
from PIL import Image

# from transformers import AutoFeatureExtractor, AutoModelForImageClassification
# from transformers import BeitForImageClassification, BeitImageProcessor

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


class DrawSketch(object):
    def __init__(self):
        pass

    def scale_sketch(self, sketch, size=(448, 448)):
        [_, _, h, w] = self.canvas_size_google(sketch)
        if h >= w:
            sketch_normalize = sketch / np.array([[h, h, 1]], dtype=np.float64)
        else:
            sketch_normalize = sketch / np.array([[w, w, 1]], dtype=np.float64)
        sketch_rescale = sketch_normalize * np.array([[size[0], size[1], 1]], dtype=np.float64)
        return sketch_rescale.astype("int16")

    def canvas_size_google(self, sketch):
        vertical_sum = np.cumsum(sketch[1:], axis=0)
        xmin, ymin, _ = np.min(vertical_sum, axis=0)
        xmax, ymax, _ = np.max(vertical_sum, axis=0)
        w = xmax - xmin
        h = ymax - ymin
        start_x = -xmin - sketch[0][0]
        start_y = -ymin - sketch[0][1]
        return [int(start_x), int(start_y), int(h), int(w)]

    def draw_three(self, sketch, random_color=False):
        coordinates_list = []

        sketch = self.scale_sketch(sketch)  # scale the sketch
        [start_x, start_y, _, _] = self.canvas_size_google(sketch=sketch)
        thickness = int(max(sketch[:, 2]) * 0.025)

        if random_color:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            color = (0, 0, 0)

        pen_now = np.array([start_x, start_y])
        first_zero = False

        for stroke in sketch:
            delta_x_y = stroke[0:0 + 2]
            state = stroke[2:]

            if first_zero:
                pen_now += delta_x_y
                first_zero = False
                continue

            coordinates_list.append(list(map(int, pen_now)))  # Convert NumPy int64 to Python int
            pen_now += delta_x_y

            if int(state) == 1:  # next stroke
                first_zero = True
                if random_color:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                else:
                    color = (0, 0, 0)

        return coordinates_list
    
    def draw_json(self, sketch, random_color=False, img_size=512):
        thickness = int(img_size * 0.025)

        sketch = self.scale_sketch(sketch, (img_size, img_size))  # scale the sketch.
        [start_x, start_y, _, _] = self.canvas_size_google(sketch=sketch)
        start_x += thickness + 1
        start_y += thickness + 1
        pen_now = [start_x, start_y]
        first_zero = False

        json_data = {"lines": [], "mouseUpPoints": []}

        for stroke in sketch:
            delta_x_y = stroke[0:2]
            state = int(stroke[2])
            if first_zero:
                pen_now[0] += delta_x_y[0]
                pen_now[1] += delta_x_y[1]
                first_zero = False
                continue

            line_data = {
                "x1": int(pen_now[0]),
                "y1": int(pen_now[1]),
                "x2": int(pen_now[0] + delta_x_y[0]),
                "y2": int(pen_now[1] + delta_x_y[1]),
                "color": "random" if random_color else "black",
                "thickness": int(thickness)
            }
            json_data["lines"].append(line_data)

            if state == 1:  # mouse up event
                json_data["mouseUpPoints"].append({
                    "x": int(pen_now[0] + delta_x_y[0]),
                    "y": int(pen_now[1] + delta_x_y[1])
                })
                first_zero = True

            pen_now[0] += delta_x_y[0]
            pen_now[1] += delta_x_y[1]

        return json.dumps(json_data)


class SketchData(object):
    def __init__(self, dataPath, model="train"):
        self.dataPath = dataPath
        self.model = model

    def load(self):
        dataset_origin_list = []
        category_list = self.getCategory()
        for each_name in category_list:
            npz_tmp = np.load(f"./{self.dataPath}/{each_name}", encoding="latin1", allow_pickle=True)[self.model]
            print(f"dataset: {each_name} added.")
            dataset_origin_list.append(npz_tmp)
        return dataset_origin_list

    def getCategory(self):
        category_list = os.listdir(self.dataPath)
        return category_list



class FastAPIDrawSketch(DrawSketch):
    def __init__(self):
        super().__init__()

    def draw_three(self, sketch, random_color=False):
        # Modify the draw_three method to directly return the coordinates list
        # without saving to JSON files.
        coordinates_list = super().draw_three(sketch, random_color)
        return coordinates_list


sketchdata = SketchData(dataPath='./datasets')
category_list = sketchdata.getCategory()
dataset_origin_list = sketchdata.load()


@app.post("/generate")
async def generate_sketch_response(category: str = Query(..., title="Category Name")):
    try:
        response_data = {}
        if category in category_list:
            save_name = category.replace(".npz", "")
            response_data[save_name] = []

            fast_api_drawsketch = FastAPIDrawSketch()  # Create an instance of FastAPIDrawSketch

            for image_index in range(100):
                sample_sketch = dataset_origin_list[category_list.index(category)][image_index]
                # Get the JSON data using draw_json method of FastAPIDrawSketch
                json_data = json.loads(fast_api_drawsketch.draw_json(sample_sketch, random_color=True))

                # Append the JSON data to the list for each image
                response_data[save_name].append({
                    "lines": json_data["lines"],
                    "mouseUpPoints": [{"x": point["x"], "y": point["y"]} for point in json_data["mouseUpPoints"]]
                })

            return response_data

        else:
            raise HTTPException(status_code=404, detail=f"Category '{category}' not found.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# new_extractor = BeitImageProcessor.from_pretrained("kmewhort/beit-sketch-classifier")
# new_model = BeitForImageClassification.from_pretrained("kmewhort/beit-sketch-classifier")

# @app.post("/recognize")
# async def recognize_sketch(image_data_url: str = Query(..., description="Data URL of the image from the canvas")):
    
#     try:
#         print("Received image_data_url:", image_data_url)
#         # Convert the data URL to an image
#         image_data = re.sub('^data:image/.+;base64,', '', image_data_url)
#         image = Image.open(io.BytesIO(base64.b64decode(image_data)))

#         # Perform image classification
#         inputs = new_extractor(images=image, return_tensors="pt")
#         outputs = new_model(**inputs)
#         logits = outputs.logits
#         # model predicts one of the 21,841 ImageNet-22k classes
#         predicted_class_idx = logits.argmax(-1).item()
#         predicted_class = new_model.config.id2label[predicted_class_idx]

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
#         inputs = new_extractor(images=image, return_tensors="pt")
#         outputs = new_model(**inputs)
#         logits = outputs.logits
#         # model predicts one of the 21,841 ImageNet-22k classes
#         predicted_class_idx = logits.argmax(-1).item()
#         predicted_class = new_model.config.id2label[predicted_class_idx]

#         return {"predicted_class": predicted_class}
#     except Exception as e:
#         return {"error": "An error occurred during classification."}
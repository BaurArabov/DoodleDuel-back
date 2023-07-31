import json
import os
import random
import time

import cv2
import imageio
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image


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

    def draw_three(self, sketch, random_color=False, show=False, img_size=512):
        thickness = int(img_size * 0.025)

        sketch = self.scale_sketch(sketch, (img_size, img_size))  # scale the sketch.
        [start_x, start_y, h, w] = self.canvas_size_google(sketch=sketch)
        start_x += thickness + 1
        start_y += thickness + 1
        canvas = np.ones((max(h, w) + 3 * (thickness + 1), max(h, w) + 3 * (thickness + 1), 3), dtype='uint8') * 255
        if random_color:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            color = (0, 0, 0)
        pen_now = np.array([start_x, start_y])
        first_zero = False

        # Store each frame of the animation in this list
        animation_frames = []

        for stroke in sketch:
            delta_x_y = stroke[0:0 + 2]
            state = stroke[2:]
            if first_zero:
                pen_now += delta_x_y
                first_zero = False
                continue
            cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=thickness)
            if int(state) == 1:  # next stroke
                first_zero = True
                if random_color:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                else:
                    color = (0, 0, 0)
            pen_now += delta_x_y
            if show:
                cv2.imshow('Drawing Animation', canvas)
                cv2.waitKey(10)

            # Append a copy of the canvas to the animation frames list
            animation_frames.append(np.copy(canvas))

        if show:
            cv2.destroyAllWindows()

        # Resize the last canvas and add it as the final frame for smooth looped animation
        last_frame = cv2.resize(canvas, (img_size, img_size))
        animation_frames.append(last_frame)

        return animation_frames
    
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



if __name__ == '__main__':
    sketchdata = SketchData(dataPath='./datasets')
    category_list = sketchdata.getCategory()
    dataset_origin_list = sketchdata.load()

    for category_index in range(len(category_list)):
        sample_category_name = category_list[category_index]
        print(sample_category_name)
        save_name = sample_category_name.replace(".npz", "")
        folder = os.path.exists(f"./save_sketch/{save_name}/")
        if not folder:
            os.makedirs(f"./save_sketch/{save_name}/")
            print(f"./save_sketch/{save_name}/ is a new directory!")
        drawsketch = DrawSketch()

        for image_index in range(2):
            sample_sketch = dataset_origin_list[category_list.index(sample_category_name)][image_index]
            animation_frames = drawsketch.draw_three(sample_sketch, True, True)  # Set show=False to avoid displaying images

            # Save the GIF animation
            gif_file = f"./save_sketch/{save_name}/{image_index}.gif"
            with imageio.get_writer(gif_file, mode='I', duration=0.1) as writer:
                for frame in animation_frames:
                    writer.append_data(frame)
            print(f"{gif_file} is saved!")

            # Save the JSON data
            json_data = drawsketch.draw_json(sample_sketch, True)
            json_file = f"./save_sketch/{save_name}/{image_index}.json"
            with open(json_file, 'w') as f:
                f.write(json_data)
            print(f"{json_file} is saved!")
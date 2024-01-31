import os
import json

import math

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def create_yolo_bboxes(json_min,
                       output_dir,
                       class_mapping):
    '''
    Opens json file with annotation, takes labels, 
    format it to obb format and writes it to txt file
    '''
    # Open json file
    with open(json_min, 'r') as json_file:
        data = json.load(json_file)

    # Iterating through annotations
    for item in data:
        image_path = item["image"]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_filename = f'{output_dir}/{image_name}.txt'

        # Iterating through labeks in one annotation
        with open(txt_filename, 'w') as txt_file:
            for label in item['label']:
                class_label = label['rectanglelabels'][0]
                class_index = class_mapping.get(class_label, -1)
                if class_index == -1:
                    print(
                        f"There is no class label '{class_label}', change class mapping.")
                    continue

                # Get points in obb format
                points = get_rotated_rectangle(label['x'],
                                               label['y'],
                                               label['width'],
                                               label['height'],
                                               label['rotation'],
                                               label['original_width'],
                                               label['original_height'])

                # Write labels in obb format to txt file
                txt_file.write(f"{class_index} " + " ".join(
                    f"{coord[0]:.6f} {coord[1]:.6f}" for coord in points) + "\n")


def get_rotated_rectangle(x, y, w, h, theta, original_width, original_height):
    x1 = x / 100
    y1 = y / 100

    w = w * original_width
    h = h * original_height

    x2 = (x * original_width + w * math.cos(math.radians(theta))) / \
        original_width / 100
    y2 = (y * original_height + w * math.sin(math.radians(theta))) / \
        original_height / 100
    x3 = (x * original_width + w * math.cos(math.radians(theta)) - h * math.sin(
        math.radians(theta))) / original_width / 100
    y3 = (y * original_height + w * math.sin(math.radians(theta)) + h * math.cos(
        math.radians(theta))) / original_height / 100

    x4 = (x * original_width - h * math.sin(math.radians(theta))) / \
        original_width / 100
    y4 = (y * original_height + h * math.cos(math.radians(theta))) / \
        original_height / 100

    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]


def read_yolo_obb(obb_label_path):
    with open(obb_label_path, 'r') as f:
        lines = f.readlines()
        labels = [list(map(float, line.strip().split())) for line in lines]
    return labels


def visualize_labels(raw_data_dir):
    # Define filling colors
    colors = ["blue", "green", "yellow", "red", "black"]

    # Define images, and labels dirs
    images_dir = os.path.join(raw_data_dir, 'images')
    labels_dir = os.path.join(raw_data_dir, "labels")

    # Iterating through images and labels
    for image_name, label_name in zip(os.listdir(images_dir), os.listdir(labels_dir)):
        # Read image
        full_image_path = os.path.join(images_dir, image_name)
        image = Image.open(full_image_path)
        image_width, image_height = image.size[0], image.size[1]
        draw = ImageDraw.Draw(image)

        # Read label
        full_label_path = os.path.join(labels_dir, label_name)
        labels = read_yolo_obb(full_label_path)

        # Iterating through each label, reformat label and draw lines
        for label in labels:
            class_index, x1, y1, x2, y2, x3, y3, x4, y4 = label
            x1, x2, x3, x4 = [x * image_width for x in (x1, x2, x3, x4)]
            y1, y2, y3, y4 = [y * image_height for y in (y1, y2, y3, y4)]
            points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)]
            draw.line(points, fill=colors[int(class_index)], width=10)

        plt.imshow(image)
        plt.show()


# visualize_labels(raw_data_dir="key-detection/raw-data")


json_file_path = 'key-detection/raw-data/labels.json'
output_dir = 'key-detection/raw-data/labels'
class_mapping = {"number": 0,
                 "date": 1,
                 "sender": 2,
                 "table": 3,
                 "appendix": 4}

create_yolo_bboxes(json_min=json_file_path,
                   output_dir=output_dir,
                   class_mapping=class_mapping)

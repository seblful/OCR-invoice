import math
import os
import json


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
        image_name = os.path.basename(image_path)
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


json_file_path = 'key-detection/raw-data/project-3-at-2024-01-30-15-48-07ee277a.json'
output_dir = 'key-detection/raw-data/labels'
class_mapping = {"number": 0,
                 "date": 1,
                 "sender": 2,
                 "table": 3,
                 "appendix": 4}

create_yolo_bboxes(json_min=json_file_path,
                   output_dir=output_dir,
                   class_mapping=class_mapping)

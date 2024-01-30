import os
import json


def create_yolo_bboxes(json_min):
    with open(json_min, 'r') as json_file:
        data = json.load(json_file)
    for image_annotation in data:
        image_name = os.path.basename(image_annotation['image'])
        for label in image_annotation:
            x = ''
            y = ''
            width = ''
            height = ''
            rotation = ''
        print(image_annotation)
        break
    pass
# https://docs.ultralytics.com/ru/tasks/obb/
# https://docs.ultralytics.com/ru/datasets/obb/


create_yolo_bboxes(
    "key-detection/raw-data/project-3-at-2024-01-30-15-48-07ee277a.json")

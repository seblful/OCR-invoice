import os
import json
from PIL import Image
import cv2
import math
import numpy as np
import torch


def extract_obb_bboxes(json_min_path):
    '''
    Opens json file with annotation, takes labels, 
    format it to obb format and writes it to txt file
    '''
    # Create empty dict to store labels
    labels_dict = dict()

    # Open json file
    with open(json_min_path, 'r') as json_file:
        data = json.load(json_file)

    # Iterating through annotations
    for item in data:
        image_path = item["image"]
        image_name = os.path.basename(image_path)

        # Iterating through labels in one annotation
        for label in item['label']:
            class_label = label['rectanglelabels'][0]
            if class_label == "table":
                # Get points in obb format
                points = get_rotated_rectangle(label['x'],
                                               label['y'],
                                               label['width'],
                                               label['height'],
                                               label['rotation'],
                                               label['original_width'],
                                               label['original_height'])

                labels_dict[image_name] = points
                break

    return labels_dict


def get_rotated_rectangle(x,
                          y,
                          w,
                          h,
                          theta,
                          original_width,
                          original_height):
    x1 = x * original_width / 100
    y1 = y * original_height / 100

    w = w * original_width
    h = h * original_height

    x2 = (x * original_width + w * math.cos(math.radians(theta))) / 100
    y2 = (y * original_height + w * math.sin(math.radians(theta))) / 100
    x3 = (x * original_width + w * math.cos(math.radians(theta)) - h * math.sin(
        math.radians(theta))) / 100
    y3 = (y * original_height + w * math.sin(math.radians(theta)) + h * math.cos(
        math.radians(theta))) / 100

    x4 = (x * original_width - h * math.sin(math.radians(theta))) / 100
    y4 = (y * original_height + h * math.cos(math.radians(theta))) / 100

    return np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.float32)


def crop_obb(image, points):
    # Find the minimum area rotated rectangle
    rect = cv2.minAreaRect(points)

    # Get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    # Represent the corners of the rectangle.
    box = cv2.boxPoints(rect)

    # Coordinate of the points in box points after the rectangle has been straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # The perspective transformation matrix
    M = cv2.getPerspectiveTransform(box, dst_pts)

    # Directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(image, M, (width, height))

    # cv2.imwrite("crop_img.jpg", warped)

    return warped


def extract_tables(images_dir,
                   labels_json,
                   output_dir):
    labels_dict = extract_obb_bboxes(labels_json)
    # print(labels_dict["2a39feb6-Document_55.jpg"])

    for image_name in os.listdir(images_dir):
        full_image_path = os.path.join(images_dir, image_name)
        image_array = np.array(Image.open(full_image_path))
        cropped_table = crop_obb(image=image_array,
                                 points=labels_dict[image_name])
        output_image_path = os.path.join(
            output_dir, os.path.splitext(image_name)[0] + "_cropped.jpg")

        cv2.imwrite(output_image_path, cropped_table)


HOME = os.getcwd()
key_raw_data = os.path.join(HOME, "key-raw-data")
images_dir = os.path.join(key_raw_data, "images")
labels_json_path = os.path.join(key_raw_data, "labels.json")
output_dir = os.path.join(HOME, "import-data")


extract_tables(images_dir,
               labels_json_path,
               output_dir)

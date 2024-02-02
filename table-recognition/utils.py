import os
import json
import cv2
import math
import numpy as np
import torch


def extract_xyxyxyxy_bboxes(json_min_path):
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


def xywhr2xyxyxyxy(rboxes):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in degrees from 0 to 90.

    Args:
        rboxes (numpy.ndarray | torch.Tensor): Input data in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    is_numpy = isinstance(rboxes, np.ndarray)
    cos, sin = (np.cos, np.sin) if is_numpy else (torch.cos, torch.sin)

    ctr = rboxes[..., :2]
    w, h, angle = (rboxes[..., i: i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(
        vec1, axis=-1) if is_numpy else torch.cat(vec1, dim=-1)
    vec2 = np.concatenate(
        vec2, axis=-1) if is_numpy else torch.cat(vec2, dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2) if is_numpy else torch.stack([pt1, pt2, pt3, pt4], dim=-2)


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
                   labels_json):
    labels_dict = extract_xyxyxyxy_bboxes(labels_json)

    for image_name in os.listdir(images_dir):
        full_image_path = os.path.join(images_dir, image_name)


HOME = os.getcwd()
key_raw_data = os.path.join(HOME, "key-raw-data")
images_dir = os.path.join(key_raw_data, "images")
labels_json_path = os.path.join(key_raw_data, "labels.json")


extract_tables(images_dir,
               labels_json_path)

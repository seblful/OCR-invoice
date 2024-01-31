import os

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def read_yolo_obb(obb_label_path):
    with open(obb_label_path, 'r') as f:
        lines = f.readlines()
        labels = [list(map(float, line.strip().split())) for line in lines]
    return labels


def visualize_labels(images_dir,
                     labels_dir):
    # Define filling colors
    colors = ["blue", "green", "yellow", "red", "black"]

    # Define images, and labels dirs
    images_listdir = [file for file in os.listdir(
        images_dir) if file.endswith('jpg')]
    labels_listdir = [file for file in os.listdir(
        labels_dir) if file.endswith('txt')]

    # Iterating through images and labels
    for image_name, label_name in zip(images_listdir, labels_listdir):
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

        fig = plt.figure()
        fig.canvas.manager.set_window_title(image_name)
        plt.imshow(image)
        plt.show()


HOME = os.getcwd()
DATASET = os.path.join(HOME, "dataset")
data = os.path.join(DATASET, 'train')

visualize_labels(images_dir=data,
                 labels_dir=data)

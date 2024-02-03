import csv
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from ultralytics import YOLO
from transformers import TableTransformerForObjectDetection

import easyocr
from tqdm.auto import tqdm


class TableKeyDetector():
    def __init__(self,
                 model_path='best.pt'):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.__model = None

        # Dict to translate indexes to classes
        self.ind2classes = {0: 'number', 1: 'date',
                            2: 'sender', 3: 'table', 4: 'appendix'}

    @property
    def model(self):
        if self.__model is None:
            model = YOLO('yolov8n-obb.pt')  # load an official model
            model = YOLO(self.model_path)
            model.to(self.device)
            self.__model = model

        return self.__model

    def predict(self, image):
        results = self.model(image)

        return results

    def visualize_detection(self, results):
        # plot a BGR numpy array of predictions
        image_array = results[0].plot(line_width=5)
        image = Image.fromarray(image_array[..., ::-1])  # RGB PIL image
        image.show()  # show image

    def crop_obb(self, image, points):
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

    def crop_keys(self, image, results):
        """
        Process the bounding boxes produced by the table detection model into
        cropped table images and cropped tokens.
        """
        # Create dict to store cropped parts of image
        crops = {v: None for k, v in self.ind2classes.items()}

        # Convert image to array
        image_array = np.array(image)

        # Extract predicted classes, confidednces, obbs
        cls = results[0].obb.cls.detach().cpu().numpy()
        # conf = results[0].obb.conf.detach().cpu().numpy()
        xyxyxyxy = results[0].obb.xyxyxyxy.detach().cpu().numpy()

        # Crop every part of image and add it to dict
        for i in range(len(cls)):
            cropped_image = self.crop_obb(image=image_array,
                                          points=xyxyxyxy[i])

            crops[self.ind2classes[cls[i]]] = cropped_image

        return crops


class TableStructureRec():
    def __init__(self,
                 max_size=1000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model = None

        self.max_size = max_size

        self.id2label = self.model.config.id2label
        self.id2label[len(self.id2label)] = "no object"

    @property
    def model(self):
        if self.__model is None:
            model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-structure-recognition-v1.1-all")
            model.to(self.device)
            self.__model = model
        return self.__model

    def resize_image(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale*width)), int(round(scale*height))))

        return resized_image

    def transform_image(self, image):
        structure_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        resized_image = self.resize_image(image)
        transf_image = structure_transform(resized_image).unsqueeze(0)
        transf_image = transf_image.to(self.device)

        return transf_image

    def predict(self, image):
        # Transform image
        transf_image = self.transform_image(image)
        # Make predictions
        with torch.no_grad():
            outputs = self.model(transf_image)

        return outputs

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def outputs_to_objects(self, outputs, image_size):
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist()
                       for elem in self.rescale_bboxes(pred_bboxes, image_size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = self.id2label[int(label)]
            if not class_label == 'no object':
                objects.append({'label': class_label, 'score': float(score),
                                'bbox': [float(elem) for elem in bbox]})

        return objects

    def visualize_cells(self, image, cells):
        visualized_image = image.copy()
        draw = ImageDraw.Draw(visualized_image)

        for cell in cells:
            draw.rectangle(cell["bbox"], outline="red")

        visualized_image.show()

    def visualize_class(self,
                        image,
                        cells,
                        class_to_visualize="table row"):
        if class_to_visualize not in self.id2label.values():
            raise ValueError("Class should be one of the available classes")

        plt.figure(figsize=(16, 10))
        plt.imshow(image)
        ax = plt.gca()

        for cell in cells:
            score = cell["score"]
            bbox = cell["bbox"]
            label = cell["label"]

            if label == class_to_visualize:
                xmin, ymin, xmax, ymax = tuple(bbox)

                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False, color="red", linewidth=3))
                text = f'{cell["label"]}: {score:0.2f}'
                ax.text(xmin, ymin, text, fontsize=15,
                        bbox=dict(facecolor='yellow', alpha=0.5))
                plt.axis('off')

        plt.show()


class TableOCRProcessor():
    def __init__(self):
        # this needs to run only once to load the model into memory
        self.reader = easyocr.Reader(['ru'])

    # Function to find cell coordinates
    def find_cell_coordinates(self, row, column):
        cell_bbox = [column['bbox'][0], row['bbox']
                     [1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    def get_cell_coordinates_by_row(self, table_data):
        # Extract rows and columns
        rows = [entry for entry in table_data if entry['label'] == 'table row']
        columns = [entry for entry in table_data if entry['label']
                   == 'table column']

        # Sort rows and columns by their Y and X coordinates, respectively
        rows.sort(key=lambda x: x['bbox'][1])
        columns.sort(key=lambda x: x['bbox'][0])

        # Generate cell coordinates and count cells in each row
        cell_coordinates = []

        for row in rows:
            row_cells = []
            for column in columns:
                cell_bbox = self.find_cell_coordinates(row, column)
                row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

            # Sort cells in the row by X coordinate
            row_cells.sort(key=lambda x: x['column'][0])

            # Append row information to cell_coordinates
            cell_coordinates.append(
                {'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

        # Sort rows from top to bottom
        cell_coordinates.sort(key=lambda x: x['row'][1])

        return cell_coordinates

    def apply_ocr(self, image, cell_coordinates):
        # OCR row by row
        data = dict()
        max_num_columns = 0
        for idx, row in enumerate(tqdm(cell_coordinates)):
            row_text = []
            for cell in row["cells"]:
                # crop cell out of image
                cell_image = np.array(image.crop(cell["cell"]))
                # apply OCR
                result = self.reader.readtext(np.array(cell_image))
                if len(result) > 0:
                    text = " ".join([x[1] for x in result])
                    row_text.append(text)

            if len(row_text) > max_num_columns:
                max_num_columns = len(row_text)

            data[idx] = row_text

        print("Max number of columns:", max_num_columns)

        # Pad rows which don't have max_num_columns elements to make sure all rows have the same number of columns
        for row, row_data in data.copy().items():
            if len(row_data) != max_num_columns:
                row_data = row_data + \
                    ["" for _ in range(max_num_columns - len(row_data))]
            data[row] = row_data

        return data

    def process(self, image, cells):
        # Get cell coordinates
        cell_coordinates = self.get_cell_coordinates_by_row(cells)
        # Apply OCR
        data = self.apply_ocr(image, cell_coordinates)

        return data

    def save_csv(self, data, csv_save_path):
        with open(csv_save_path, 'w') as result_file:
            wr = csv.writer(result_file, dialect='excel')

            for row, row_text in data.items():
                wr.writerow(row_text)

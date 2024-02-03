from processors import TableKeyDetector, TableStructureRec, TableOCRProcessor

import os
from PIL import Image

HOME = os.getcwd()
IMAGES_DIR = os.path.join(HOME, "images")
IMAGE_PATH = os.path.join(IMAGES_DIR, "2a39feb6-Document_55.jpg")
CSV_SAVE_PATH = os.path.join(HOME, 'output.csv')


def main():
    # Load image
    image = Image.open(IMAGE_PATH).convert("RGB")

    # Make detection and crop keys
    key_detector = TableKeyDetector()
    results = key_detector.predict(image)
    crops = key_detector.crop_keys(image=image,
                                   results=results)
    cropped_table = Image.fromarray(crops['table'])

    # # Visualize detection
    # key_detector.visualize_detection(results)

    # Make table recognition
    table_str_rec = TableStructureRec()
    outputs = table_str_rec.predict(cropped_table)
    cells = table_str_rec.outputs_to_objects(outputs, cropped_table.size)

    # # Visualize table recognition
    # # 0: 'table', 1: 'table column', 2: 'table row', 3: 'table column header', 4: 'table projected row header', 5: 'table spanning cell', 6: 'no object'
    # table_str_rec.visualize_cells(cropped_table, cells)
    # table_str_rec.visualize_class(
    #     cropped_table, cells, class_to_visualize="table row")
    # table_str_rec.visualize_class(
    #     cropped_table, cells, class_to_visualize="table column")

    # Make OCR
    table_ocr_proc = TableOCRProcessor()
    data = table_ocr_proc.process(image=cropped_table,
                                  cells=cells)
    print(data)
    # Save table
    table_ocr_proc.save_csv(data, CSV_SAVE_PATH)


if __name__ == "__main__":
    main()

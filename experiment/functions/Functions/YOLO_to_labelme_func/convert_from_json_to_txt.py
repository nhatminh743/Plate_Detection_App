import os
import json
from PIL import Image

# Define directories
image_dir = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLO_to_labelme_func/train/process'
input_dir = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLO_to_labelme_func/train/json_output'
output_dir = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLO_to_labelme_func/train/labels'
class_txt = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/data/train/labels/classes.txt'

# Load class labels from class_txt
with open(class_txt, 'r') as f:
    class_labels = f.read().splitlines()

# Create a mapping of class label names to indices
class_map = {label: idx for idx, label in enumerate(class_labels)}

def convert_labelme_to_yolo(json_file, image_width, image_height, output_dir):
    # Load the LabelMe JSON data
    with open(json_file) as f:
        data = json.load(f)

    # Get image file name (without extension) to name the YOLO txt file
    image_filename = os.path.splitext(os.path.basename(json_file))[0]

    yolo_filename = os.path.join(output_dir, f"{image_filename}.txt")

    with open(yolo_filename, 'w') as yolo_file:
        # Loop through the shapes (annotations) in the LabelMe file
        for shape in data['shapes']:
            label = shape['label']  # Object class label
            if label not in class_map:
                print(f"Warning: Label '{label}' not found in class map!")
                continue

            class_id = class_map[label]  # Convert label to class index
            points = shape['points']  # Polygon points (should be a rectangle)

            # Get the bounding box (min x, min y, max x, max y)
            x_min = min(points, key=lambda x: x[0])[0]
            x_max = max(points, key=lambda x: x[0])[0]
            y_min = min(points, key=lambda x: x[1])[1]
            y_max = max(points, key=lambda x: x[1])[1]

            # Calculate YOLO format coordinates (normalized)
            x_center = (x_min + x_max) / 2.0 / image_width
            y_center = (y_min + y_max) / 2.0 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            # Write to YOLO txt file
            yolo_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"YOLO annotations saved to: {yolo_filename}")


# Function to get image dimensions (width and height)
def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size  # (width, height)


# Iterate over all files in the input directory
for filename in os.listdir(input_dir):
    if not filename.endswith('.json'):
        print(f'Skipping {filename}')
        continue

    json_file = os.path.join(input_dir, filename)

    # Get corresponding image file to extract width and height

    base_filename = os.path.splitext(filename)[0]

    baseImg_filename = base_filename + '.jpg'

    image_filename = os.path.join(image_dir, baseImg_filename)


    if not os.path.exists(image_filename):
        print(f"Image file not found for {image_filename}")
        continue

    # Get image dimensions (width, height)
    width, height = get_image_dimensions(image_filename)

    # Call the conversion function
    convert_labelme_to_yolo(json_file, width, height, output_dir)

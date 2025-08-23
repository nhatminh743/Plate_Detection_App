import os
import json
import base64
from PIL import Image

#Change dir from linux to window
image_dir = r'/Model_training/YOLO_to_labelme_func/train/images'
label_dir = r'/Model_training/YOLO_to_labelme_func/train/temp_txt'
output_dir = r'/Model_training/YOLO_to_labelme_func/train/json_output'
class_txt = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/data/train/labels/classes.txt'
label_names = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]

for filename in os.listdir(label_dir):
    if not filename.endswith(".txt"):
        continue

    base = os.path.splitext(filename)[0]
    image_path = os.path.join(image_dir, base + ".jpg")
    label_path = os.path.join(label_dir, filename)

    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')

    with Image.open(image_path) as img:
        image_width, image_height = img.size

    shapes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id, x, y, w, h = map(float, parts)
            x1 = (x - w / 2) * image_width
            y1 = (y - h / 2) * image_height
            x2 = (x + w / 2) * image_width
            y2 = (y + h / 2) * image_height
            shape = {
                "label": label_names[int(class_id)],
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            shapes.append(shape)

    data = {
        "version": "4.5.9",
        "flags": {},
        "shapes": shapes,
        "imagePath": base + ".jpg",
        "imageData": encoded,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, base + ".json"), "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Converted {filename} to LabelMe format")

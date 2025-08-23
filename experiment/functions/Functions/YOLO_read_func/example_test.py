from ultralytics import YOLO

best_model_dir = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/runs/content/runs/detect/train2/weights/best.pt'
best_model = YOLO(best_model_dir)

res = best_model.predict('/home/minhpn/Desktop/Green_Parking/Dummy_Data_For_Small_Test/Extracted_Plate_Data/0229_05817_b_plate.jpg')[0]

names = best_model.model.names

# Step 1: Extract relevant info: (x_center, y_center, class_name)
detections = []

for box in res.boxes:
    cls_id = int(box.cls[0])
    class_name = names[cls_id]
    x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
    y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
    detections.append((x_center.item(), y_center.item(), class_name))

# Step 2: Determine a threshold to separate the two lines
# We'll use the median y_center to split
y_values = [d[1] for d in detections]
y_median = sorted(y_values)[len(y_values) // 2]

line1 = []  # top line
line2 = []  # bottom line

for det in detections:
    if det[1] < y_median:
        line1.append(det)
    else:
        line2.append(det)

# Step 3: Sort each line by x (left to right)
line1 = sorted(line1, key=lambda x: x[0])
line2 = sorted(line2, key=lambda x: x[0])

# Step 4: Combine characters from top line then bottom line
final_characters = [char for _, _, char in line1 + line2]
predicted_text = ''.join(final_characters)

cut = predicted_text[:2] + '-' + predicted_text[2:4] + ' ' + predicted_text[4:]

print("Predicted plate number:", cut)
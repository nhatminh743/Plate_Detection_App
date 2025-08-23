from imutils import contours
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import os
import cv2
import imutils
import math

def convert_to_pixel_coords(box, img_width, img_height):
    """
    Convert YOLO format box (cx, cy, w, h) to pixel coordinates.
    """
    return [
        box[0] * img_width,  # cx
        box[1] * img_height, # cy
        box[2] * img_width,  # w
        box[3] * img_height  # h
    ]

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    Boxes are in the format (cx, cy, w, h).
    """
    # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
    x1_box1, y1_box1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x2_box1, y2_box1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2

    x1_box2, y1_box2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_box2, y2_box2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    # Calculate intersection
    inter_x1 = max(x1_box1, x1_box2)
    inter_y1 = max(y1_box1, y1_box2)
    inter_x2 = min(x2_box1, x2_box2)
    inter_y2 = min(y2_box1, y2_box2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union
    box1_area = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    box2_area = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def plot_result(rgb, result, label=None, iou_threshold=0.5):
    """
    Plot YOLO prediction results and compare with ground truth labels if provided.

    Parameters:
      - rgb: numpy array of the RGB image.
      - result: YOLO prediction result.
      - label: Optional ground truth labels [(class_id, [cx, cy, w, h]), ...].
      - iou_threshold: IoU threshold to match predictions with ground truth.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)

    class_names = result.names
    height, width, _ = rgb.shape

    predictions = result.boxes.xywh.cpu().numpy()
    pred_classes = result.boxes.cls.cpu().numpy()

    gt_used = [False] * len(label) if label else []

    for i, pred_box in enumerate(predictions):
        pred_class_id = int(pred_classes[i])
        pred_box = pred_box.tolist()
        matched = False

        if label:
            for j, (gt_class_id, gt_box) in enumerate(label):
                if not gt_used[j] and gt_class_id == pred_class_id:
                    # Convert ground truth box to pixel values
                    gt_box_pixel = convert_to_pixel_coords(gt_box, width, height)
                    iou = calculate_iou(pred_box, gt_box_pixel)
                    if iou >= iou_threshold:
                        matched = True
                        gt_used[j] = True
                        break

        color = 'green' if matched else 'red'
        cx, cy, w, h = pred_box
        hw, hh = w / 2, h / 2

        ax.add_patch(Rectangle(
            (cx - hw, cy - hh), w, h,
            edgecolor=color,
            fill=None,
            linewidth=2
        ))

        label_text = f"{class_names[pred_class_id]} ({iou:.2f})" if matched else class_names[pred_class_id]
        ax.text(
            cx - hw, cy - hh - 5,
            label_text,
            color=color,
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor=color, alpha=0.7)
        )

    if label:
        for gt_class_id, gt_box in label:
            # Convert ground truth box to pixel values
            gt_box_pixel = convert_to_pixel_coords(gt_box, width, height)
            cx, cy, w, h = gt_box_pixel
            hw, hh = w / 2, h / 2

            ax.add_patch(Rectangle(
                (cx - hw, cy - hh), w, h,
                edgecolor='blue',
                fill=None,
                linestyle='--',
                linewidth=1
            ))

    plt.show()

def crop_and_save_rois(rgb, result, save_dir, filename, conf_threshold=0.5):
    """
    Crop ROIs from YOLO predictions and save them to a folder.

    Parameters:
      - rgb: numpy array of the RGB image.
      - result: YOLO prediction result.
      - save_dir: directory to save cropped raw_image.
      - conf_threshold: confidence threshold to filter predictions.
    """
    os.makedirs(save_dir, exist_ok=True)
    height, width, _ = rgb.shape

    # boxes = result.boxes.xywh.cpu().numpy()
    # scores = result.boxes.conf.cpu().numpy()
    # classes = result.boxes.cls.cpu().numpy()

    boxes = result.boxes
    not_pass_confidence = 0
    saved = 0

    # for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
    #     if score < conf_threshold:
    #         not_pass_confidence += 1
    #         continue
    #
    #     cx, cy, w, h = box
    #     x1 = int(max(cx - w / 2, 0))
    #     y1 = int(max(cy - h / 2, 0))
    #     x2 = int(min(cx + w / 2, width))
    #     y2 = int(min(cy + h / 2, height))
    #
    #     roi = rgb[y1:y2, x1:x2]
    #
    #     save_path = os.path.join(save_dir, f"{filename[:12]}_plate_{saved}.jpg")
    #     cv2.imwrite(save_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
    #     print(f"Saved ROI to {save_path}")
    #     saved += 1

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        confidence = box.conf[0].item()

        if confidence < conf_threshold:
            not_pass_confidence += 1
            continue

        roi = rgb[y1:y2, x1:x2]

        if filename.endswith('.jpg') or filename.endswith('.png'):
            filename = filename[:-4]
        elif filename.endswith('.jpeg'):
            filename = filename[:-5]

        save_path = os.path.join(save_dir, f"{filename}_plate_{saved}.jpg")

        cv2.imwrite(save_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        print(f"Saved ROI to {save_path}")
        saved += 1

    return not_pass_confidence

# def postprocess(image, imshow_mode = False):
#     original_image = image.copy()
#     original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
#     # Resize
#     image = imutils.resize(image, width=300)
#     if imshow_mode:
#         cv2.imshow("original image", image)
#
#     # Gray image
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     if imshow_mode:
#         cv2.imshow("gray image", gray_image)
#
#     # Smoothing
#     bilateral_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
#     if imshow_mode:
#         cv2.imshow("smoothened image", bilateral_image)
#
#     # # Otsu's thresholding
#     # ret, thresh = cv2.threshold(bilateral_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     # if imshow_mode:
#     #     cv2.imshow("thresh image", thresh)
#     #
#     # # Morph
#     # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     # morph_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     # if imshow_mode:
#     #     cv2.imshow("morphed image", morph_image)
#
#     #Canny
#     morph_image = cv2.Canny(bilateral_image, 30, 200)
#
#     # Find contours
#     cnts, new = cv2.findContours(morph_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#     largest_cnts = cnts[0]
#
#     copy_image = image.copy()
#
#     if imshow_mode:
#         cv2.imshow("largest cnt", cv2.drawContours(copy_image, cnts, -1, (0, 255, 0), 2))
#
#     (x1, y1) = largest_cnts[0, 0]
#     (x2, y2) = largest_cnts[1, 0]
#     (x3, y3) = largest_cnts[2, 0]
#     (x4, y4) = largest_cnts[3, 0]
#     array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
#     sorted_array = array.sort(reverse=True, key=lambda x: x[1])
#     (x1, y1) = array[0]
#     (x2, y2) = array[1]
#     doi = abs(y1 - y2)
#     ke = abs(x1 - x2)
#     angle = math.atan(doi / ke) * (180.0 / math.pi)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


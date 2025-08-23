import os
import numpy as np
from sklearn.decomposition import PCA
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression, scale_boxes
import cv2
from ultralytics.data.augment import LetterBox
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import silhouette_score
from .show_result import PlotImageS
from scipy.spatial import Voronoi, voronoi_plot_2d


class LetterExtractor:
    def __init__(self, data_dir, save_dir, best_model_file, debug_mode=False):
        self.model_dir = best_model_file
        self.model = YOLO(self.model_dir)
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.debug_mode = debug_mode
        os.makedirs(save_dir, exist_ok=True)
        self.names = self.model.model.names
        self.scale_factor = 6
        self.two_row=True

    def process_images(self):
        output_path = os.path.join(self.save_dir, 'ocr_results.txt')

        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith('.jpg'):
                final = self._process_single_image(filename)
                with open(output_path, 'a') as f:
                    cut_index = filename.index("_plate_")
                    clean_name = filename[:cut_index]
                    f.write(f'{clean_name}: {final}\n')
                    print(f"Finished processing {clean_name}")
        print("Successfully processed images")

    def _process_single_image(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        detections = []
        data_for_kMeans = []
        points = []

        # Load image
        original_image = cv2.imread(filepath)
        assert original_image is not None, f"Failed to load {filepath}"
        h, w, channel = original_image.shape

        # Resize image
        new_size = (w * self.scale_factor, h * self.scale_factor)
        resized_image = cv2.resize(original_image, new_size, interpolation=cv2.INTER_LINEAR)

        results = self.model.predict(resized_image, imgsz=new_size, conf=0.25, iou=0.7, agnostic_nms = True)[0]

        # If no boxes found
        if results.boxes is None or len(results.boxes) == 0:
            print(f"No characters detected in {filename}")
            output_path = os.path.join(self.save_dir, 'ocr_results.txt')
            with open(output_path, 'a') as f:
                f.write(f"{filename[:12]}.jpg: None\n")
            return

        # Process each detected box
        for box in results.boxes:
            x, y, w, h = box.xywh[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = self.model.names[cls]
            if self.debug_mode:
               print(f"Box: ({x:.0f}, {y:.0f}, {w:.0f}, {h:.0f}), conf: {conf:.2f}, class: {class_name}")

            detections.append([x, y, class_name])
            points.append([int(x), int(y)])
            data_for_kMeans.append(y)

        points = np.array(points)
        data_for_kMeans = np.array(data_for_kMeans).reshape(-1, 1)
        y_coords = np.array([row[1] for row in points]).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(y_coords)
        labels = kmeans.labels_

        #Evaluate whether there is one-row or two-row
        score = silhouette_score(data_for_kMeans, labels)

        if score < 0.8:
            self.two_row=False
        else:
            self.two_row=True
        if self.debug_mode:
            print(f'Silhouette score: {score}')
        if self.debug_mode:

            plot_image_func = PlotImageS(
                model_dir = self.model_dir,
                image_dir = self.data_dir,
                output_dir = r'/home/minhpn/Desktop/Green_Parking/one_image/visualization',
            )

            plot_image_func.plot_all()

        if self.two_row:

            clusters = defaultdict(list)
            for (x, y, cls_name), label in zip(detections, labels):
                clusters[label].append((x, y, cls_name))

            sorted_labels = sorted(clusters.keys(), key=lambda label: np.mean([y for _, y, _ in clusters[label]]))

            sorted_class_names = []
            for label in sorted_labels:
                sorted_cluster = sorted(clusters[label], key=lambda tup: tup[0])
                class_names = [cls_name for _, _, cls_name in sorted_cluster]
                sorted_class_names.append(class_names)
            if self.debug_mode:
                print(f"(Low average y - sorted by x center): {sorted_class_names[0]}")
                print(f"(High average y - sorted by x center): {sorted_class_names[1]}")

            upper_row = lower_row = ''
            for letter in sorted_class_names[0]:
                upper_row += letter

            upper_row = upper_row[:2] + '-' + upper_row[2:]

            for letter in sorted_class_names[1]:
                lower_row += letter

            final_plate = upper_row + ' ' +  lower_row
            if self.debug_mode:
               print(f"The plate is {final_plate}")

            return final_plate


            # cluster1 = points[labels == 0]
            # cluster2 = points[labels == 1]
            #
            #
            # if cluster1[:, 1].mean() < cluster2[:, 1].mean():
            #    top_cluster, bottom_cluster = cluster1, cluster2
            # else:
            #    top_cluster, bottom_cluster = cluster2, cluster1
            # # Find the line
            # pca = PCA(n_components=1)
            # pca.fit(np.vstack([top_cluster, bottom_cluster]))
            # direction = pca.components_[0]
            # direction = direction / np.linalg.norm(direction)
            #
            # top_mean = top_cluster.mean(axis=0)
            # bottom_mean = bottom_cluster.mean(axis=0)
            # #Find the point it need to passthrough
            # mid_point = (top_mean + bottom_mean) / 2
            #
            # line_len = 600
            # line_vector = direction * line_len / 2
            # pt1 = mid_point - line_vector
            # pt2 = mid_point + line_vector
            #
            # if self.debug_mode:
            #    plt.figure(figsize=(6, 5))
            #    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')
            #    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r--', label='Dividing Line (PCA aligned)')
            #    plt.title("Clustered Rows with Dividing Line")
            #    plt.xlabel("Center X")
            #    plt.ylabel("Center Y")
            #    plt.legend()
            #    plt.gca().invert_yaxis()
            #    plt.grid(True)
            #    plt.show()
            #
            # #Cut the plate


            # Optional: visualization
            # if self.debug_mode:
            #     plt.scatter([x for x, y, _ in detections], [y for x, y, _ in detections], c=labels, cmap='viridis')
            #     plt.gca().invert_yaxis()
            #     plt.title("Clustering Boxes into Rows")
            #     plt.xlabel("Center X")
            #     plt.ylabel("Center Y")
            #     plt.show()

        else:
            clusters = defaultdict(list)
            for (x, y, cls_name), label in zip(detections, labels):
                clusters[label].append((x, y, cls_name))

            sorted_labels = sorted(detections, key = lambda row: row[0])

            sorted_labels = [row[2] for row in sorted_labels]
            final = ''

            for letter in sorted_labels:
                final += letter

            final = final[:2] + '-' + final[2:-5] + ' ' + final[-5:][::-1]
            if self.debug_mode:
                print("Single line detected")
                print(f'Sorted labels: {sorted_labels}')
                print(f'Final plate is {final}')

            # Optional: visualization
            if self.debug_mode:
                plt.scatter([x for x, y, _ in detections], [y for x, y, _ in detections], c=labels, cmap='viridis')
                plt.gca().invert_yaxis()
                plt.title("Clustering Boxes into Rows")
                plt.xlabel("Center X")
                plt.ylabel("Center Y")
                plt.show()

            return final

        # Organize characters into 2 lines based on y-median
        # y_values = [d[1] for d in detections]
        # sum_y = sum(y_values)
        # y_mean = sum_y / len(y_values)
        # line1 = [det for det in detections if det[1] < y_mean]
        # line2 = [det for det in detections if det[1] >= y_mean]
        #
        # # Sort characters left to right
        # line1 = sorted(line1, key=lambda x: x[0])
        # line2 = sorted(line2, key=lambda x: x[0])
        # final_characters = [char for _, _, char in line1 + line2]
        # predicted_text = ''.join(final_characters)
        #
        # # Format plate: XX-YY ZZZZ (or whatever format you want)
        # predicted_text_process = predicted_text[:2] + '-' + predicted_text[2:4] + ' ' + predicted_text[4:]

        # Debug print
        # if self.debug_mode:
        #     print(f"{filename}: Raw Prediction = {predicted_text}")
        #     print(f"{filename}: Formatted Plate = {predicted_text_process}")
        #
        # # Save result
        # output_path = os.path.join(self.save_dir, 'ocr_results.txt')
        # with open(output_path, 'a') as f:
        #     f.write(f'{filename[:12]}.jpg: {predicted_text_process}\n')
        #     print(f"Finished processing {filename[:12]}")

    # def _process_single_image(self, filename):
    #     filepath = os.path.join(self.data_dir, filename)
    #     detections = []
    #
    #     # Load image
    #     img0 = cv2.imread(filepath)
    #     assert img0 is not None, f"Failed to load {filepath}"
    #
    #     # Run model inference using Ultralytics high-level API
    #     results = self.model.predict(img0, imgsz=640, conf=0.5, iou=0.65)[0]
    #
    #     # If no boxes found
    #     if results.boxes is None or len(results.boxes) == 0:
    #         print(f"No characters detected in {filename}")
    #         output_path = os.path.join(self.save_dir, 'ocr_results.txt')
    #         with open(output_path, 'a') as f:
    #             f.write(f"{filename[:12]}.jpg: None\n")
    #         return
    #
    #     # Process each detected box
    #     for box in results.boxes:
    #         x1, y1, x2, y2 = box.xyxy[0].tolist()
    #         conf = float(box.conf[0])
    #         cls = int(box.cls[0])
    #         class_name = self.model.names[cls]
    #         print(f"Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), conf: {conf:.2f}, class: {class_name}")
    #
    #         # Compute center for sorting
    #         x_center = (x1 + x2) / 2
    #         y_center = (y1 + y2) / 2
    #         detections.append((x_center, y_center, class_name))
    #
    #     # Organize characters into 2 lines based on y-median
    #     y_values = [d[1] for d in detections]
    #     y_median = sorted(y_values)[len(y_values) // 2]
    #     line1 = [det for det in detections if det[1] < y_median]
    #     line2 = [det for det in detections if det[1] >= y_median]
    #
    #     # Sort characters left to right
    #     line1 = sorted(line1, key=lambda x: x[0])
    #     line2 = sorted(line2, key=lambda x: x[0])
    #     final_characters = [char for _, _, char in line1 + line2]
    #     predicted_text = ''.join(final_characters)
    #
    #     # Format plate: XX-YY ZZZZ (or whatever format you want)
    #     predicted_text_process = predicted_text[:2] + '-' + predicted_text[2:4] + ' ' + predicted_text[4:]
    #
    #     # Debug print
    #     if self.debug_mode:
    #         print(f"{filename}: Raw Prediction = {predicted_text}")
    #         print(f"{filename}: Formatted Plate = {predicted_text_process}")
    #
    #     # Save result
    #     output_path = os.path.join(self.save_dir, 'ocr_results.txt')
    #     with open(output_path, 'a') as f:
    #         f.write(f'{filename[:12]}.jpg: {predicted_text_process}\n')
    #         print(f"Finished processing {filename[:12]}")

    # def _process_single_image(self, filename):
    #     filepath = os.path.join(self.data_dir, filename)
    #     detections = []
    #
    #     # Load image
    #     img0 = cv2.imread(filepath)
    #     assert img0 is not None, f"Failed to load {filepath}"
    #
    #     # Resize and pad image using letterbox
    #     transformer = LetterBox(new_shape=(640, 640))
    #     img_resized = transformer(image=img0)
    #
    #     # Convert BGR to RGB, transpose to CHW, normalize to [0,1]
    #     img = img_resized[:, :, ::-1].transpose(2, 0, 1)
    #     img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    #     img = torch.from_numpy(img).unsqueeze(0).to(self.model.device)
    #
    #     # Run model
    #     with torch.no_grad():
    #         pred = self.model.model(img)[0]
    #
    #     # Apply NMS
    #     pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.65)[0]
    #
    #     if pred is not None:
    #         for box in pred:
    #             x1, y1, x2, y2, conf, cls = box
    #             print(f"Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), conf: {conf:.2f}, class: {self.model.names[int(cls)]}")
    #     else:
    #         print("No detections after NMS.")
    #
    #     results = self.model.predict(img0, imgsz=640, conf=0.5, iou=0.65)[0]
    #     for box in results.boxes:
    #         x1, y1, x2, y2 = box.xyxy[0].tolist()
    #         conf = float(box.conf[0])
    #         cls = int(box.cls[0])
    #         print(f"Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), conf: {conf:.2f}, class: {self.model.names[cls]}")
    #
    #     if pred is None or len(pred) == 0:
    #         print(f"No characters detected in {filename}")
    #         output_path = os.path.join(self.save_dir, 'ocr_results.txt')
    #         with open(output_path, 'a') as f:
    #             f.write(f"{filename[:12]}.jpg: None\n")
    #         return
    #
    #     # Scale boxes to original image
    #     pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], img0.shape).round()
    #
    #     for *xyxy, conf, cls in pred:
    #         x1, y1, x2, y2 = xyxy
    #         x_center = (x1 + x2) / 2
    #         y_center = (y1 + y2) / 2
    #         class_name = self.names[int(cls)]
    #         detections.append((x_center.item(), y_center.item(), class_name))
    #
    #     y_values = [d[1] for d in detections]
    #     y_median = sorted(y_values)[len(y_values) // 2]
    #     line1 = [det for det in detections if det[1] < y_median]
    #     line2 = [det for det in detections if det[1] >= y_median]
    #
    #     line1 = sorted(line1, key=lambda x: x[0])
    #     line2 = sorted(line2, key=lambda x: x[0])
    #     final_characters = [char for _, _, char in line1 + line2]
    #     predicted_text = ''.join(final_characters)
    #     predicted_text_process = predicted_text[:2] + '-' + predicted_text[2:4] + ' ' + predicted_text[4:]
    #
    #     if self.debug_mode:
    #         print(f"{filename}: Raw Prediction = {predicted_text}")
    #         print(f"{filename}: Formatted Plate = {predicted_text_process}")
    #
    #     output_path = os.path.join(self.save_dir, 'ocr_results.txt')
    #     with open(output_path, 'a') as f:
    #         f.write(f'{filename[:12]}.jpg: {predicted_text_process}\n')
    #         print(f"Finished processing {filename[:12]}")

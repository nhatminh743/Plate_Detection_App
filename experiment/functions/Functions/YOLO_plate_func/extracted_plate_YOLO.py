import os
import cv2
from .plotting import crop_and_save_rois
from ultralytics import YOLO


class PlateExtractor:
    def __init__(self, data_dir, save_dir, best_model_file, debug_mode=False):
        self.model_file = best_model_file
        self.model = YOLO(self.model_file)
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.debug_mode = debug_mode
        os.makedirs(save_dir, exist_ok=True)

        self.fail_confidence = 0
        self.fail_ratio = 0
        self.fail_count = 0
        self.total_images = 0
        self.all_image_dimension = []
        self.org_dim = []
        self.model_dim = []

    def process_images(self):
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith('.jpg') or filename.lower().endswith('.png') or filename.lower().endswith('.jpeg'):
                dimension, org_dim, model_dim = self._process_single_image(filename)
                if dimension:
                    self.all_image_dimension.extend(dimension)
                    self.org_dim.append(org_dim)
                    self.model_dim.append(model_dim)
        self._report()
        return self.all_image_dimension, self.org_dim, self.model_dim

    def _process_single_image(self, filename):
        file_path = os.path.join(self.data_dir, filename)
        img = cv2.imread(file_path)

        if img is None:
            print(f"❌ Failed to read image: {filename}")
            self.fail_count += 1
            return None, None, 1

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.model.predict(rgb_img)[0]

        model_h, model_w = result.orig_shape[:2]
        orig_h, orig_w = img.shape[:2]

        scale_x = orig_w / model_w
        scale_y = orig_h / model_h

        if result.boxes is None or len(result.boxes) < 1:
            print(f"❌ No plate detected: {filename}")
            return None, None, 2

        # Save all bounding boxes after scaling
        dimensions = []
        for box in result.boxes.xyxy.tolist():
            x1, y1, x2, y2 = box
            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y
            dimensions.append([x1, y1, x2, y2])

        # Save cropped region & calculate failure score
        not_pass_confidence = crop_and_save_rois(rgb_img, result, self.save_dir, filename)

        self.total_images += 1

        return dimensions, [orig_w, orig_h], [model_w, model_h]

    def _report(self):
        print("\n" + "=" * 50)
        print("SUMMARY REPORT")
        print("=" * 50)
        print(f"Total Images Processed   : {self.total_images}")
        print(f"Images Failed to Load    : {self.fail_count}")
        print("=" * 50 + "\n")
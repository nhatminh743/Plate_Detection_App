import os
import cv2
import numpy as np
import re
from collections import defaultdict
from paddleocr import TextDetection, TextRecognition
from .infer import inference


class PaddleOCRLineExtractor:
    def __init__(self, data_dir, temporary_dir, save_dir, text_recognition_dir):
        self.data_dir = data_dir
        self.temporary_dir = temporary_dir
        self.save_dir = save_dir
        self.text_detector = TextDetection(model_name="PP-OCRv5_mobile_det")
        self.text_recognition_dir = text_recognition_dir
        self.text_recognition = TextRecognition(model_name="PP-OCRv5_mobile_rec")

        os.makedirs(self.save_dir, exist_ok=True)

    def detect_and_crop_lines(self, filepath, filename):
        # Read the input image
        image = cv2.imread(filepath)
        if image is None:
            print(f"❌ Failed to read image at: {filepath}")
            return

        # Run text detection model
        output = self.text_detector.predict(filepath, batch_size=1)

        # Create a folder based on filename (without extension)
        folder_name = os.path.splitext(filename)[0]
        temporary_folder = os.path.join(self.temporary_dir, folder_name)
        os.makedirs(temporary_folder, exist_ok=True)

        # Iterate through each detection result
        for res_idx, res in enumerate(output):
            # Sort detected polygons by vertical position (y-coordinate)
            sort_based_on_y = sorted(res['dt_polys'], key=lambda coor: np.mean(coor[:, 1]))

            for i, poly in enumerate(sort_based_on_y):
                x_coords = poly[:, 0]
                y_coords = poly[:, 1]

                x1, x2 = int(np.min(x_coords)), int(np.max(x_coords))
                y1, y2 = int(np.min(y_coords)), int(np.max(y_coords))

                # Extract region of interest (ROI)
                roi = image[y1:y2, x1:x2]

                if roi.size == 0:
                    print(f"❌ Empty ROI at index {i}, skipping.")
                    continue

                # Convert ROI to BGR (if needed for saving)
                roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)

                # Save cropped image to temporary folder
                roi_filename = f"{folder_name}_roi_{res_idx}_{i}.jpg"
                temporary_filepath = os.path.join(temporary_folder, roi_filename)
                cv2.imwrite(temporary_filepath, roi_bgr)
                print(f"Saved ROI to {temporary_filepath}")

    def recognize_text_from_lines(self, output_path, folderpath):
        # Group files by res_idx
        grouped_files = defaultdict(list)

        for filename in sorted(os.listdir(folderpath)):
            if not filename.lower().endswith('.jpg'):
                continue

            match = re.search(r'_roi_(\d+)_\d+\.jpg$', filename)
            if match:
                res_idx = int(match.group(1))
                grouped_files[res_idx].append(filename)
            else:
                print(f"⚠️ Filename does not match expected pattern: {filename}")

        for res_idx in sorted(grouped_files.keys()):
            all_texts = []

            for filename in sorted(grouped_files[res_idx]):
                filepath = os.path.join(folderpath, filename)

                # Run text recognition
                # output = inference.main(use_gpu=False, image_dir=filepath, rec_model_dir=self.text_recognition_dir)
                #
                #
                # text = output['rec_text']
                # score = output['rec_score']
                #
                # print(f'Text: {text}, Confidence: {score}')
                # all_texts.append(text)

                output = self.text_recognition.predict(filepath, batch_size=1)

                for res in output:
                    all_texts.append(res['rec_text'])
                    # -----------------------------------------------EXPERIMENTAL-------------------------------------------------------
                    text = res['rec_text']
                    confidence = res['rec_score']
                    print(f'Text: {text}, Confidence: {confidence}')
                    # -----------------------------------------------EXPERIMENTAL-------------------------------------------------------

            final_line = ' '.join(all_texts)

            # Try to get original image name
            example_filename = grouped_files[res_idx][0]
            try:
                cut_index = example_filename.index("_plate_")
                clean_name = example_filename[:cut_index]
                org_img_filename = clean_name + '.jpg'
            except ValueError:
                org_img_filename = example_filename

            print(f"{org_img_filename} [res_idx={res_idx}]: {final_line}")

            # Append result to file
            with open(output_path, 'a') as f:
                f.write(f'{org_img_filename}: {final_line}\n')

    def run(self):
        print("Detecting and cropping lines...")
        for filename in sorted(os.listdir(self.data_dir)):
            filepath = os.path.join(self.data_dir, filename)
            self.detect_and_crop_lines(filepath, filename)

        print("Recognizing text from cropped lines...")
        output_path = os.path.join(self.save_dir, 'ocr_results.txt')
        os.makedirs(self.save_dir, exist_ok=True)

        # Clear previous content before appending new results
        with open(output_path, 'w') as f:
            f.write('')

        for folder_name in sorted(os.listdir(self.temporary_dir)):
            folder_path = os.path.join(self.temporary_dir, folder_name)
            if os.path.isdir(folder_path):
                self.recognize_text_from_lines(output_path, folder_path)


# # Usage
# ocr_runner = PaddleOCRLineExtractor(
#     data_dir="/home/minhpn/Desktop/Green_Parking/one_image/Extracted_Plate_Data",
#     temporary_dir="/home/minhpn/Desktop/Green_Parking/one_image/Extracted_Line",
#     save_dir="/home/minhpn/Desktop/Green_Parking/one_image/Final_Result"
# )
#
# ocr_runner.run()

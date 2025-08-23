import os
import cv2
import numpy as np
from .extract_plate_function import detect_license_plate

class PlateExtractor:
    def __init__(self, data_dir, save_dir, debug_mode=False):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.debug_mode = debug_mode
        os.makedirs(save_dir, exist_ok=True)

        # Counters
        self.fail_count_no_plate = 0
        self.fail_area = 0
        self.fail_ratio = 0
        self.total_count = 0

    def process_images(self):
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith('.jpg'):
                self._process_single_image(filename)
        self._print_summary()

    def _process_single_image(self, filename):
        file_path = os.path.join(self.data_dir, filename)
        img = cv2.imread(file_path)

        _, imgThresh, status = detect_license_plate(img)

        fail_detected = False
        fail_area_check = False
        fail_ratio_check = False

        # Detection check
        if status == 0:
            imgThresh = np.full((3, 3), 255, dtype=np.uint8)
            self.fail_count_no_plate += 1
            fail_detected = True

        img_area = imgThresh.shape[0] * imgThresh.shape[1]
        img_ratio = imgThresh.shape[1] / imgThresh.shape[0]

        if self.debug_mode:
            print('=' * 80)
            print(f"File: {filename}")
            print(f"Plate Area: {img_area}")
            print(f"Plate Ratio: {img_ratio:.2f}")
            print(f"Plate Shape: {imgThresh.shape}")

        # Area check
        if img_area > 60000 or img_area < 7000:
            imgThresh = np.full((3, 3), 255, dtype=np.uint8)
            if not fail_detected:
                self.fail_area += 1
                fail_area_check = True

        # Ratio check
        if img_ratio <= 1 or img_ratio > 3:
            imgThresh = np.full((3, 3), 255, dtype=np.uint8)
            if not fail_detected and not fail_area_check:
                self.fail_ratio += 1
                fail_ratio_check = True

        # Error Report
        if self.debug_mode:
            if not any([fail_detected, fail_area_check, fail_ratio_check]):
                print("Status: Successfully processed")
            else:
                print("Errors:")
                if fail_detected:
                    print("- No plate detected")
                if fail_area_check:
                    print("- Plate area failed")
                if fail_ratio_check:
                    print("- Plate ratio failed")
            print('=' * 80)

        # Save output
        save_path = os.path.join(self.save_dir, f"{filename[:12]}_plate.jpg")
        cv2.imwrite(save_path, imgThresh)
        self.total_count += 1

    def _print_summary(self):
        print("\n=== SUMMARY REPORT ===")
        print(f"Total Images Processed: {self.total_count}")
        print(f"Failed to Detect Plate: {self.fail_count_no_plate}")
        print(f"Failed Area Test (excluding undetected): {self.fail_area}")
        print(f"Failed Ratio Test (excluding undetected & area fails): {self.fail_ratio}")
        success_count = self.total_count - self.fail_count_no_plate - self.fail_area - self.fail_ratio
        print(f"Successfully Passed All Tests: {success_count} / {self.total_count}")


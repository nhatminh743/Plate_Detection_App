import os
import cv2

def threshold_images(source_dir, target_dir, thresh_value=127):
    """
    Threshold all raw_image in source_dir and save to target_dir preserving folder structure.
    """
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                source_path = os.path.join(root, file)
                # Create the same structure in target_dir
                relative_path = os.path.relpath(root, source_dir)
                save_dir = os.path.join(target_dir, relative_path)
                os.makedirs(save_dir, exist_ok=True)
                target_path = os.path.join(save_dir, file)

                # Read and threshold
                image = cv2.imread(source_path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Save
                cv2.imwrite(target_path, thresh)
                print(f"Processed & saved: {target_path}")

# Example usage
source_folder = '/home/minhpn/Desktop/Green_Parking/dataset_vietnam_license_plate_val'
target_folder = '/home/minhpn/Desktop/Green_Parking/dataset_vietnam_license_plate_val_thresh'
threshold_images(source_folder, target_folder)

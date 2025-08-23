import os
import shutil
import random

def split_dataset(dataset_dir, val_dir, val_ratio=0.2, copy=False):
    """
    Splits dataset by moving/copying val_ratio raw_image into val_dir.

    Args:
        dataset_dir (str): Path to the dataset directory.
        val_dir (str): Path to the validation directory.
        val_ratio (float): Proportion of raw_image to use for validation.
        copy (bool): If True, copy files. If False, move files.
    """
    os.makedirs(val_dir, exist_ok=True)

    for class_folder in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_folder)
        if not os.path.isdir(class_path):
            continue  # Skip files, only process folders

        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        num_val = int(len(images) * val_ratio)
        val_images = random.sample(images, num_val)

        val_class_folder = os.path.join(val_dir, class_folder)
        os.makedirs(val_class_folder, exist_ok=True)

        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_class_folder, img)
            if copy:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)

        print(f"{class_folder}: {num_val} raw_image {'copied' if copy else 'moved'} to validation.")


# Example usage:
dataset_dir = '/train_data_for_CNN/dataset_vietnam_licenses_plate_train'  # Your dataset folder
val_dir = '/train_data_for_CNN/dataset_vietnam_license_plate_val'  # Where you want to store validation data
split_dataset(dataset_dir, val_dir, val_ratio=0.2, copy=True)  # Set copy=False to move instead of copy

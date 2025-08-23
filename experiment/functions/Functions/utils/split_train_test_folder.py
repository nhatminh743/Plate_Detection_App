import os
import shutil
import random

def split_dataset(data_dir, train_dir, val_dir, val_split=0.2, move_files=False):
    """
    Split raw_image in `data_dir` into train and validation folders.

    Args:
        data_dir (str): Directory containing image files.
        train_dir (str): Directory to save training raw_image.
        val_dir (str): Directory to save validation raw_image.
        val_split (float): Fraction of raw_image for validation (between 0 and 1).
        move_files (bool): If True, move files; otherwise, copy files.
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    image_files = [f for f in os.listdir(data_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    random.shuffle(image_files)

    split_index = int(len(image_files) * (1 - val_split))
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    for file in train_files:
        src = os.path.join(data_dir, file)
        dst = os.path.join(train_dir, file)
        if move_files:
            shutil.move(src, dst)
        else:
            shutil.copy(src, dst)

    for file in val_files:
        src = os.path.join(data_dir, file)
        dst = os.path.join(val_dir, file)
        if move_files:
            shutil.move(src, dst)
        else:
            shutil.copy(src, dst)

    print(f"Train: {len(train_files)} raw_image")
    print(f"Validation: {len(val_files)} raw_image")

# split_dataset(data_dir=r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/data/Raw',
#               train_dir=r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/data/train/images',
#               val_dir=r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/data/validation/images',
#               val_split=0.2,
#               move_files=True)
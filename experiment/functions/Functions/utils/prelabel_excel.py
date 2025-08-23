import numpy as np
import os
import shutil
from paddleocr import TextRecognition
from pathlib import Path
import cv2

#-------------------------- DECLARATION -----------------------------
input_dir = '/home/minhpn/Desktop/Green_Parking/Model_training/PaddleOCR_finetune/data/extracted_line'
output_dir = '/Model_training/PaddleOCR_finetune/temp_data/label'
log_file_path = os.path.join(output_dir, "low_confidence_predictions.txt")

model = TextRecognition()
os.makedirs(output_dir, exist_ok=True)

#-------------------------- MAIN PIPELINE -----------------------------

with open(log_file_path, "w", encoding="utf-8") as log_file:
    log_file.write("Filepath - Prediction - Confidence\n")
    log_file.write("="*60 + "\n")

    # Traverse all images recursively
    image_paths = list(Path(input_dir).rglob("*.jpg")) + list(Path(input_dir).rglob("*.png"))

    for img_path in image_paths:
        img_path = str(img_path)
        img = cv2.imread(img_path)

        try:
            output = model.predict(img_path)  # returns list of dicts

            for res in output:
                text = res['rec_text']
                confidence = res['rec_score']

                if confidence <= 0.85:
                    filename = os.path.basename(img_path)
                    new_path = os.path.join(output_dir, filename)
                    shutil.copy(img_path, new_path)

                    log_file.write(f"{img_path} - {text}\n")
                    print(f"[Saved] {filename}: {text} ({confidence:.5f})")
                else:
                    print(f"[Skip] {img_path}: {text} ({confidence:.5f})")

        except Exception as e:
            print(f"[Error] {img_path}: {str(e)}")




#Need to do:

# Create an txt file

# Loop through all sub-folder in the big folder

# Add image_path and it's prediction, confidence level to the txt file.

# Only take the one with confidence level <= 0.85

# Save the image into a seperate folder






        # for dir_path, dirnames, filenames in os.walk(img_folder):
        #     if not filenames:
        #         list_store = os.listdir(os.path.join(dir_path, "store"))
        #         for dirname in dirnames:
        #             if dirname != "logo":
        #                 continue
        #             output_subfolder = os.path.join(output_folder, dirname)
        #             os.makedirs(output_subfolder, exist_ok=True)
        #
        #             excel_list = []
        #             names = glob.glob(f"{os.path.join(dir_path, dirname)}/*.jpg")
        #
        #             for name in tqdm(names, desc='Processing'):
        #                 image_name = name.split("_")[-1].replace(".jpg", "")
        #                 # try:
        #                 labels = []
        #                 for store_name in list_store:
        #                     if image_name in store_name:
        #                         label = model.predict(img_path=os.path.join(dir_path, "store", store_name), label='', draw_output=False, model=model)[0].replace(" ", "")
        #                         labels.append(label)
        #                 # label = labels
        #                 # except Exception as e:
        #                 if labels == []:
        #                     labels = model.predict(img_path=name, label='', draw_output=False, model=model)[0].replace(" ", "")
        #                 excel_list.append((Path(name).name, np.nan, labels))
        #
        #             step = int(np.ceil(len(excel_list) / 1600))
        #             for i in range(step):
        #                 write_excel(excel_list[i::step],
        #                             os.path.join(dir_path, dirname),
        #                             output_subfolder,
        #                             order=i)

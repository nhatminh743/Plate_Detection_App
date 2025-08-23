import os
from experiment.functions.Functions.utils import extracted_letter_function


class PlateLetterExtractor:
    def __init__(self, data_dir, save_dir, imshow=False):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.imshow = imshow

        os.makedirs(self.save_dir, exist_ok=True)

    def extract_letters(self):
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith('.jpg'):
                file_path = os.path.join(self.data_dir, filename)
                # print(f"Processing {file_path} ...")
                extracted_letter_function.extract_letter_from_plate(self.save_dir, file_path, imshow=self.imshow)

        print("Processing completed.")



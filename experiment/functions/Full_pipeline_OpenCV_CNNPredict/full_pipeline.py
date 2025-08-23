import experiment.functions.Functions as F

ALPHA = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
         'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T',
         'U', 'V', 'X', 'Y', 'Z']

# Paths
RAW_DATA_DIR = r'/home/minhpn/Desktop/Green_Parking/Test_models/Data'
EXTRACTED_PLATE_DIR = r'/home/minhpn/Desktop/Green_Parking/Test_models/OpenCV_CNN/Extracted_Plate_Data'
LETTER_DIR = r'/home/minhpn/Desktop/Green_Parking/Test_models/OpenCV_CNN/Extracted_Letter_Data'
FINAL_RESULT_DIR = r'/home/minhpn/Desktop/Green_Parking/Test_models/OpenCV_CNN/Final_Result'
model_dir = r'/home/minhpn/Desktop/Green_Parking/Model_training/CNN_training/CNN_Model.keras'

def main_pipeline():
    # Step 1: Plate Extraction
    extractor = F.PlateExtractor(
        data_dir=RAW_DATA_DIR,
        save_dir=EXTRACTED_PLATE_DIR,
        debug_mode=False
    )
    extractor.process_images()

    # Step 2: Extract letter from plate
    letter_extraction = F.PlateLetterExtractor(
        data_dir=EXTRACTED_PLATE_DIR,
        save_dir=LETTER_DIR,
    )
    letter_extraction.extract_letters()

    # Step 3: Predict letter using CNN

    prediction_model = F.PlateCNNPredictor(
        model_path=model_dir,
        folder_path=LETTER_DIR,
        output_dir=FINAL_RESULT_DIR,
        ALPHA=ALPHA,
        add_blur=False,
    )
    prediction_model.predict_and_save()

if __name__ == '__main__':
    main_pipeline()


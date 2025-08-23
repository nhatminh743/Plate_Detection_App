import experiment.functions.Functions as F

# Paths
RAW_DATA_DIR = r'/home/minhpn/Desktop/Green_Parking/Test_models/Data'
EXTRACTED_PLATE_DIR = r'/home/minhpn/Desktop/Green_Parking/Test_models/OpenCV_EasyOCR/Extracted_Plate_Data'
FINAL_RESULT_DIR = r'/home/minhpn/Desktop/Green_Parking/Test_models/OpenCV_EasyOCR/Final_Result'

def main_pipeline():
    # Step 1: Plate Extraction
    extractor = F.PlateExtractor(
        data_dir=RAW_DATA_DIR,
        save_dir=EXTRACTED_PLATE_DIR,
        debug_mode=False
    )
    extractor.process_images()

    # Step 2: OCR Processing
    ocr_processor = F.PlateOCRProcessor(
        data_dir=EXTRACTED_PLATE_DIR,
        save_dir=FINAL_RESULT_DIR
    )
    ocr_processor.process_images()

if __name__ == '__main__':
    main_pipeline()

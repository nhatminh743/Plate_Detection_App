import experiment.functions.Functions as F
from experiment.functions.Functions.YOLO_plate_func import extracted_plate_YOLO

# Paths
RAW_DATA_DIR = r'/home/minhpn/Desktop/Green_Parking/Test_models/Data'
EXTRACTED_PLATE_DIR = r'/home/minhpn/Desktop/Green_Parking/Test_models/YOLO_EasyOCR/Extracted_Plate_Data'
FINAL_RESULT_DIR = r'/home/minhpn/Desktop/Green_Parking/Test_models/YOLO_EasyOCR/Final_Result'
YOLO_model_dir = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_training/runs/detect/train2/weights/best.pt'

def main_pipeline():
    # Step 1: Plate Extraction
    extractor = extracted_plate_YOLO.PlateExtractor(
        data_dir=RAW_DATA_DIR,
        save_dir=EXTRACTED_PLATE_DIR,
        best_model_file= YOLO_model_dir,
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

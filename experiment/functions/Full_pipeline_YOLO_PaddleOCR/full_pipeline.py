import time
from experiment.functions.Functions.YOLO_plate_func import extracted_plate_YOLO
from experiment.functions.Functions.utils import clear_directory, sort_txt_by_title
from experiment.functions.Functions.PaddleOCR.paddleOCR import PaddleOCRLineExtractor

RAW_DATA_DIR = r'/home/minhpn/Desktop/Green_Parking/one_image/saved'
EXTRACTED_PLATE_DIR = r'/home/minhpn/Desktop/Green_Parking/one_image/Extracted_Plate_Data'
EXTRACTED_LINE_DIR = r'/home/minhpn/Desktop/Green_Parking/one_image/Extracted_Line'
FINAL_RESULT_DIR = r'/home/minhpn/Desktop/Green_Parking/one_image/Final_Result'
YOLO_plate_dir = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_training/runs/detect/train2/weights/best.pt'
TEXT_RECOGNITION_DIR = r'/home/minhpn/Desktop/Green_Parking/Model_training/PaddleOCR_finetune/content/PaddleOCR/output/inference/PP-OCRv5_server_rec'

def main_pipeline():
    total_start = time.time()

    print("\nStep 0: Clearing directories...")
    t0 = time.time()
    clear_directory(EXTRACTED_PLATE_DIR)
    clear_directory(FINAL_RESULT_DIR)
    clear_directory(EXTRACTED_LINE_DIR)
    t_clear = time.time() - t0
    print(f"Clear directory done in {t_clear:.2f} seconds\n")

    print("Step 1: Extracting plates using YOLO...")
    t1 = time.time()
    extractor = extracted_plate_YOLO.PlateExtractor(
        data_dir=RAW_DATA_DIR,
        save_dir=EXTRACTED_PLATE_DIR,
        best_model_file= YOLO_plate_dir,
    )
    extractor.process_images()
    t_yolo = time.time() - t1
    print(f"Plate extraction done in {t_yolo:.2f} seconds\n")

    print("Step 2: OCR Line Extraction...")
    t2 = time.time()
    read = PaddleOCRLineExtractor(
        data_dir=EXTRACTED_PLATE_DIR,
        save_dir=FINAL_RESULT_DIR,
        temporary_dir=EXTRACTED_LINE_DIR,
        text_recognition_dir=TEXT_RECOGNITION_DIR
    )
    read.run()
    t_ocr = time.time() - t2
    print(f"OCR done in {t_ocr:.2f} seconds\n")

    total_time = time.time() - total_start
    print("====== Time Summary ======")
    print(f"Clear directories: {t_clear:.2f} sec")
    print(f"YOLO plate extraction: {t_yolo:.2f} sec")
    print(f"PaddleOCR: {t_ocr:.2f} sec")
    print(f"Total time: {total_time:.2f} sec")

    # Optional: highlight the slowest step
    steps = {
        'Clear directories': t_clear,
        'YOLO plate extraction': t_yolo,
        'PaddleOCR': t_ocr
    }
    slowest = max(steps, key=steps.get)
    print(f"Slowest step: {slowest} ({steps[slowest]:.2f} sec)")

if __name__ == "__main__":
    main_pipeline()

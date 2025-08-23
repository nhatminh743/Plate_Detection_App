#########################       DECLARE PATHS       #################################
import sys
from pathlib import Path
from experiment.functions.Functions.declaration import ROOT_DIR, join_path

ROOT_DIR = ROOT_DIR()
BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
sys.path.append(str(PARENT_DIR))
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "saved_uploads"
PLATE_DIR = STATIC_DIR / "extracted_plates"
LINE_DIR = STATIC_DIR / "extracted_lines"
RESULT_DIR = STATIC_DIR / "final_results"
PLOT_DIR = STATIC_DIR / "plotted_images"

YOLO_plate_model = join_path(r"experiment/model_training/YOLOv11_plate_detection/runs/detect/train2/weights/best.pt")
Paddle_OCR_Text_Recognition_Model = join_path(r"C:\Users\ACER\Documents\Plate_Detection\experiment\model_training\PaddleOCR_finetune\content\PaddleOCR\output\inference\PP-OCRv5_server_rec")

#########################    END OF DECLARE PATHS   #################################
#######################      IMPORT LIBRARIES     ################################

from fastapi import FastAPI, UploadFile, File
from typing import List
from datetime import datetime
import os
from fastapi.staticfiles import StaticFiles
from experiment.functions.Functions.utils import clear_directory, sort_txt_by_title
from experiment.functions.Functions.YOLO_plate_func import extracted_plate_YOLO
from experiment.functions.Functions.PaddleOCR.paddleOCR import PaddleOCRLineExtractor
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches

########################        DECLARE APP        ##################################
app = FastAPI()

##########################################      UTILS     ####################################################

def extract_pure_name(filename):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            return filename[:-4]
        elif filename.endswith('.jpeg'):
            return filename[:-5]
    # elif filename.endswith('.zip'):
    else:
        return {'error': 'Only accept file type of .jpg, .png, .jpeg'}

def create_unique_folder(filename, base_dir=UPLOAD_DIR):
    now = datetime.now()
    time_now = now.strftime('%Y-%m-%d_%H-%M-%S')
    folder_path = os.path.join(base_dir, f"{filename}_{time_now}")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

#########################     CREATE IMPORTANT DIR IF NOT     #################################

for d in [UPLOAD_DIR, PLATE_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

##########################     ALLOW USER TO ACCESS     #############################

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

###########################       FUNCTIONS         ##################################

@app.post('/upload-files-multiple')
async def upload_files_multiple(files: List[UploadFile] = File(...)):
    pure_filename = extract_pure_name(files[0].filename)
    folder = create_unique_folder(pure_filename)

    CURR_PLATE_DIR = create_unique_folder(pure_filename, base_dir=PLATE_DIR)
    CURR_RESULT_DIR = create_unique_folder(pure_filename, base_dir=RESULT_DIR)
    CURR_LINE_DIR = create_unique_folder(pure_filename, base_dir=LINE_DIR)

    for file in files:
        file_path = os.path.join(folder, file.filename)
        with open(file_path, 'wb') as f:
            f.write(await file.read())
            print(f'Saved: {file.filename}')

    req = ProcessRequest(
        session_path=folder,
        CURR_PLATE_DIR=CURR_PLATE_DIR,
        CURR_RESULT_DIR=CURR_RESULT_DIR,
        CURR_LINE_DIR= CURR_LINE_DIR
    )

    result = process_uploaded_folder(req)

    return result

@app.post('/upload-files-single')
async def upload_files_single(file: UploadFile = File(...)):
    pure_filename = extract_pure_name(file.filename)
    folder = create_unique_folder(pure_filename)

    CURR_PLATE_DIR = create_unique_folder(pure_filename, base_dir=PLATE_DIR)
    CURR_RESULT_DIR = create_unique_folder(pure_filename, base_dir=RESULT_DIR)
    CURR_LINE_DIR = create_unique_folder(pure_filename, base_dir=LINE_DIR)

    file_path = os.path.join(folder, file.filename)

    contents = await file.read()

    with open(file_path, 'wb') as f:
        f.write(contents)
    print(f'Saved: {file.filename}')

    req = ProcessRequest(
        session_path=folder,
        CURR_PLATE_DIR=CURR_PLATE_DIR,
        CURR_RESULT_DIR=CURR_RESULT_DIR,
        CURR_LINE_DIR= CURR_LINE_DIR,
    )

    image = Image.open(io.BytesIO(contents))

    result, dimension, org_dim, model_dim = process_uploaded_folder(req)
    fig, ax = plt.subplots()
    ax.imshow(image)

    print("Result keys:", result.keys())
    print("Dimensions:", len(dimension))

    if len(result) == 0:
        return JSONResponse(content={
            "text": 'No plate is detected in this picture.',
            "dimension_of_plate": dimension,
            'org_dim': org_dim,
            'model_dim': model_dim,
            "image": None
        })

    first_key = next(iter(result))
    texts = result[first_key].get("text", [])

    for i, bbox in enumerate(dimension):
        x1, y1, x2, y2 = bbox
        polygon_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        polygon = patches.Polygon(polygon_points, closed=True, edgecolor='blue', linewidth=2, facecolor='none')
        ax.add_patch(polygon)

        label_text = texts[i] if i < len(texts) else f"Plate {i + 1}"

        ax.text(
            x1, y1 - 5 - 15 * i,
            label_text,
            fontsize=10,
            color='white',
            bbox=dict(facecolor='blue', edgecolor='none', boxstyle='round,pad=0.2')
        )

    plt.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    img_bytes = buf.getvalue()
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    img_data_url = f"data:image/png;base64,{base64_image}"

    return JSONResponse(content={
        "text": result,
        "dimension_of_plate": dimension,
        'org_dim': org_dim,
        'model_dim': model_dim,
        "image": img_data_url
    })

class ProcessRequest(BaseModel):
    session_path: str  # The path returned by `/upload-file`
    CURR_PLATE_DIR: str
    CURR_RESULT_DIR: str
    CURR_LINE_DIR: str



@app.post("/process-folder")
def process_uploaded_folder(req: ProcessRequest):
    session_path = Path(req.session_path)

    CURR_LINE_DIR = Path(req.CURR_LINE_DIR)
    CURR_PLATE_DIR = Path(req.CURR_PLATE_DIR)
    CURR_RESULT_DIR = Path(req.CURR_RESULT_DIR)

    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session path does not exist.")

    if not os.path.exists(CURR_PLATE_DIR):
        raise HTTPException(status_code=404, detail="Current direction for saving plate does not exist.")

    if not os.path.exists(CURR_RESULT_DIR):
        raise HTTPException(status_code=404, detail="Current direction for saving result does not exist.")

    # Clear previous results
    clear_directory(str(CURR_PLATE_DIR))
    clear_directory(str(CURR_RESULT_DIR))
    clear_directory(str(CURR_LINE_DIR))

    # Step 1: Detect plates
    extractor = extracted_plate_YOLO.PlateExtractor(
        data_dir=str(session_path),
        save_dir=str(CURR_PLATE_DIR),
        best_model_file=str(YOLO_plate_model),
    )
    dimension, ori_dimension, model_dimension = extractor.process_images()

    # Step 2: Read characters
    read = PaddleOCRLineExtractor(
        data_dir=CURR_PLATE_DIR,
        save_dir=CURR_RESULT_DIR,
        temporary_dir=CURR_LINE_DIR,
        text_recognition_dir=Paddle_OCR_Text_Recognition_Model
    )
    read.run()

    # Step 3: Sort OCR results
    result_txt = CURR_RESULT_DIR / "ocr_results.txt"
    result_txt.touch(exist_ok=True)
    sort_txt_by_title(str(result_txt))

    # Step 4: Format result into JSON
    results = {}

    print("Opening result file:", result_txt)

    def extract_plate_from_text(text: str) -> str | None:
        replace_char = '.'
        # replace_char_2 = '-'
        new_text = text.replace(replace_char, '')
        # final_text = new_text.replace(replace_char_2, '')
        text = new_text.strip()
        patterns = [
            r'\b\d{2}-[A-Z]{2}\s\d{4,5}\b',  # e.g., 74-FA 12345
            r'\b\d{2}-\d{2}\s\d{4,5}\b',  # e.g., 74-44 12345
            r'\b\d{2}-[A-Z]\d\s\d{4,5}\b',  # e.g., 59-X4 12194
            r'\b\d{2}[A-Z]{1,2}[-]?\d{4,5}\b',  # e.g., 65F-12345, 74FA12345
            r'\b\d{2}[A-Z]{1,2}\s\d{4,5}\b',  # e.g., 74FA 12345
            r'\b\d{2}[A-Z]{1}\s\d{4,5}\b'     #30Y 9999
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None

    with open(result_txt, "r", encoding="utf-8") as f:
        for line in f:
            print(f"[Raw Line] {repr(line)}")

            if ':' not in line:
                print("  [Skipped] No colon found")
                continue

            try:
                img_name, ocr_text = line.strip().split(':', 1)
                short_name = Path(img_name.strip()).stem
                plate_candidate = extract_plate_from_text(ocr_text.strip())
                if not plate_candidate:
                    print(f"  [Filtered out] No valid plate found in: {ocr_text.strip()}")
                    continue

                if len(plate_candidate) <7 or len(plate_candidate) > 14:
                    print(plate_candidate)
                    print("  [Skipped] OCR text is too short / too long")
                    continue

                if not any(char.isdigit() for char in plate_candidate):
                    print(f"  [Filtered out] No digit found: {plate_candidate}")
                    continue

                if short_name not in results:
                    results[short_name] = {"text": []}

                results[short_name]["text"].append(plate_candidate)
                print(f"  [Accepted] {short_name}: {plate_candidate}")
            except Exception as e:
                print("  [Error parsing line]", e)

    print("Final results:", results)
    print("Final dimension:", dimension)

    return results, dimension, ori_dimension, model_dimension





################################### END OF MAIN FUNCTION #####################################################


#######################################      EXCESS CODE      ########################################

# @app.post("/upload-file")
# async def upload_file(file: UploadFile = File(...)):
#     pure_filename = extract_pure_name(file.filename)
#     folder = create_unique_folder(pure_filename)
#     file_path = os.path.join(folder, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())
#     print(f"Upload file successfully, located at: {folder}")
#
#     session_path = folder
#
#     clear_directory(str(PLATE_DIR))
#     clear_directory(str(RESULT_DIR))
#
#     # Step 1: Detect plates
#     extractor = extracted_plate_YOLO.PlateExtractor(
#         data_dir=str(session_path),
#         save_dir=str(PLATE_DIR),
#         best_model_file=str(YOLO_plate_model),
#     )
#     extractor.process_images()
#
#     # Step 2: Read characters
#     reader = letter_YOLO.LetterExtractor(
#         data_dir=str(PLATE_DIR),
#         save_dir=str(RESULT_DIR),
#         best_model_file=str(YOLO_read_model),
#         debug_mode=False,
#     )
#     reader.process_images()
#
#     # Step 3: Sort results
#     result_txt = RESULT_DIR / "ocr_results.txt"
#
#     if not result_txt.exists():
#         result_txt.touch()
#
#     sort_txt_by_title(str(result_txt))
#
#     # Step 4: Load results into JSON
#     results = {}
#     if result_txt.exists():
#         with open(result_txt, "r") as f:
#             for line in f:
#                 if ':' in line:
#                     img_name, ocr_text = line.strip().split(':', 1)
#                     plate_path = PLATE_DIR / f"{img_name.strip()[:12]}.jpg"
#                     results[img_name.strip()[:12]] = {
#                         "text": ocr_text.strip()
#                     }
#
#     return {"results": results}
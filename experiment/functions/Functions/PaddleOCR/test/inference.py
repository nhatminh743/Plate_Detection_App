import sys
from utils import parse_args

image_dir="/home/minhpn/Desktop/Green_Parking/one_image/Extracted_Line/0006_06797_b_plate_0/0006_06797_b_plate_0_roi_0_0.jpg"
rec_model_dir="/home/minhpn/Desktop/Green_Parking/Model_training/PaddleOCR_finetune/content/PaddleOCR/output/inference/PP-OCRv5_server_rec"

args = parse_args(
    use_gpu=False,
    image_dir=image_dir,
    rec_model_dir=rec_model_dir,
)

sys.path.append('/home/minhpn/anaconda3/envs/paddleTrainENV/lib/python3.10/site-packages/PaddleOCR/tools/infer')

from predict_rec import main

main(args)

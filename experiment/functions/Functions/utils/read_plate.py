import easyocr
import os

from experiment.functions.Functions.utils import postprocessing


class PlateOCRProcessor:
    def __init__(self, data_dir, save_dir, output_filename='ocr_results.txt'):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.output_file = os.path.join(save_dir, output_filename)
        os.makedirs(save_dir, exist_ok=True)
        self.reader = easyocr.Reader(['en'])

    def process_images(self):
        with open(self.output_file, 'w') as a:
            for filename in os.listdir(self.data_dir):
                if filename.lower().endswith('.jpg'):
                    file_path = os.path.join(self.data_dir, filename)

                    # image = cv2.imread(file_path)
                    # image_HUE = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    # V_channel = cv2.split(image_HUE)[2]
                    #
                    # T = threshold_local(V_channel, 15, offset = 10, method = 'gaussian')
                    #
                    # thresh = (V_channel > T).astype(uint8) * 255

                    res = self.reader.readtext(file_path, detail=0)

                    # res = self.reader.readtext(thresh, detail = 0)
                    print(f'Res: {res}')
                    res_str = ' '.join(res)
                    res = postprocessing.cleanUpPlate(res_str)
                    filename = filename[:12]
                    print(f"Processing: {filename} : {res}")
                    a.write(f"{filename}: {res}\n")

    def __repr__(self):
        return f"<PlateOCRProcessor(data_dir='{self.data_dir}', save_dir='{self.save_dir}')>"


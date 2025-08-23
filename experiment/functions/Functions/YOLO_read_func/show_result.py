import os
from ultralytics import YOLO
import cv2
from typing import List

def find_matching_plate_file(partial_name, directory):
    """
    Find a file in `directory` that starts with `partial_name` and contains '_plate'.
    Returns the filename if found, else None.
    """
    for fname in os.listdir(directory):
        if fname.startswith(partial_name) and '_plate' in fname and fname.endswith('.jpg'):
            return fname
    return None

class PlotImageS():
    def __init__(
            self,
            model_dir: str,
            image_dir: str,
            output_dir: str,
            bounding_box_font_size: float = 20,
            bounding_box_bezel_size: float = 4,
            list_of_selection: List[str] = [],
            scale_factor: int = 6,
            plot_selection: bool = False
    ):
        self.model: YOLO = YOLO(model_dir)  # YOLO model object
        self.image_dir: str = image_dir  # Path to input images
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir: str = output_dir  # Path to save output images
        self.scale_factor: int = scale_factor  # Image resize factor
        self.plot_selection: bool = plot_selection  # Flag to control selective plotting
        self.list_of_selection: List[str] = list_of_selection  # Selected images to plot (if any)
        self.bounding_box_font_size: float = bounding_box_font_size
        self.bounding_box_bezel_size: float = bounding_box_bezel_size

    def plot_all(self):
        if self.plot_selection == False:
            for img_filename in os.listdir(self.image_dir):
                img_path = os.path.join(self.image_dir, img_filename)

                original_image = cv2.imread(img_path)
                assert original_image is not None, f"Failed to load {img_path}"
                h, w, channel = original_image.shape

                # Resize image
                new_size = (w * self.scale_factor, h * self.scale_factor)
                resized_image = cv2.resize(original_image, new_size, interpolation=cv2.INTER_LINEAR)

                results = self.model.predict(resized_image, imgsz=new_size, conf=0.25, iou=0.7, agnostic_nms = True)[0]
                
                img_with_boxes = results.plot(font_size= self.bounding_box_font_size, pil=True,
                                              line_width= self.bounding_box_bezel_size)

                output_path = os.path.join(self.output_dir, img_filename)
                img_with_boxes.save(output_path)
                print(f'Saved {img_filename} to {output_path}')

        if self.plot_selection == True:
            if len(self.list_of_selection) == 0:
                raise Exception('No images selected while plot selection is True')
            for img_filename in self.list_of_selection:
                img_filename = find_matching_plate_file(img_filename, self.image_dir)
                img_path = os.path.join(self.image_dir, img_filename)

                original_image = cv2.imread(img_path)
                assert original_image is not None, f"Failed to load {img_path}"
                h, w, channel = original_image.shape

                # Resize image
                new_size = (w * self.scale_factor, h * self.scale_factor)
                resized_image = cv2.resize(original_image, new_size, interpolation=cv2.INTER_LINEAR)

                results = self.model.predict(resized_image, imgsz=new_size, conf=0.25, iou=0.7, agnostic_nms = True)[0]

                img_with_boxes = results.plot(font_size= 10, pil=True, line_width= 2)

                output_path = os.path.join(self.output_dir, img_filename)
                img_with_boxes.save(output_path)
                print(f'Saved {img_filename} to {output_path}')
        print()
        print("Successfully plotted all images in the direction")

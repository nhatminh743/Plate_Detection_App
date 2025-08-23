import cv2
import numpy as np
import os
from skimage import measure

def extract_letter_from_plate(save_path, filepath, imshow=False):
    # Read the image
    image = cv2.imread(filepath)
    if image is None:
        print(f"Error: Unable to read image at {filepath}")
        return

    # Split image horizontally (top and bottom halves)
    height = image.shape[0]
    mid_line = height // 2
    halves = [image[:mid_line, :], image[mid_line:, :]]

    # Prepare save directory
    filename = os.path.basename(filepath)
    folder_name = os.path.splitext(filename)[0]
    base_dir = os.path.join(save_path, folder_name)
    os.makedirs(base_dir, exist_ok=True)

    total_ROI = 0

    for idx, half in enumerate(halves):
        gray = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        morph_image = thresh

        labels = measure.label(morph_image, connectivity=2, background=0)
        properties = measure.regionprops(labels)

        properties = sorted(properties, key=lambda prop: prop.bbox[1])

        mask = np.zeros_like(gray)
        ROI_number = 0

        for prop in properties:
            minr, minc, maxr, maxc = prop.bbox
            w, h = maxc - minc, maxr - minr
            area = prop.area
            ratio = h / float(w) if w != 0 else 0

            if 35 < area < 300 and 0.3 < ratio < 4.2:
                ROI = thresh[minr:maxr, minc:maxc]
                cv2.rectangle(mask, (minc, minr), (maxc, maxr), 255, -1)

                if 1.0 <= ratio <= 1.5:
                    mid = w // 2
                    left_half = ROI[:, :mid]
                    right_half = ROI[:, mid:]

                    save_path_left = os.path.join(base_dir, f'half_{idx}_ROI_{ROI_number}.jpg')
                    cv2.imwrite(save_path_left, left_half)
                    ROI_number += 1

                    save_path_right = os.path.join(base_dir, f'half_{idx}_ROI_{ROI_number}.jpg')
                    cv2.imwrite(save_path_right, right_half)
                    ROI_number += 1
                else:
                    save_path_roi = os.path.join(base_dir, f'half_{idx}_ROI_{ROI_number}.jpg')
                    cv2.imwrite(save_path_roi, ROI)
                    ROI_number += 1

        total_ROI += ROI_number

        if imshow:
            cv2.imshow(f'Mask Half {idx}', mask)
            cv2.imshow(f'Morph Half {idx}', morph_image)

    if imshow:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Successfully extracted {total_ROI} letters from plate halves and saved to: {base_dir}")

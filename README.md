# Plate Detection Project

This project focuses on detecting license plates from vehicle images and extracting the corresponding plate numbers using computer vision techniques.

## Project Structure

| Folder                     | Description                                                           |
|----------------------------|-----------------------------------------------------------------------|
| frontend/                  | Gradio-based frontend interface for user interaction.                 |
| backend/                   | FastAPI backend for serving plate detection and recognition models.   |
| experiment/functions/      | Functions and pipeline needed for the applications.                   |
| experiment/model_training/ | Stores model training scripts, weights, and configuration files.      |
| experiment/test_folder/    | Contains data to benchmark model accuracy on some test images.        |

## How to run the project

### For deployment

1. Install Anaconda or miniconda

2. Create a new virtual environment and install necessary packages
- Open Anaconda Prompt (prefer as Administrator).
- Create a new environment using this line: `conda create -n deploymentENV python=3.10`
- Activate your environment: `conda activate deploymentENV`
- Change directory to your own local path for the Plate_Detection, for example mine is `C:\Users\ACER\Documents\nhatminh743\Plate_Detection` using this line:
`cd <your-path>`
- Install the requirements package using: `pip install -r requirements.txt`

3. Run backend
- Open new Anaconda Prompt (prefer as Administrator).
- Activate environment: `conda activate deploymentENV`
- Change directory to folder: E.g. `cd C:\Users\ACER\Documents\Plate_Detection`
- Run backend: `uvicorn backend.main:app --reload --port 8000`
- Check if backend run: http://127.0.0.1:8000/docs

4. Run frontend
- Open new Anaconda Prompt
- Activate environment: `conda activate deploymentENV`
- Change directory to frontend folder: E.g. `cd C:\Users\ACER\Documents\Plate_Detection\frontend`
- Run frontend: `gradio gradio_app.py`
- Check if frontend run: http://127.0.0.1:7860/

That's it.

## How the Project Works

### General Approach

The plate number is extracted using two main steps:

1. Plate Detection: Detect and crop the license plate from the vehicle image.
2. Text Recognition: Perform OCR on the cropped plate to extract the alphanumeric characters.

### Details Approach

---

#### 1. Plate Detection Approaches

---

##### Traditional OpenCV-Based Method

In the initial phase, I implemented a classical pipeline using OpenCV to detect license plates. The process involved:

- **Preprocessing:**
  - Resizing the image
  - Grayscale conversion
  - Smoothing and thresholding
  - Morphological operations to enhance features

- **Contour Detection:**
  - Finding contours in the image
  - Approximating the top contours to polygons
  - Selecting quadrilateral shapes (4 sides) likely to be plates

- **Plate Extraction:**
  - Cropping the detected plate region
  - Saving the cropped image for OCR (Optical Character Recognition)

While this method performed reasonably well on clear, front-facing images, it struggled significantly with:
- Angled or rotated plates  
- Blurry or low-resolution inputs  
- Images with strong shadows or reflections

Due to these limitations and poor test performance, I turned to a modern approach.

---

##### YOLO Detection

To improve robustness and accuracy, I adopted a YOLOv11 (medium) model for license plate detection. This version provided a better trade-off between accuracy and speed compared to the nano and small variants.

- The model was trained using annotated plate data and evaluated through standard metrics such as precision, recall, and mAP (mean Average Precision).
- The best-performing weights (`best.pt`) are stored in the `YOLOv11_training` directory within the `Model_training` folder.

The training curve (`result.png`) clearly demonstrates:
- Smooth and consistent loss reduction
- High precision and recall, reaching ~1.0
- Strong mAP@0.5 ≈ 1.0 and mAP@0.5:0.95 ≈ 0.85
- No signs of overfitting or underfitting

This suggests the model is not only learning effectively but also generalizes well.

---

#### Conclusion

While the traditional OpenCV-based method was a useful starting point, it lacked the robustness needed for real-world license plate detection.

In contrast, the YOLOv11 model significantly outperformed the classical approach by providing:
- High detection accuracy
- Robustness to varied lighting and angles
- Stable training performance with no major issues

Ultimately, YOLOv11 proved to be a reliable and production-ready solution for license plate detection.

--------
#### 2. Text Recognition

---

##### CNN Approach

My initial approach for text recognition was to train a Convolutional Neural Network (CNN) model to classify individual characters from cropped license plate images.

###### Model Architecture

The CNN model was built using Keras and had the following structure:

- Input size: `(12, 28, 3)`
- Two convolutional layers with 32 filters each, followed by max pooling
- One additional convolutional layer with 64 filters
- Fully connected layers:
  - A dense layer with 128 units
  - A softmax output layer with 31 classes (likely representing alphanumeric characters)

###### Dataset and Preprocessing

- Training data: 80 images/character
- Validation data: 20 images/character
- The dataset was collected from an online source and consisted of pre-thresholded character crops. However, the fonts and styles did not match real Vietnamese license plates, reducing generalizability.

To improve robustness, I applied the following augmentations using `ImageDataGenerator`:
- Rescaling
- Random shear, rotation, zoom
- Brightness and channel shift

###### Training Setup

- Optimizer: Adam with learning rate `0.0001`
- Loss function: Categorical cross-entropy
- Batch size: 8
- Early stopping was used to prevent overfitting

The model achieved a validation accuracy of **92.9%** during training.

###### Limitations

Despite the promising validation accuracy, the model performed poorly on real-world license plates due to:

- **Small dataset size** (only 100 images in total per character)
- **Mismatch in font style** between training data and actual plates
- **Lack of diverse environmental conditions** in the training set (e.g., lighting, blur, rotation)

###### Summary

This CNN approach was a good baseline for experimentation and achieved high performance on its limited validation set. However, due to data constraints and domain mismatch, it failed to generalize well in practice. This motivated the shift toward a more robust and scalable detection + OCR pipeline based on modern object detection models.

In addition, the intermediate step (extract letter from plate using thresholding and contours) also achieve poor result, making the overall prediction on the real world dataset not reliable.

--- 

##### EasyOCR Approach

After encountering poor performance with a CNN model trained on thresholded, contour-extracted characters, I adopted EasyOCR to handle full plate recognition directly, aiming for better generalization.

###### Implementation

EasyOCR was applied on the entire plate region detected. The process involved:

- Reading all `.jpg` files in a directory.
- Passing them to `easyocr.Reader(['en'])` for text extraction.
- Cleaning up the text output using `postprocessing.cleanUpPlate()` — this function reordered characters based on their bounding box positions.
- Writing the final results to a `.txt` file for evaluation.

###### Observations

- Preprocessing methods like HSV thresholding and adaptive binarization hurt performance, so raw image input was preferred.
- The EasyOCR model handled many real-world scenarios better than the CNN, though Vietnamese plate fonts still caused some confusion (e.g., misreading similar characters).
- No major slowdowns were observed during batch inference, even on larger sets.

###### Performance

Although EasyOCR required no training and was more robust to lighting and noise, its overall accuracy was still not sufficient for production-level deployment. Nonetheless, it provided a clear improvement over the CNN baseline and served as a useful benchmark for comparison.

---

##### YOLO Detection

To improve robustness and accuracy, I adopted a YOLOv11 (large) model for license plate detection.

- The model was trained using annotated plate data and evaluated through standard metrics such as precision, recall, and mAP (mean Average Precision).
- The best-performing weights (`best.pt`) are stored in the `YOLOv11_Detect_Number_From_Plate` directory within the `Model_training` folder.

The training curve (`result.png`) clearly demonstrates:
- Smooth and consistent loss reduction
- High precision and recall, reaching ~1.0
- Strong mAP@0.5 ≈ 1.0 and mAP@0.5:0.95 ≈ 0.85
- No signs of overfitting or underfitting (original dataset)

This suggests the model is not only learning effectively but also generalizes well.

---

##### Paddle OCR

PaddleOCR Approach

To further enhance text recognition, I employed PaddleOCR, which combines Text Segmentation and Text Recognition in a unified framework.

- Implementation and Fine-tuning:

  - I fine-tuned the PaddleOCR Text Recognition model to improve performance on Vietnamese license plates.

One limitation observed with the YOLO-based approach was that the model, trained on the original dataset (which contained only one plate image taken at a fixed angle from a security camera), performed poorly on images captured from different angles, such as phone pictures.

To address the insufficient data for PaddleOCR, I generated additional synthetic training data. This involved using a text generator and selecting fonts similar to Vietnamese license plate styles to improve the realism and diversity of the training set.

This approach allowed PaddleOCR to better generalize to real-world variations in plate appearance and capture angle, providing a more robust alternative to the previous CNN and EasyOCR experiments.

#### Conclusion

Conclusion

While CNN segmentation with EasyOCR provided a simple OCR pipeline, it lacked reliability and scalability for real-world conditions. It suffered from low accuracy, unstable character ordering, and heavy dependence on postprocessing.

YOLOv11 detection improved robustness and accuracy, handling noise and distortions more effectively with minimal manual intervention. However, its performance was limited when dealing with variations in plate angles and real-world diversity.

In contrast, PaddleOCR, fine-tuned with additional synthetic data and realistic Vietnamese fonts, demonstrated superior generalization and accuracy across diverse scenarios. It effectively handled variations in lighting, angle, and plate style, making 
it the most reliable and scalable solution for license plate recognition and the preferred choice for deployment in real-world applications.

--- 

### Connect Two Main Steps:

I connect total of 5 pipeline to process the plate image and test them on a test set (original dataset):

1. OpenCV - CNN: 0%
2. OpenCV - EasyOCR: 21%
3. YOLO - CNN: 4%
4. YOLO - EasyOCR: 36%
5. YOLO - YOLO: 94%

The test set contains harsher condition compare to overall train set.

I compare the YOLO - YOLO pipeline and YOLO - PaddleOCR on a test set different from original dataset with multiple conditions (multiple plates, blurry pictures, day and night, etc):

1. YOLO - YOLO: 35%
2. YOLO - PaddleOCR: 62%

Note: This test was conducted on a different machine, and the original test set is no longer available by the time this README was updated.

---

## Deployment

For deployment, I used **FastAPI** as the backend and **Gradio** as the frontend due to their popularity, strong performance, and ease of use.

The system is designed to support two main use cases:

1. **Mass Upload**: Users can upload multiple images at once. The system returns results in the format:  
   `image_name: detected_text`

2. **Single Image Processing**: Users can upload a single image and receive:
   - The image with the detected license plate labeled alongside its recognized text
   - The extracted plate number in text form for easy copying

---

## Project Limitations

The model was trained on a dataset collected from a parking company's check-in system and multiple source collected from the internet.

This limits generalization to:
- Plates at sharp angles (too difficult to read).
- Plates at a different ratio to image (plate is too big in an image, too small in an image).
- Plates from a different country (the model is optimized for Vietnamese plates).
- More diverse real-world scenes (e.g., traffic or surveillance footage)

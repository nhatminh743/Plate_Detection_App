import os
import cv2
import numpy as np
import keras
from experiment.functions.Functions.utils import postprocessing


class PlateCNNPredictor:
    def __init__(self, model_path, folder_path, output_dir, ALPHA, add_blur=False, image_size=(28, 12)):
        self.model_path = model_path
        self.folder_path = folder_path
        self.output_dir = output_dir
        self.ALPHA = ALPHA
        self.image_size = image_size
        self.add_blur = add_blur
        self.average_confidence = None

        os.makedirs(self.output_dir, exist_ok=True)
        self.model = keras.models.load_model(self.model_path)

    def predict_and_save(self):
        output_path = os.path.join(self.output_dir, 'ocr_results.txt')
        with open(output_path, 'w') as f:
            for subfolder in os.listdir(self.folder_path):
                subfolder_path = os.path.join(self.folder_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue  # Skip non-folders

                predicted_text, confidences = self._predict_single_plate(subfolder_path)
                cleaned_text = postprocessing.cleanUpPlate(predicted_text)

                # Write result to file
                f.write(f"{subfolder[:12]}: {cleaned_text}\n")
                print(f"Saved result for {subfolder[:12]}: {cleaned_text}")

                # Print per-character confidence
                print("Character Predictions with Confidence:")
                for char, conf in zip(predicted_text, confidences):
                    print(f"Character: {char}, Confidence: {conf:.2f}")

                # Print average confidence
                if confidences:
                    avg_conf = np.mean(confidences)
                    print(f"Average Confidence: {avg_conf:.2f}")
                else:
                    print("No valid predictions for this plate.")

                print()

        print(f"\nAll results saved to {output_path}")

    def _predict_single_plate(self, subfolder_path):
        image_files = sorted([
            f for f in os.listdir(subfolder_path)
            if f.lower().endswith(('.jpg', '.png'))
        ])

        predicted_text = ''
        confidences = []
        for filename in image_files:
            image_path = os.path.join(subfolder_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Warning: Unable to read image {filename} in {os.path.basename(subfolder_path)}, skipping...")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.add_blur:
                kernel = (3, 3)
                image = cv2.blur(image, kernel)
            image = cv2.resize(image, self.image_size)
            image = image.astype(np.float32) / 255.0
            input_batch = np.expand_dims(image, axis=0)

            prediction = self.model.predict(input_batch, verbose=0)[0]
            predicted_index = np.argmax(prediction)
            predicted_label = self.ALPHA[predicted_index]
            confidence = prediction[predicted_index]

            if confidence >= 0.5:
                predicted_text += predicted_label
                confidences.append(confidence)
            else:
                print(f"Skipping character '{predicted_label}' due to low confidence ({confidence:.2f})")

        return predicted_text, confidences


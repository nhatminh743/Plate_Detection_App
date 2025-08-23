# This is the pipeline I create using model and techniques

## First part: Extracting Plate from Image using OpenCV (no model)

## Second part: Using EasyOCR to read the text from the extracted plate

I want my code to check specifically the format of the first 4 character - Except the third one always be a letter, the rest will be always a number. After the fourth character, indend by a space if it didn't have and remove any extrac comma or dot. The rest after fourth should always be number. All the character should be number except the third one.
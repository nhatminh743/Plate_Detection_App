#Need a code: Read through each file in a folder (assume folder only contains file)

# Give the option to clean up the txt_file before execution, default=True=clear the file

# Write it into a txt_file, directory known with the format:

# Filename = CustomName_RandomNumber.jpg => Extract CustomName only

# Write: <Absolute path>/t<CustomName>

# User can give absolute path to the current directory

import os

def write_file_summary(
    folder_path: str,
    output_txt_path: str,
    clear_file: bool = True
):
    """
    Processes all files in a folder and writes:
    <absolute_path>\t<CustomName>
    into a text file. Assumes filenames are like: CustomName_RandomNumber.jpg

    Args:
        folder_path (str): Absolute path to the folder with files.
        output_txt_path (str): Path to the .txt file to write.
        clear_file (bool): Whether to clear the file before writing. Default: True
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if clear_file:
        open(output_txt_path, 'w').close()

    with open(output_txt_path, 'a') as outfile:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # Extract "CustomName" from "CustomName_RandomNumber.jpg"
                if '_' in filename:
                    custom_name = filename.split('_')[0]
                    abs_path = os.path.abspath(file_path)
                    outfile.write(f"{abs_path}\t{custom_name}\n")

write_file_summary(
    folder_path="/home/minhpn/Desktop/Green_Parking/Model_training/PaddleOCR_finetune/data/label",
    output_txt_path="/home/minhpn/Desktop/Green_Parking/Model_training/PaddleOCR_finetune/data/final.txt",
    clear_file=True
)
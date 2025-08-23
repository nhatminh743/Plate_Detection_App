import os

def write_filenames_to_txt(root_dir, output_txt_path, recursive=False):
    """
    Write sorted file names in root_dir to output_txt_path.

    Parameters:
        root_dir (str): Directory to scan.
        output_txt_path (str): Output text file to save the file names.
        recursive (bool): Whether to include files in subdirectories.
    """
    file_list = []

    if recursive:
        # Walk through all subdirectories
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                file_list.append(file)
    else:
        # Only files in root_dir
        for file in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file)
            if os.path.isfile(file_path):
                file_list.append(file)

    # Sort the file names alphabetically
    file_list = sorted(file_list)

    # Write to file
    with open(output_txt_path, 'w') as f:
        for file_name in file_list:
            f.write(file_name +': ' +'\n')  # Removed the colon for cleaner output

    print(f"Saved {len(file_list)} sorted file names to {output_txt_path}")


# write_filenames_to_txt("/home/minhpn/Desktop/Green_Parking/Test_models/Data", output_txt_path="/home/minhpn/Desktop/Green_Parking/Test_models/Validation.txt", recursive=True)
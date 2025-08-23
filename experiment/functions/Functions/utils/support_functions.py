import numpy as np
import math
import os
import shutil

def angle(pt1, pt2, pt0):
    dx1 = pt1[0] - pt0[0]
    dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]
    dy2 = pt2[1] - pt0[1]
    angle = math.degrees(math.acos(
        (dx1 * dx2 + dy1 * dy2) /
        (math.sqrt((dx1 ** 2 + dy1 ** 2) * (dx2 ** 2 + dy2 ** 2)) + 1e-10)
    ))
    return angle



def clear_directory(dir_path):
    """
    Deletes all files and folders in the given directory.

    Parameters:
        dir_path (str): Path to the directory to clear.
    """
    if not os.path.exists(dir_path):
        print(f"Directory does not exist: {dir_path}")
        return

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # delete file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # delete folder
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def create_txt_file(file_path, content=""):
    """
    Creates a new .txt file with optional initial content.

    Parameters:
        file_path (str): Path where the file will be created.
        content (str): Initial content to write into the file. Default is empty.
    """
    with open(file_path, 'w') as f:
        f.write(content)


from .extract_plate_function import detect_license_plate
from .extracted_letter import PlateLetterExtractor
from .extracted_letter_function import extract_letter_from_plate
from .extracted_plate import PlateExtractor
from .mass_thresh_image import threshold_images
from .postprocessing import remove_specific_characters, build_char_map, map_character, correct_text, cleanUpPlate, upperCase, is_letter, is_number
# from .read_plate import PlateOCRProcessor
from .split_train_test_folder import split_dataset
from .predict_usingCNN import PlateCNNPredictor
from .support_functions import angle, clear_directory, create_txt_file
from .write_all_filename import write_filenames_to_txt
from .sort_alphabetically_txt import sort_txt_by_title

__all__ = [
    'detect_license_plate',
    'PlateLetterExtractor',
    "extract_letter_from_plate",
    "PlateExtractor",
    "threshold_images",
    "remove_specific_characters",
    "build_char_map",
    "map_character",
    'correct_text',
    'cleanUpPlate',
    'upperCase',
    'is_number',
    'is_letter',
    # 'PlateOCRProcessor',
    "split_dataset",
    "PlateCNNPredictor",
    "angle",
    "clear_directory",
    "create_txt_file",
    "write_filenames_to_txt",
    "sort_alphabetically_txt",
]
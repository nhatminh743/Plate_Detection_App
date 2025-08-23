def sort_txt_by_title(input_file, output_file=None):
    """
    Sort lines in a TXT file by the part before ':' and save result.

    Parameters:
        input_file (str): Path to the input TXT file.
        output_file (str, optional): Path to save sorted file. If None, overwrite input file.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Sort lines by title before ':'
    sorted_lines = sorted(lines, key=lambda x: x.split(':')[0].strip())

    # Write result
    output_path = output_file if output_file else input_file
    with open(output_path, 'w') as f:
        f.writelines(sorted_lines)

    print(f"Sorted lines saved to {output_path}")

# sort_txt_by_title('/home/minhpn/Desktop/Green_Parking/Test_models/YOLO_EasyOCR/Final_Result/ocr_results.txt')
# sort_txt_by_title('/home/minhpn/Desktop/Green_Parking/Test_models/YOLO_CNN/Final_Result/ocr_results.txt')
# sort_txt_by_title('/home/minhpn/Desktop/Green_Parking/Test_models/OpenCV_EasyOCR/Final_Result/ocr_results.txt')
# sort_txt_by_title('/home/minhpn/Desktop/Green_Parking/Test_models/OpenCV_CNN/Final_Result/ocr_results.txt')
# sort_txt_by_title('/home/minhpn/Desktop/Green_Parking/Test_models/YOLO_YOLO/Final_Result/ocr_results.txt')
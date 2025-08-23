input_file = r'/Model_training/PaddleOCR_finetune/temp_data/label/low_confidence_predictions.txt'  # Replace with your actual input file path
output_file = r'/Model_training/PaddleOCR_finetune/data/final.txt'

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue  # Skip blank lines

        parts = line.rsplit(" - ", maxsplit=1)
        if len(parts) < 2:
            continue  # Skip malformed lines

        image_path = parts[0].strip()
        label = parts[1].strip()

        if not label or label == "-":
            continue  # Skip empty or dash-only labels

        outfile.write(f"{image_path}\t{label}\n")

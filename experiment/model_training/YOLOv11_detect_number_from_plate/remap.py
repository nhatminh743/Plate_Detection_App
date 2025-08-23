import os

# Define the directory you want to use
working_dir = "/Model_training/YOLOv11_Detect_Number_From_Plate/data/train"

# Check if it exists
if os.path.exists(working_dir):
    os.chdir(working_dir)
    print(f"Working directory changed to: {os.getcwd()}")
else:
    print(f"Directory does not exist: {working_dir}")

def load_classes(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def create_class_index_map(old_classes, new_classes):
    return {i: new_classes.index(name) for i, name in enumerate(old_classes)}


def remap_yolo_labels(labels_dir, old_class_file, new_class_file):
    old_classes = load_classes(old_class_file)
    print('Loading classes...')
    print('Classes:', len(old_classes))
    print(old_classes)
    new_classes = load_classes(new_class_file)

    index_map = create_class_index_map(old_classes, new_classes)

    for file in os.listdir(labels_dir):

        if not file.endswith('.txt'):
            continue

        path = os.path.join(labels_dir, file)
        with open(path, 'r') as f:
            lines = f.readlines()


        new_lines = []
        for line in lines:
            if not line.strip(): continue
            parts = line.strip().split()

            old_idx = int(parts[0])

            new_idx = index_map[old_idx]

            parts[0] = str(new_idx)
            new_lines.append(' '.join(parts))

        with open(path, 'w') as f:
            f.write('\n'.join(new_lines) + '\n')

    print(f"[✓] Remapped all label files in: {labels_dir}")

remap_yolo_labels(
    labels_dir='/Model_training/YOLOv11_Detect_Number_From_Plate/data/validation/labels',  # or val/labels
    old_class_file=r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/data/validation/classes.txt',
    new_class_file=r'/Model_training/YOLOv11_Detect_Number_From_Plate/data/validation/labels/classes.txt'
)

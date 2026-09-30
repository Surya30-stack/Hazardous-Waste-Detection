from pathlib import Path
import cv2
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"F:\Proitbridge study course\Hazardous waste detection\Study materials\VP\New folder\Code\Datasets"
)

CLASS_NAMES = {
    0: "Cylinder",
    1: "Shock Absorber"
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLITS = ["train", "valid", "test"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_files(images_dir):
    """Return all supported image files from a folder."""
    
    if not images_dir.exists():
        return []

    return [
        file for file in images_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_label_files(labels_dir):
    """Return all YOLO label files from a folder."""
    
    if not labels_dir.exists():
        return []

    return [
        file for file in labels_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() == ".txt"
    ]


def read_segmentation_labels(label_path):
    """
    Read YOLO segmentation annotations.

    Format:
    class_id x1 y1 x2 y2 x3 y3 ...

    Returns:
        list of tuples:
        (class_id, [(x1,y1), (x2,y2), ...])
    """

    annotations = []

    try:
        with open(label_path, "r") as file:
            lines = file.readlines()

    except Exception as e:
        print(f"Could not read {label_path.name}: {e}")
        return annotations

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        # Ignore empty lines
        if not line:
            continue

        values = line.split()

        # Minimum:
        # class_id + at least 3 coordinate pairs
        if len(values) < 7:
            continue

        # After class ID, coordinates must come in x,y pairs
        if (len(values) - 1) % 2 != 0:
            continue

        try:
            class_id = int(float(values[0]))

            coordinates = [
                float(value)
                for value in values[1:]
            ]

        except ValueError:
            continue

        # Class ID check
        if class_id not in CLASS_NAMES:
            continue

        points = []

        for i in range(0, len(coordinates), 2):

            x = coordinates[i]
            y = coordinates[i + 1]

            points.append((x, y))

        annotations.append(
            (class_id, points)
        )

    return annotations


def validate_segmentation_label(label_path):
    """
    Validate a YOLO segmentation label file.

    Returns:
        valid_count
        invalid_count
        invalid_reasons
    """

    valid_count = 0
    invalid_count = 0
    invalid_reasons = []

    try:
        with open(label_path, "r") as file:
            lines = file.readlines()

    except Exception as e:
        return 0, 1, [f"Could not read file: {e}"]

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        values = line.split()

        # class + x/y pairs
        if len(values) < 7:
            invalid_count += 1
            invalid_reasons.append(
                f"Line {line_number}: not enough values for segmentation"
            )
            continue

        if (len(values) - 1) % 2 != 0:
            invalid_count += 1
            invalid_reasons.append(
                f"Line {line_number}: coordinate values are not in x/y pairs"
            )
            continue

        try:
            class_id = int(float(values[0]))

            coordinates = [
                float(value)
                for value in values[1:]
            ]

        except ValueError:
            invalid_count += 1
            invalid_reasons.append(
                f"Line {line_number}: non-numeric value found"
            )
            continue

        # Check class
        if class_id not in CLASS_NAMES:
            invalid_count += 1
            invalid_reasons.append(
                f"Line {line_number}: invalid class ID {class_id}"
            )
            continue

        # Check coordinates
        coordinates_valid = True

        for coordinate in coordinates:

            if coordinate < 0 or coordinate > 1:
                coordinates_valid = False
                break

        if not coordinates_valid:
            invalid_count += 1
            invalid_reasons.append(
                f"Line {line_number}: coordinates outside 0-1 range"
            )
            continue

        # At least 3 points required for a polygon
        if len(coordinates) < 6:
            invalid_count += 1
            invalid_reasons.append(
                f"Line {line_number}: polygon has fewer than 3 points"
            )
            continue

        valid_count += 1

    return valid_count, invalid_count, invalid_reasons


def draw_segmentation(image_path, label_path):
    """
    Draw YOLO segmentation polygons on an image.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]

    annotations = read_segmentation_labels(label_path)

    for class_id, points in annotations:

        pixel_points = []

        for x, y in points:

            px = int(x * width)
            py = int(y * height)

            pixel_points.append([px, py])

        if len(pixel_points) < 3:
            continue

        # Convert to NumPy array
        import numpy as np

        polygon = np.array(
            pixel_points,
            dtype=np.int32
        )

        # Draw polygon outline
        cv2.polylines(
            image,
            [polygon],
            isClosed=True,
            color=(255, 0, 0),
            thickness=3
        )

        # Create transparent polygon overlay
        overlay = image.copy()

        cv2.fillPoly(
            overlay,
            [polygon],
            color=(255, 0, 0)
        )

        image = cv2.addWeighted(
            overlay,
            0.20,
            image,
            0.80,
            0
        )

        # Find label position
        x_min = min(point[0] for point in pixel_points)
        y_min = min(point[1] for point in pixel_points)

        class_name = CLASS_NAMES[class_id]

        # Draw class name
        cv2.putText(
            image,
            class_name,
            (x_min, max(y_min - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    return image


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("HAZARDOUS WASTE DATASET CHECK")
print("=" * 70)

print("\nDataset path:")
print(DATASET_DIR)


# ============================================================
# DATASET EXISTENCE CHECK
# ============================================================

if not DATASET_DIR.exists():

    print("\nERROR: Dataset folder does not exist.")
    print("Please check DATASET_DIR in this script.")
    raise SystemExit

print("\nDataset folder found successfully.")


# ============================================================
# DATASET STRUCTURE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATASET STRUCTURE CHECK")
print("=" * 70)

split_status = {}

for split in SPLITS:

    split_dir = DATASET_DIR / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    print(f"\n{split.upper()}")

    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")

    if not split_dir.exists():

        print(f"ERROR: {split} folder does not exist.")

        split_status[split] = False

    elif not images_dir.exists():

        print("ERROR: images folder does not exist.")

        split_status[split] = False

    elif not labels_dir.exists():

        print("ERROR: labels folder does not exist.")

        split_status[split] = False

    else:

        print("Status: OK")

        split_status[split] = True


# ============================================================
# GLOBAL STATISTICS
# ============================================================

total_images_all = 0
total_labels_all = 0
total_objects_all = 0
total_invalid_all = 0

overall_class_objects = {
    class_id: 0
    for class_id in CLASS_NAMES
}

overall_class_images = {
    class_id: 0
    for class_id in CLASS_NAMES
}


# ============================================================
# CHECK EACH DATASET SPLIT
# ============================================================

for split in SPLITS:

    if not split_status.get(split, False):

        print("\n" + "=" * 70)
        print(f"{split.upper()} DATASET CHECK")
        print("=" * 70)

        print(
            f"\nSkipping {split} because required folders are missing."
        )

        continue

    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"

    images = get_image_files(images_dir)
    labels = get_label_files(labels_dir)

    print("\n" + "=" * 70)
    print(f"{split.upper()} DATASET CHECK")
    print("=" * 70)

    # --------------------------------------------------------
    # IMAGE INFORMATION
    # --------------------------------------------------------

    print("\nIMAGE INFORMATION")
    print("-" * 50)

    print(f"Total images: {len(images)}")
    print(f"Total label files: {len(labels)}")

    total_images_all += len(images)
    total_labels_all += len(labels)

    # --------------------------------------------------------
    # IMAGE-LABEL MATCHING
    # --------------------------------------------------------

    print("\nIMAGE-LABEL CHECK")
    print("-" * 50)

    label_names = {
        label.stem
        for label in labels
    }

    missing_labels = []

    for image in images:

        if image.stem not in label_names:

            missing_labels.append(
                image.name
            )

    print(
        f"Images without labels: {len(missing_labels)}"
    )

    if missing_labels:

        print("\nExamples:")

        for name in missing_labels[:10]:
            print(f" - {name}")

    else:

        print("All images have corresponding label files.")

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\nCLASS DISTRIBUTION")
    print("-" * 50)

    class_objects = {
        class_id: 0
        for class_id in CLASS_NAMES
    }

    class_images = {
        class_id: 0
        for class_id in CLASS_NAMES
    }

    total_objects = 0
    invalid_annotations = 0

    invalid_examples = []

    for label_path in labels:

        annotations = read_segmentation_labels(
            label_path
        )

        # Count objects
        for class_id, points in annotations:

            class_objects[class_id] += 1
            total_objects += 1

        # Count images containing class
        classes_in_image = {
            class_id
            for class_id, points in annotations
        }

        for class_id in classes_in_image:

            class_images[class_id] += 1

        # Validate
        valid_count, invalid_count, reasons = (
            validate_segmentation_label(label_path)
        )

        invalid_annotations += invalid_count

        if invalid_count > 0:

            for reason in reasons[:5]:

                invalid_examples.append(
                    f"{label_path.name} - {reason}"
                )

    # --------------------------------------------------------
    # PRINT CLASS DISTRIBUTION
    # --------------------------------------------------------

    for class_id, class_name in CLASS_NAMES.items():

        print(f"\n{class_name}")

        print(
            f"  Objects: {class_objects[class_id]}"
        )

        print(
            f"  Images containing class: "
            f"{class_images[class_id]}"
        )

        overall_class_objects[class_id] += (
            class_objects[class_id]
        )

        overall_class_images[class_id] += (
            class_images[class_id]
        )

    print(
        f"\nTotal annotated objects: {total_objects}"
    )

    total_objects_all += total_objects

    # --------------------------------------------------------
    # ANNOTATION QUALITY
    # --------------------------------------------------------

    print("\nANNOTATION QUALITY CHECK")
    print("-" * 50)

    print(
        f"Invalid annotations: {invalid_annotations}"
    )

    if invalid_annotations == 0:

        print("\nAll segmentation annotations are valid.")

    else:

        print("\nExamples:")

        for example in invalid_examples[:20]:

            print(f" - {example}")

    total_invalid_all += invalid_annotations


# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OVERALL DATASET SUMMARY")
print("=" * 70)

print(f"\nTotal images: {total_images_all}")
print(f"Total label files: {total_labels_all}")

print(
    f"Total annotated objects: {total_objects_all}"
)

print(
    f"Total invalid annotations: {total_invalid_all}"
)

print("\nCLASS DISTRIBUTION")
print("-" * 50)

for class_id, class_name in CLASS_NAMES.items():

    print(f"\n{class_name}")

    print(
        f"  Total objects: "
        f"{overall_class_objects[class_id]}"
    )

    print(
        f"  Images containing class: "
        f"{overall_class_images[class_id]}"
    )


# ============================================================
# VALIDATION SPLIT WARNING
# ============================================================

if not split_status.get("valid", False):

    print("\n" + "=" * 70)
    print("VALIDATION SPLIT NOTICE")
    print("=" * 70)

    print(
        "\nThe 'valid' folder was not found."
    )

    print(
        "Your current local dataset contains train/test "
        "but no valid folder."
    )

    print(
        "\nThis is NOT an annotation error."
    )

    print(
        "We will handle the validation split separately "
        "before training."
    )


# ============================================================
# DISPLAY SAMPLE ANNOTATED IMAGES
# ============================================================

print("\n" + "=" * 70)
print("DISPLAYING SAMPLE ANNOTATED IMAGES")
print("=" * 70)


def display_samples(split, number_of_samples=3):

    if not split_status.get(split, False):
        return

    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"

    images = get_image_files(images_dir)

    samples_displayed = 0

    for image_path in images:

        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        annotations = read_segmentation_labels(
            label_path
        )

        if not annotations:
            continue

        annotated_image = draw_segmentation(
            image_path,
            label_path
        )

        if annotated_image is None:
            continue

        plt.figure(figsize=(10, 7))

        plt.imshow(annotated_image)

        plt.title(
            f"{split.upper()} - {image_path.name}"
        )

        plt.axis("off")

        plt.tight_layout()

        plt.show()

        samples_displayed += 1

        if samples_displayed >= number_of_samples:
            break

    if samples_displayed == 0:

        print(
            f"No valid annotated images found in {split}."
        )


print("\nDisplaying samples from TRAIN:")

display_samples(
    "train",
    number_of_samples=3
)


print("\nDisplaying samples from TEST:")

display_samples(
    "test",
    number_of_samples=3
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("DATASET CHECKING COMPLETED")
print("=" * 70)

if total_invalid_all == 0:

    print(
        "\nSUCCESS: No invalid segmentation annotations detected."
    )

else:

    print(
        f"\nWARNING: {total_invalid_all} invalid "
        "segmentation annotations detected."
    )

print()
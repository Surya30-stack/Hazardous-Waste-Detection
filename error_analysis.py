from ultralytics import YOLO
from pathlib import Path
import cv2
import csv


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "runs"
    / "detect"
    / "Hazardous_Waste_Detection"
    / "yolov8n_run-2"
    / "weights"
    / "best.pt"
)

IMAGE_DIR = Path("Datasets/test/images")
LABEL_DIR = Path("Datasets/test/labels")

OUTPUT_DIR = Path(
    "runs/detect/error_analysis"
)

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")

print("\nClasses:")
for class_id, class_name in model.names.items():
    print(f"  {class_id}: {class_name}")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# BOUNDING BOX IOU
# ============================================================

def calculate_box_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union = (
        area1
        + area2
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# READ GROUND-TRUTH LABELS
# ============================================================

def read_ground_truth(
    label_path,
    image_width,
    image_height
):

    ground_truth = []

    if not label_path.exists():
        return ground_truth

    with open(label_path, "r") as file:

        lines = file.readlines()

    for line in lines:

        values = line.strip().split()

        if len(values) < 5:
            continue

        class_id = int(values[0])

        coordinates = list(
            map(float, values[1:])
        )

        # ----------------------------------------------------
        # NORMAL YOLO BOUNDING BOX
        # class x_center y_center width height
        # ----------------------------------------------------

        if len(coordinates) == 4:

            x_center = coordinates[0]
            y_center = coordinates[1]
            width = coordinates[2]
            height = coordinates[3]

            x1 = (
                x_center - width / 2
            ) * image_width

            y1 = (
                y_center - height / 2
            ) * image_height

            x2 = (
                x_center + width / 2
            ) * image_width

            y2 = (
                y_center + height / 2
            ) * image_height

        # ----------------------------------------------------
        # YOLO SEGMENTATION POLYGON
        # class x1 y1 x2 y2 x3 y3 ...
        # ----------------------------------------------------

        elif len(coordinates) >= 6:

            x_coordinates = coordinates[0::2]
            y_coordinates = coordinates[1::2]

            x1 = min(x_coordinates) * image_width
            y1 = min(y_coordinates) * image_height

            x2 = max(x_coordinates) * image_width
            y2 = max(y_coordinates) * image_height

        else:
            continue

        ground_truth.append({
            "class_id": class_id,
            "box": [
                x1,
                y1,
                x2,
                y2
            ]
        })

    return ground_truth


# ============================================================
# PROCESS IMAGES
# ============================================================

image_paths = sorted(
    list(IMAGE_DIR.glob("*.jpg"))
    + list(IMAGE_DIR.glob("*.jpeg"))
    + list(IMAGE_DIR.glob("*.png"))
)

print(
    f"\nTotal images found: {len(image_paths)}"
)


# ============================================================
# STATISTICS
# ============================================================

total_gt = 0
total_tp = 0
total_fp = 0
total_fn = 0

class_stats = {}

for class_id, class_name in model.names.items():

    class_stats[class_id] = {
        "name": class_name,
        "gt": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0
    }


# ============================================================
# CSV RESULTS
# ============================================================

csv_rows = []


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for image_index, image_path in enumerate(
    image_paths,
    start=1
):

    print(
        f"\n[{image_index}/{len(image_paths)}] "
        f"{image_path.name}"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print("  Could not read image.")
        continue

    image_height, image_width = image.shape[:2]

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    label_path = (
        LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    ground_truth = read_ground_truth(
        label_path,
        image_width,
        image_height
    )

    # --------------------------------------------------------
    # MODEL PREDICTIONS
    # --------------------------------------------------------

    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=CONF_THRESHOLD,
        device="cpu",
        verbose=False
    )

    result = results[0]

    predictions = []

    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            coordinates = box.xyxy[0].tolist()

            predictions.append({
                "class_id": class_id,
                "confidence": confidence,
                "box": coordinates
            })

    # --------------------------------------------------------
    # UPDATE GROUND TRUTH COUNTS
    # --------------------------------------------------------

    total_gt += len(ground_truth)

    for gt in ground_truth:

        class_id = gt["class_id"]

        if class_id in class_stats:
            class_stats[class_id]["gt"] += 1

    # --------------------------------------------------------
    # MATCH PREDICTIONS WITH GROUND TRUTH
    # --------------------------------------------------------

    matched_gt = set()

    image_tp = 0
    image_fp = 0
    image_fn = 0

    image_visual = image.copy()

    # Draw ground-truth boxes first
    for gt_index, gt in enumerate(
        ground_truth
    ):

        x1, y1, x2, y2 = map(
            int,
            gt["box"]
        )

        cv2.rectangle(
            image_visual,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        class_name = model.names[
            gt["class_id"]
        ]

        cv2.putText(
            image_visual,
            f"GT: {class_name}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )

    # Match predictions
    for prediction in predictions:

        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt in enumerate(
            ground_truth
        ):

            if gt_index in matched_gt:
                continue

            if (
                gt["class_id"]
                != prediction["class_id"]
            ):
                continue

            iou = calculate_box_iou(
                prediction["box"],
                gt["box"]
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        x1, y1, x2, y2 = map(
            int,
            prediction["box"]
        )

        class_name = model.names[
            prediction["class_id"]
        ]

        # ----------------------------------------------------
        # TRUE POSITIVE
        # ----------------------------------------------------

        if (
            best_gt_index is not None
            and best_iou >= IOU_THRESHOLD
        ):

            matched_gt.add(
                best_gt_index
            )

            total_tp += 1
            image_tp += 1

            class_stats[
                prediction["class_id"]
            ]["tp"] += 1

            cv2.rectangle(
                image_visual,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image_visual,
                f"TP: {class_name} "
                f"{prediction['confidence']:.2f} "
                f"IoU:{best_iou:.2f}",
                (x1, min(
                    image_height - 10,
                    y2 + 20
                )),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            prediction_type = "TRUE_POSITIVE"

        # ----------------------------------------------------
        # FALSE POSITIVE
        # ----------------------------------------------------

        else:

            total_fp += 1
            image_fp += 1

            class_stats[
                prediction["class_id"]
            ]["fp"] += 1

            cv2.rectangle(
                image_visual,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                image_visual,
                f"FP: {class_name} "
                f"{prediction['confidence']:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

            prediction_type = "FALSE_POSITIVE"

        csv_rows.append([
            image_path.name,
            prediction_type,
            class_name,
            round(
                prediction["confidence"],
                4
            ),
            round(best_iou, 4),
            round(prediction["box"][0], 2),
            round(prediction["box"][1], 2),
            round(prediction["box"][2], 2),
            round(prediction["box"][3], 2)
        ])

    # --------------------------------------------------------
    # FALSE NEGATIVES
    # --------------------------------------------------------

    for gt_index, gt in enumerate(
        ground_truth
    ):

        if gt_index in matched_gt:
            continue

        total_fn += 1
        image_fn += 1

        class_stats[
            gt["class_id"]
        ]["fn"] += 1

        x1, y1, x2, y2 = map(
            int,
            gt["box"]
        )

        class_name = model.names[
            gt["class_id"]
        ]

        cv2.rectangle(
            image_visual,
            (x1, y1),
            (x2, y2),
            (0, 165, 255),
            3
        )

        cv2.putText(
            image_visual,
            f"FN: {class_name}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2
        )

    # --------------------------------------------------------
    # IMAGE CATEGORY
    # --------------------------------------------------------

    if image_fp > 0 and image_fn > 0:
        image_type = "FALSE_POSITIVE_AND_FALSE_NEGATIVE"

    elif image_fp > 0:
        image_type = "FALSE_POSITIVE"

    elif image_fn > 0:
        image_type = "FALSE_NEGATIVE"

    else:
        image_type = "CORRECT"

    # Save image only if there is an error
    if (
        image_fp > 0
        or image_fn > 0
    ):

        output_image_path = (
            OUTPUT_DIR
            / f"{image_type}_{image_path.name}"
        )

        cv2.imwrite(
            str(output_image_path),
            image_visual
        )

    print(
        f"  GT={len(ground_truth)} "
        f"Pred={len(predictions)} "
        f"TP={image_tp} "
        f"FP={image_fp} "
        f"FN={image_fn}"
    )


# ============================================================
# FINAL METRICS
# ============================================================

precision = (
    total_tp
    / (total_tp + total_fp)
    if (total_tp + total_fp) > 0
    else 0
)

recall = (
    total_tp
    / (total_tp + total_fn)
    if (total_tp + total_fn) > 0
    else 0
)

f1 = (
    2 * precision * recall
    / (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

print(
    f"Ground Truth Objects : {total_gt}"
)

print(
    f"True Positives       : {total_tp}"
)

print(
    f"False Positives      : {total_fp}"
)

print(
    f"False Negatives      : {total_fn}"
)

print(
    f"Precision            : {precision:.4f}"
)

print(
    f"Recall               : {recall:.4f}"
)

print(
    f"F1 Score             : {f1:.4f}"
)


# ============================================================
# CLASS-WISE RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("CLASS-WISE RESULTS")
print("=" * 60)

for class_id, stats in class_stats.items():

    class_precision = (
        stats["tp"]
        / (stats["tp"] + stats["fp"])
        if (
            stats["tp"]
            + stats["fp"]
        ) > 0
        else 0
    )

    class_recall = (
        stats["tp"]
        / (stats["tp"] + stats["fn"])
        if (
            stats["tp"]
            + stats["fn"]
        ) > 0
        else 0
    )

    print(
        f"\n{stats['name']}"
    )

    print(
        f"  GT : {stats['gt']}"
    )

    print(
        f"  TP : {stats['tp']}"
    )

    print(
        f"  FP : {stats['fp']}"
    )

    print(
        f"  FN : {stats['fn']}"
    )

    print(
        f"  Precision : {class_precision:.4f}"
    )

    print(
        f"  Recall    : {class_recall:.4f}"
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "prediction_results.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "image",
        "prediction_type",
        "class",
        "confidence",
        "iou",
        "x1",
        "y1",
        "x2",
        "y2"
    ])

    writer.writerows(
        csv_rows
    )


# ============================================================
# SUMMARY FILE
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "summary.txt"
)

with open(
    summary_path,
    "w"
) as file:

    file.write(
        "HAZARDOUS WASTE DETECTION "
        "ERROR ANALYSIS\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"Ground Truth Objects : {total_gt}\n"
    )

    file.write(
        f"True Positives       : {total_tp}\n"
    )

    file.write(
        f"False Positives      : {total_fp}\n"
    )

    file.write(
        f"False Negatives      : {total_fn}\n"
    )

    file.write(
        f"Precision            : {precision:.4f}\n"
    )

    file.write(
        f"Recall               : {recall:.4f}\n"
    )

    file.write(
        f"F1 Score             : {f1:.4f}\n"
    )

    file.write(
        f"\nIoU Threshold        : {IOU_THRESHOLD}\n"
    )

    file.write(
        f"Confidence Threshold : {CONF_THRESHOLD}\n"
    )


print("\n")
print("=" * 60)
print("ERROR ANALYSIS COMPLETE")
print("=" * 60)

print(
    f"\nResults saved to:"
)

print(
    OUTPUT_DIR
)

print(
    f"\nCSV:"
)

print(
    csv_path
)

print(
    f"\nSummary:"
)

print(
    summary_path
)
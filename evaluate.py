from ultralytics import YOLO
from pathlib import Path
import yaml


# ============================================================
# 1. PATHS
# ============================================================

MODEL_PATH = "F:/Proitbridge study course/Hazardous waste detection/Study materials/VP/New folder/Code/runs/detect/Hazardous_Waste_Detection/yolov8n_run-2/weights/best.pt"
DATA_YAML = "F:/Proitbridge study course/Hazardous waste detection/Study materials/VP/New folder/Code/data.yaml"


# ============================================================
# 2. CHECK FILES
# ============================================================

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Please check the path to best.pt"
    )

if not Path(DATA_YAML).exists():
    raise FileNotFoundError(
        f"Dataset YAML not found: {DATA_YAML}"
    )


# ============================================================
# 3. LOAD DATASET CONFIGURATION
# ============================================================

with open(DATA_YAML, "r") as file:
    data_config = yaml.safe_load(file)

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Classes : {data_config.get('names')}")
print(f"Number of classes : {len(data_config.get('names', []))}")

print("=" * 60)


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# 5. DETERMINE EVALUATION SPLIT
# ============================================================

if "test" in data_config:
    evaluation_split = "test"
else:
    evaluation_split = "val"

print(f"\nEvaluation split: {evaluation_split}")


# ============================================================
# 6. RUN MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("STARTING MODEL EVALUATION")
print("=" * 60)

results = model.val(
    data=DATA_YAML,
    split=evaluation_split,

    # Image size
    imgsz=640,

    # Batch size
    batch=16,

    # Save evaluation plots
    plots=True,

    # Save predictions in JSON format
    save_json=True,

    # Evaluation output directory
    project="runs/evaluation",
    name="hazardous_waste_model"
)


# ============================================================
# 7. PRINT OVERALL METRICS
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)


# ------------------------------------------------------------
# BOX METRICS
# ------------------------------------------------------------

if hasattr(results, "box"):

    print("\n--- Detection / Bounding Box Metrics ---")

    print(f"Precision     : {results.box.mp:.4f}")
    print(f"Recall        : {results.box.mr:.4f}")
    print(f"mAP@50        : {results.box.map50:.4f}")
    print(f"mAP@50-95     : {results.box.map:.4f}")


# ------------------------------------------------------------
# SEGMENTATION METRICS
# ------------------------------------------------------------

if hasattr(results, "seg") and results.seg is not None:

    print("\n--- Segmentation Metrics ---")

    print(f"Precision     : {results.seg.mp:.4f}")
    print(f"Recall        : {results.seg.mr:.4f}")
    print(f"mAP@50        : {results.seg.map50:.4f}")
    print(f"mAP@50-95     : {results.seg.map:.4f}")


# ============================================================
# 8. CLASS-WISE RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CLASS-WISE RESULTS")
print("=" * 60)

class_names = data_config.get("names", {})

if hasattr(results, "seg") and results.seg is not None:

    print("\nSegmentation class-wise mAP@50-95:")

    for class_id, map_value in enumerate(results.seg.maps):

        if isinstance(class_names, dict):
            class_name = class_names.get(class_id, str(class_id))
        else:
            class_name = class_names[class_id]

        print(
            f"{class_id:>3} | "
            f"{class_name:<25} | "
            f"mAP50-95: {map_value:.4f}"
        )

elif hasattr(results, "box"):

    print("\nDetection class-wise mAP@50-95:")

    for class_id, map_value in enumerate(results.box.maps):

        if isinstance(class_names, dict):
            class_name = class_names.get(class_id, str(class_id))
        else:
            class_name = class_names[class_id]

        print(
            f"{class_id:>3} | "
            f"{class_name:<25} | "
            f"mAP50-95: {map_value:.4f}"
        )


# ============================================================
# 9. OUTPUT LOCATION
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print("\nEvaluation results saved to:")

print(
    "runs/evaluation/hazardous_waste_model/"
)

print("\nCheck this folder for:")
print("  - Confusion Matrix")
print("  - Normalized Confusion Matrix")
print("  - Precision-Recall Curve")
print("  - F1 Curve")
print("  - Precision Curve")
print("  - Recall Curve")
print("  - Evaluation plots")
from ultralytics import YOLO
from pathlib import Path
import torch
from dotenv import load_dotenv
import os   

load_dotenv()


# ============================================================
# MODEL CONFIGURATION
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
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.5"))



# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


# ============================================================
# DEVICE
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print(f"Model path: {MODEL_PATH}")
print(f"Device: {DEVICE}")
print("Loading YOLO model...")


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print("inference.py is ready.")
print(f"Image size: {IMAGE_SIZE}")
print(f"Confidence threshold: {CONF_THRESHOLD}")

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(source):

    results = model.predict(
        source=source,
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        device=DEVICE,
        verbose=False
    )

    return results
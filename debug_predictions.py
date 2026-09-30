from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "runs"
    / "detect"
    / "Hazardous_Waste_Detection"
    / "yolov8n_run-2"
    / "weights"
    / "best.pt"
)

model = YOLO(str(MODEL_PATH))

print("Model loaded:")
print(MODEL_PATH)

# Find the first test image
test_images = list(
    Path("Datasets").rglob("test/images/*")
)

print("\nTest images found:", len(test_images))

if len(test_images) == 0:
    print("ERROR: No test images found.")
    exit()

image_path = test_images[0]

print("\nTesting image:")
print(image_path)

results = model.predict(
    source=str(image_path),
    imgsz=640,
    conf=0.5,
    device="cpu",
    verbose=False
)

result = results[0]

print("\n--- MODEL OUTPUT ---")

print("Boxes object:", result.boxes)

if result.boxes is not None:
    print("Number of boxes:", len(result.boxes))

    for i, box in enumerate(result.boxes):
        print(
            f"Prediction {i}: "
            f"class={int(box.cls[0])}, "
            f"confidence={float(box.conf[0]):.4f}, "
            f"box={box.xyxy[0].tolist()}"
        )

else:
    print("NO BOXES FOUND")

print("\nMasks object:", result.masks)
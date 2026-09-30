from ultralytics import YOLO

MODEL_PATH = "F:/Proitbridge study course/Hazardous waste detection/Study materials/VP/New folder/Code/runs/detect/Hazardous_Waste_Detection/yolov8n_run-2/weights/best.pt"

IMAGE_PATH = r"F:\Proitbridge study course\Hazardous waste detection\Study materials\VP\New folder\Code\Datasets\test\images\Screenshot-2025-02-17-050725_png_png.rf.ed30326b3f0ef8b642213b0d84d91932.jpg"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    conf=0.25,
    save=True,
    verbose=True
)

result = results[0]

print("\n" + "=" * 60)
print("PREDICTION CHECK")
print("=" * 60)

print("Boxes:", result.boxes)

if result.boxes is not None:
    print("Number of boxes:", len(result.boxes))

if result.masks is not None:
    print("Masks found:", result.masks.data.shape)
else:
    print("Masks: None")

if result.boxes is not None and len(result.boxes) > 0:

    print("\nPredicted classes:")

    print(
        result.boxes.cls
        .cpu()
        .numpy()
    )

    print("\nConfidence scores:")

    print(
        result.boxes.conf
        .cpu()
        .numpy()
    )

print("\nPrediction image saved in the runs folder.")
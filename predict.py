from pathlib import Path
import cv2

from inference import predict


# ============================================================
# INPUT IMAGE
# ============================================================

INPUT_IMAGE = Path(
    r"F:\Proitbridge study course\Hazardous waste detection\Study materials\VP\New folder\Code\Datasets\test\images\Screenshot-2025-02-16-224941_png_png.rf.8175364e307e62d957b22e03f09d379d.jpg"
)

OUTPUT_DIR = Path("inference_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# CHECK IMAGE
# ============================================================

if not INPUT_IMAGE.exists():
    raise FileNotFoundError(
        f"Image not found:\n{INPUT_IMAGE}"
    )


print(f"Running inference on: {INPUT_IMAGE}")


# ============================================================
# PREDICTION
# ============================================================

results = predict(str(INPUT_IMAGE))

result = results[0]


# ============================================================
# PRINT RESULTS
# ============================================================

if result.boxes is not None and len(result.boxes) > 0:

    print("\nDetections:")

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = result.names[class_id]

        print(
            f"Class: {class_name} | "
            f"Confidence: {confidence:.2f}"
        )

else:

    print("\nNo objects detected.")


# ============================================================
# SAVE RESULT
# ============================================================

annotated_image = result.plot()

output_path = OUTPUT_DIR / "prediction.jpg"

cv2.imwrite(
    str(output_path),
    annotated_image
)

print(f"\nResult saved to:")
print(output_path)
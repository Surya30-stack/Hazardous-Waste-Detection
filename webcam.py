import cv2
import time

from inference import predict


# ============================================================
# WEBCAM CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

WINDOW_NAME = "Hazardous Waste Detection"

# Minimum confidence for displaying predictions.
# This matches the value in inference.py.
CONF_THRESHOLD = 0.5


# ============================================================
# START WEBCAM
# ============================================================

print("Starting webcam...")

cap = cv2.VideoCapture(CAMERA_INDEX)

# For Windows, DirectShow can sometimes help camera detection.
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open the webcam.\n"
        "Check whether your camera is connected and available."
    )

print("Webcam started successfully.")
print("Press Q to quit.")


# ============================================================
# FPS VARIABLES
# ============================================================

prev_time = time.time()


# ============================================================
# WEBCAM LOOP
# ============================================================

while True:

    # Read one frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from webcam.")
        break

    # --------------------------------------------------------
    # RUN YOLO INFERENCE
    # --------------------------------------------------------

    results = predict(frame)

    result = results[0]

    # --------------------------------------------------------
    # DRAW DETECTIONS / SEGMENTATION
    # --------------------------------------------------------

    annotated_frame = result.plot()

    # --------------------------------------------------------
    # CALCULATE FPS
    # --------------------------------------------------------

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # --------------------------------------------------------
    # DISPLAY FRAME
    # --------------------------------------------------------

    cv2.imshow(
        WINDOW_NAME,
        annotated_frame
    )

    # --------------------------------------------------------
    # QUIT WITH Q
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# RELEASE RESOURCES
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Webcam stopped.")
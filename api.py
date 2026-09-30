from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from inference import predict,model


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Hazardous Waste Detection API",
    description="YOLO-based hazardous waste detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Hazardous Waste Detection API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # CHECK FILE TYPE
    # --------------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    # --------------------------------------------------------
    # READ UPLOADED FILE
    # --------------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # --------------------------------------------------------
    # CONVERT IMAGE BYTES TO OPENCV IMAGE
    # --------------------------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded image."
        )

    # --------------------------------------------------------
    # RUN YOLO INFERENCE
    # --------------------------------------------------------

    results = predict(image)

    result = results[0]

    # --------------------------------------------------------
    # EXTRACT DETECTIONS
    # --------------------------------------------------------

    detections = []

    if result.boxes is not None and len(result.boxes) > 0:

        for index, box in enumerate(result.boxes):

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = result.names[class_id]

            # Bounding box coordinates
            x1, y1, x2, y2 = map(
                float,
                box.xyxy[0].tolist()
            )

            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),

                "bounding_box": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            }

            # ------------------------------------------------
            # SEGMENTATION
            # ------------------------------------------------

            if (
                result.masks is not None
                and index < len(result.masks.xy)
            ):
                polygon = result.masks.xy[index].tolist()

                detection["segmentation"] = polygon

            detections.append(detection)

    # --------------------------------------------------------
    # CREATE ANNOTATED IMAGE
    # --------------------------------------------------------

    annotated_image = result.plot()

    success, encoded_image = cv2.imencode(
        ".jpg",
        annotated_image
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Could not encode annotated image."
        )

    # --------------------------------------------------------
    # CONVERT IMAGE TO BASE64
    # --------------------------------------------------------

    image_base64 = base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")

    # --------------------------------------------------------
    # RETURN RESPONSE
    # --------------------------------------------------------

    return {
        "status": "success",
        "filename": file.filename,
        "detection_count": len(detections),
        "detections": detections,
        "annotated_image_base64": image_base64
    }
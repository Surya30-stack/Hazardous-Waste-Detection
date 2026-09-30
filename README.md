# Hazardous Waste Detection

An end-to-end computer vision application for detecting **cylinders** and **shock absorbers** in scrap-yard environments using **YOLOv8n** object detection.

The project covers the complete workflow from dataset preparation and model training to model evaluation, real-time inference, API integration, frontend development, and Docker-based local deployment.

> **Note:** This project was developed using a client-provided dataset under NDA. Therefore, the original dataset, annotations, trained model weights, and other confidential client information are not included in this repository.

---

## Project Overview

In scrap-yard environments, identifying different discarded metal components manually can be time-consuming and inconsistent.

This project explores a computer vision-based approach for automatically detecting two types of components:

* **Cylinder**
* **Shock Absorber**

The trained object detection model identifies the location of these components in an image and returns their predicted class and confidence score along with bounding boxes.

The model is exposed through a **FastAPI backend** and connected to a **React frontend**, allowing users to upload images and view the detection results through a web interface.

The complete application is containerized using **Docker and Docker Compose** for easier local deployment.

---

## Objectives

The main objectives of the project were to:

* Build an object detection model for identifying scrap-yard components.
* Prepare and validate an annotated image dataset.
* Handle class imbalance during model development.
* Fine-tune a pretrained YOLOv8 model using transfer learning.
* Evaluate the model using object detection metrics.
* Develop an image and webcam inference pipeline.
* Expose the model through a REST API.
* Build a web-based frontend for interacting with the model.
* Containerize the backend and frontend using Docker.

---

## Dataset

The project used a **client-provided dataset** containing approximately **607 annotated images**.

### Dataset characteristics

| Property             | Details                  |
| -------------------- | ------------------------ |
| Source               | Client-provided          |
| Approx. images       | 607                      |
| Number of classes    | 2                        |
| Classes              | Cylinder, Shock Absorber |
| Annotation type      | Bounding-box annotations |
| Class distribution   | Imbalanced               |
| Dataset availability | Not publicly included    |

Because the dataset is covered by an NDA, the original images, annotation files, dataset configuration, and client-specific information have been excluded from this repository.

---

## Data Preparation

The dataset preparation workflow included:

1. Checking the image and annotation files.
2. Validating image-label pairs.
3. Checking class IDs and annotations.
4. Identifying the classes present in the dataset.
5. Creating training, validation, and test splits.
6. Examining class distribution.
7. Addressing class imbalance through augmentation during model development.

The dataset was kept outside the public GitHub repository to protect client confidentiality.

---

## Model

### YOLOv8n

The project uses **YOLOv8n (YOLOv8 Nano)** for object detection.

YOLO was selected because it provides a practical balance between:

* Detection accuracy
* Inference speed
* Model size
* Ease of deployment

A pretrained YOLOv8n model was fine-tuned on the project dataset using **transfer learning**.

### Detection Pipeline

```text
Input Image / Webcam
        ↓
Image Preprocessing
        ↓
YOLOv8n Model
        ↓
Object Detection
        ↓
Bounding Boxes
        ↓
Class + Confidence Score
        ↓
Annotated Output
```

---

## Model Evaluation

The model was evaluated using commonly used object detection metrics.

| Metric    | Validation Result |
| --------- | ----------------: |
| Precision |         **0.795** |
| Recall    |         **0.775** |
| mAP@50    |         **0.842** |
| mAP@50–95 |         **0.641** |

### Metric Interpretation

**Precision** measures how many of the objects predicted by the model were actually correct.

**Recall** measures how many of the actual objects in the images were successfully detected.

**mAP@50** evaluates detection performance when a predicted bounding box is considered correct at an IoU threshold of 0.50.

**mAP@50–95** evaluates the model across multiple IoU thresholds from 0.50 to 0.95, providing a stricter measure of both object detection and bounding-box localization quality.

> These results are from the validation evaluation used during project development and should not be interpreted as production performance.

---

## Inference

The trained model was integrated into an inference pipeline capable of processing:

* Image files
* Webcam input

For each detected object, the system provides:

* Detected class
* Confidence score
* Bounding box coordinates
* Annotated image output

Example workflow:

```text
Upload Image
     ↓
FastAPI API
     ↓
YOLOv8 Inference
     ↓
Detection Results
     ↓
Annotated Image
     ↓
React Frontend
```

---

## Application Architecture

The project consists of three primary components.

### 1. Machine Learning Layer

Responsible for:

* Loading the trained YOLOv8n model
* Image preprocessing
* Object detection
* Confidence thresholding
* Generating bounding boxes and predictions

### 2. Backend — FastAPI

The FastAPI backend provides REST endpoints for interacting with the model.

Main endpoints include:

```text
GET  /
GET  /health
POST /predict
```

The `/predict` endpoint:

1. Receives an uploaded image.
2. Validates the image format.
3. Decodes the image using OpenCV.
4. Runs YOLO inference.
5. Extracts detected classes, confidence scores, and bounding boxes.
6. Generates an annotated image.
7. Returns the results to the frontend.

### 3. Frontend — React

The frontend provides a web interface through which users can interact with the detection system.

It communicates with the FastAPI backend and displays the model's detection results.

---

## Technology Stack

### Machine Learning

* Python
* YOLOv8
* Ultralytics
* PyTorch
* OpenCV
* NumPy
* Pandas
* Scikit-learn

### Backend

* FastAPI
* Uvicorn
* Python

### Frontend

* React
* Vite
* JavaScript
* HTML
* CSS

### Deployment

* Docker
* Docker Compose

### Development

* VS Code
* Git
* GitHub

---

##  Project Structure

```text
Hazardous-Waste-Detection/
│
├── Frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   ├── package-lock.json
│   └── Dockerfile
│
├── api.py
├── inference.py
├── predict.py
├── webcam.py
│
├── dataset_check.py
├── debug_label.py
├── debug_predictions.py
├── check_predictions.py
├── evaluate.py
├── error_analysis.py
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .env.example
└── README.md
```

### Files used during development

| File                   | Purpose                            |
| ---------------------- | ---------------------------------- |
| `dataset_check.py`     | Dataset and annotation validation  |
| `debug_label.py`       | Label/annotation debugging         |
| `debug_predictions.py` | Prediction debugging               |
| `check_predictions.py` | Prediction inspection              |
| `evaluate.py`          | Model evaluation                   |
| `error_analysis.py`    | Detection error analysis           |
| `inference.py`         | Loads model and performs inference |
| `predict.py`           | Image-based prediction             |
| `webcam.py`            | Webcam-based inference             |
| `api.py`               | FastAPI backend                    |
| `Frontend/`            | React frontend                     |

---

## Docker Deployment

The application is containerized to simplify running the backend and frontend together.

Docker Compose is used to manage the application services.

### Architecture

```text
                Docker Compose
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    FastAPI Backend          React Frontend
       Container                Container
          │                       │
          │                       │
          └─────── API ───────────┘
                    │
                    ▼
              YOLOv8n Model
```

The application was tested using Docker for **local deployment**.

---

## Running the Application

### Prerequisites

Install:

* Python 3.x
* Node.js
* Docker Desktop
* Git

Because the project uses a client-provided dataset and trained model covered by NDA, those files must be obtained separately through the appropriate authorized source.

---

### Option 1 — Run with Docker Compose

Clone the repository:

```bash
git clone https://github.com/Surya30-stack/Hazardous-Waste-Detection.git
cd Hazardous-Waste-Detection
```

Start the application:

```bash
docker compose up --build
```

After the containers start, the frontend can be accessed at:

```text
http://localhost:5173
```

The FastAPI backend runs at:

```text
http://localhost:8000
```

API documentation is available at:

```text
http://localhost:8000/docs
```

Stop the application with:

```bash
docker compose down
```

---

### Option 2 — Run the Backend Locally

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI application:

```bash
uvicorn api:app --reload
```

The backend will be available at:

```text
http://localhost:8000
```

---

### Run the Frontend Locally

Move into the frontend directory:

```bash
cd Frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:5173
```

---

##  NDA & Data Privacy

This repository intentionally does **not** contain:

* Client-provided images
* Original annotation files
* `data.yaml`
* Trained model weights
* Client-identifying information
* Prediction outputs generated from confidential data
* Local environment/secrets

The following types of files are excluded through `.gitignore`:


Datasets/
runs/
inference_results/
*.pt
*.pth
*.onnx
.env


The `.env.example` file is included only as a configuration template and does not contain secrets.

---

## Configuration

The project supports configurable inference parameters through environment variables.

Example:


IMAGE_SIZE=640
CONF_THRESHOLD=0.5
MODEL_PATH=models/best.pt


> The trained model itself is not included in this public repository because of the project NDA.

---

## Key Technical Work

The project involved the following machine learning and software engineering tasks:

* Dataset validation and annotation checking
* Train/validation/test data preparation
* Exploratory analysis of object classes
* Handling class imbalance
* Image augmentation
* Transfer learning
* YOLOv8 model fine-tuning
* Object detection evaluation
* Precision and recall analysis
* mAP-based evaluation
* Prediction debugging
* Error analysis
* Image inference
* Webcam inference
* FastAPI REST API development
* React frontend integration
* Docker containerization
* Docker Compose orchestration
* Git/GitHub version control

---

## Project Challenges

### 1. Limited Dataset

The dataset contained approximately 607 images, which required careful dataset validation and model evaluation.

### 2. Class Imbalance

The two object classes were not equally represented. Class distribution was therefore considered during model development and augmentation.

### 3. Confidential Client Data

Because the project was developed using client-provided data under NDA, the public repository had to be structured without exposing the original dataset or model artifacts.

### 4. End-to-End Integration

The project extended beyond model training by integrating the trained model with:


YOLOv8
   ↓
Python Inference
   ↓
FastAPI
   ↓
React
   ↓
Docker Compose


---

## Limitations

* The training dataset is relatively small.
* The dataset has class imbalance.
* The model is trained for only two object categories.
* The trained model weights are not publicly distributed.
* Validation performance may differ from performance on unseen real-world environments.
* The application is demonstrated as a local deployment rather than a production cloud deployment.
* Detection performance can be affected by lighting, object orientation, occlusion, image quality, and background conditions.

---

## Possible Future Improvements

Potential future development areas include:

* Increasing the size and diversity of the training dataset.
* Improving class balance.
* Collecting more challenging real-world images.
* Performing systematic hyperparameter tuning.
* Comparing YOLOv8 model variants.
* Further optimizing inference speed.
* Adding confidence-based filtering and improved error handling.
* Adding more object categories if additional labeled data becomes available.
* Deploying the application to a cloud environment.
* Adding monitoring and logging for production usage.
* Integrating detection results with downstream scrap-yard inventory or sorting workflows.

---

## What I Learned

This project provided practical experience across the complete machine learning lifecycle:


Problem Understanding
        ↓
Data Preparation
        ↓
Data Validation
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Inference
        ↓
Backend Development
        ↓
Frontend Integration
        ↓
Docker Deployment


It also provided hands-on experience with taking a computer vision model beyond experimentation and integrating it into a usable application.

---

## Author

**Surya S.**

Data Science / Machine Learning

GitHub:
https://github.com/Surya30-stack

---

## Disclaimer

This repository contains a portfolio representation of a client project.

The original dataset, annotations, trained model weights, and confidential client information are intentionally excluded due to NDA restrictions.

The code and documentation provided here are intended to demonstrate the technical workflow and implementation approach without exposing proprietary information.

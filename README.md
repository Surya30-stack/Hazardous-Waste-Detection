# Hazardous Waste Detection

YOLOv8-based object detection system for identifying discarded metal components in scrap-yard environments.

## Project Overview

This project uses computer vision to detect two types of metal components:

- Cylinder
- Shock Absorber

The system was developed as a client project and includes model training, evaluation, image/webcam inference, a FastAPI backend, React frontend, and Docker-based deployment.

## Tech Stack

- Python
- YOLOv8n
- Ultralytics
- OpenCV
- FastAPI
- React
- Docker
- Docker Compose

## Model

The YOLOv8n model was fine-tuned using transfer learning and evaluated using:

- Precision: 0.795
- Recall: 0.775
- mAP@50: 0.842
- mAP@50-95: 0.641

## Application Flow


Image / Webcam
      ↓
YOLOv8n Detection
      ↓
Bounding Boxes + Classes + Confidence
      ↓
FastAPI Backend
      ↓
React Frontend


## Deployment

The application can be run locally using Docker and Docker Compose.

## Confidentiality

This project was developed using a client-provided dataset.

Due to NDA and confidentiality restrictions, the following are intentionally excluded from this repository:

* Client dataset and images
* Annotation files
* Trained model weights
* Client-specific files and outputs

The repository contains the non-sensitive application code and project structure for demonstration purposes.





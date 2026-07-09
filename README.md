# Shixin’an Intelligent Health Monitoring System

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A multimodal health monitoring system based on computer vision and artificial intelligence, integrating human pose detection, pharmaceutical OCR recognition, and non‑contact heart rate monitoring. Supports real‑time video stream analysis and web‑based visualisation.

## 📋 Table of Contents

- [Project Introduction](#project-introduction)
- [Core Features](#core-features)
- [Technical Architecture](#technical-architecture)
- [Project Structure](#project-structure)
- [Technical Documentation](#technical-documentation)
- [Project Demo](#project-demo)

## 🎯 Project Introduction

Zhenghe Health Monitoring System is an innovative intelligent health management platform that combines advanced deep learning algorithms with traditional medical knowledge to provide users with comprehensive health monitoring services. The system adopts a modular design and supports three independent working modes:

- **Pose Monitoring Mode**: Real‑time detection of human keypoints to identify abnormal postures such as falls.
- **Drug Recognition Mode**: OCR recognition of text on drug packaging, matched against a medication advice database.
- **Heart Rate Monitoring Mode**: Non‑contact heart rate measurement based on rPPG technology.

## ✨ Core Features

### 1. Human Pose Detection and Fall Warning

- High‑precision human keypoint detection based on the YOLOv11 Pose model.
- Supports simultaneous detection of multiple people, with real‑time calculation of the human aspect ratio to determine fall status.
- BPU hardware‑accelerated inference for significantly improved processing speed.
- Visual annotation of keypoints, bounding boxes, and status labels.

### 2. Intelligent Drug OCR Recognition

- Integrates the EasyOCR engine, supporting mixed Chinese‑English text recognition.
- Built‑in database of 1000+ common drugs (med‑txt.txt).
- Intelligent filtering of irrelevant text for precise matching of drug names and usage recommendations.
- Asynchronous queue processing to avoid blocking the main thread.

### 3. Non‑contact Heart Rate Monitoring

- Based on remote photoplethysmography (rPPG) technology.
- POS (Plane Orthogonal to Skin) algorithm for motion artifact resistance.
- Dual frequency‑domain and time‑domain verification for improved accuracy.
- Signal Quality Index (SQI) to assess data reliability.
- Haar cascade classifier for face detection and ROI extraction.

### 4. Real‑time Web Monitoring Interface

- Modern Glassmorphism UI design.
- Responsive layout, supporting access from multiple devices.
- Real‑time video streaming (MJPEG format).
- Dynamic data updates and mode switching.

## 🏗️ Technical Architecture

### Core Technology Stack

| Module | Technology Selection | Description |
|--------|----------------------|-------------|
| **Backend Framework** | Flask 2.0+ | Lightweight web server |
| **Pose Detection** | YOLOv11 Pose | High‑performance human keypoint detection |
| **OCR Engine** | EasyOCR | Multi‑language text recognition |
| **Heart Rate Algorithm** | rPPG (POS) | Remote photoplethysmography |
| **Face Detection** | OpenCV Haar Cascade | Fast face localisation |
| **Hardware Acceleration** | Hobot BPU | Horizon Robotics processor |
| **Frontend UI** | HTML5 + CSS3 | Modern web technologies |

## 📋 Technical Documentation
- [Technical Documentation](https://github.com/houyujia520/-RDK-X5-/blob/main/docs/%E6%8A%80%E6%9C%AF%E6%96%87%E6%A1%A3.pdf)

- ## 🎥 Project Demo

- [Nodehub Technical Showcase](https://developer.d-robotics.cc/nodehubdetail/2075056640813641729) — Contains complete project introduction, feature demonstrations, and technical details.

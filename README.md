# 🍕 Pizza Store Scooper Violation Detection System

A **microservices-based Computer Vision system** for monitoring hygiene compliance in food preparation environments.
The system automatically detects whether workers are using a **scooper** when handling ingredients, and flags **bare-hand violations** in real time.

---

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Docker%20Compose-2496ED.svg)
![YOLO](https://img.shields.io/badge/Model-YOLO12-orange.svg)
![RabbitMQ](https://img.shields.io/badge/Broker-RabbitMQ-orange.svg)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue.svg)

---

## 📖 Overview

Maintaining hygiene standards in busy pizza stores is critical but difficult to enforce manually.
This system automates supervision by:

1. **Ingesting** live or recorded video streams
2. **Detecting** hands, scoopers, and ingredient interaction zones (ROIs)
3. **Verifying** whether a scooper is used correctly
4. **Flagging and logging** hygiene violations
5. **Streaming** annotated video and statistics to a web dashboard

The system is designed with **scalability, modularity, and real-time performance** in mind.

---

## 🏗️ System Architecture

The system follows a **decoupled microservices architecture**:

* **Frame Reader Service**
  Reads video input (RTSP or file) and publishes frames to the message broker.

* **RabbitMQ**
  Acts as a message broker between services using frame and result queues.

* **Detector Service**
  Runs YOLO12 inference, object tracking, and violation logic.

* **PostgreSQL Database**
  Stores violation events, timestamps, and snapshot paths.

* **Streamer Service**
  Streams annotated video and exposes REST APIs for statistics.

---

## 🧭 System Diagram

<p align="center">
  <img src="assets/system-diagram.jpg" alt="Pizza Store Scooper Detection Architecture" width="850"/>
</p>

<p align="center">
  <em>End-to-End Microservices Architecture for Real-Time Scooper Violation Detection</em>
</p>

---

## 📂 Project Structure

```text
Pizza-Store-Scooper-Violation-Detection/
├── .env.example
├── .gitignore
├── docker-compose.yaml
├── README.md
│
├── detector/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── detector.py
│   ├── violation_logic.py
│   ├── database.py
│   └── model/
│       ├── yolo12m-v2.pt
│       └── yolo12m-v3.pt
│
├── frame_reader/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── reader.py
│
├── streamer/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── streamer.py
│
├── videos/
│   ├── .gitkeep
│   └── Sah w b3dha ghalt (2).mp4
│
└── violation_images/
    ├── violation_frame_407_1766858296.jpg
    ├── violation_frame_407_1766861636.jpg
    ├── violation_frame_503_1766858304.jpg
    └── violation_frame_503_1766861646.jpg
```

---

## 🚀 Getting Started

### Prerequisites

* **Docker Desktop** (with Docker Compose)
* **NVIDIA GPU** (optional, recommended for real-time inference)

---

### Installation

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/Moaz0009/Pizza-Store-Scooper-Violation-Detection.git
cd Pizza-Store-Scooper-Violation-Detection
```

---

#### 2️⃣ Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set values such as database credentials and video source.

---

#### 3️⃣ Build and run the system

```bash
docker-compose up --build
```

All services will start automatically.

---

#### 4️⃣ Access the dashboard

* **Video Stream:**
  `http://localhost:5000/video_feed`

* **Statistics API:**
  `http://localhost:5000/stats`

---

## ⚙️ Configuration (`.env`)

| Variable            | Description                           | Example                  |
| ------------------- | ------------------------------------- | ------------------------ |
| `VIDEO_SOURCE`      | Video file path or RTSP URL           | `/app/videos/sample.mp4` |
| `ROI_POINTS`        | Polygon defining ingredient container | `454,388,514,397,...`    |
| `MODEL_PATH`        | YOLO model path inside container      | `model/yolo12m-v2.pt`    |
| `POSTGRES_PASSWORD` | Database password                     | `secret`                 |

---

## 🧠 Violation Logic

The system uses a **state-machine-based logic** with ROI awareness:

* 🟢 **SAFE**
  Hand Leaves ROI while holding a scooper.

* 🔴 **VIOLATION**
  Hand leaves ROI without a scooper.

* ⏱️ **Linger Rule**
  If a hand separates from the scooper inside the ROI for more than **1 second**, a violation is triggered when leaving.

Snapshots are saved automatically and linked to database records.

---

## 💾 Database Schema

Violations are stored in a PostgreSQL table:

| Column           | Type      | Description            |
| ---------------- | --------- | ---------------------- |
| `id`             | SERIAL    | Unique violation ID    |
| `timestamp`      | TIMESTAMP | Time of violation      |
| `frame_id`       | INTEGER   | Frame number           |
| `violation_type` | TEXT      | Description            |
| `image_path`     | TEXT      | Path to saved snapshot |

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Model:** Ultralytics YOLO12 (Medium)
* **Computer Vision:** OpenCV
* **Message Broker:** RabbitMQ (Pika)
* **Database:** PostgreSQL 13
* **Streaming:** Flask (MJPEG)
* **Containerization:** Docker & Docker Compose

---


## 📜 License

This project is licensed under the **MIT License**.
See `LICENSE` for details.

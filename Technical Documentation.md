# 📘 Technical Documentation: Pizza Store Scooper Violation Detection System

## 1. Executive Summary
This system is a microservices-based Computer Vision solution designed to enforce hygiene protocols in a pizza store environment. It utilizes Deep Learning (**YOLO12**) to monitor specific Regions of Interest (ROIs), ensuring that workers use a scooper when handling ingredients like proteins. The system processes video feeds in real-time, detects violations (bare hand contact), and logs evidence for auditing.

---

## 2. System Architecture

The solution adheres to a decoupled **Microservices Architecture** to ensure scalability and maintainability.

### 2.1 Microservices Breakdown
The system consists of five distinct services:

1.  **Frame Reader Service (Ingestion)**
    * **Function:** Connects to the video source (RTSP Camera or File).
    * **Responsibility:** Reads raw frames, resizes them for optimization, and publishes them to the message broker.
    * **Output:** Serialized raw frames sent to the `frames` queue.

2.  **Message Broker (RabbitMQ)**
    * **Function:** Asynchronous communication bus.
    * **Responsibility:** Decouples the ingestion rate from the processing rate, handling buffering and stream management.
    * **Queues:**
        * `frames`: Raw video data awaiting processing.
        * `results`: Processed frames with bounding boxes and metadata.

3.  **Detection Service (Core Logic)**
    * **Function:** AI Inference and Business Logic.
    * **Model:** **YOLO12 Medium**.
    * **Responsibility:**
        * Detects objects: `Hand`, `Person`, `Pizza`, `Scooper`.
        * Tracks objects across frames using IDs.
        * Executes the Violation State Machine (see Section 3).
        * Persists violation events to the Database.
    * **Output:** Annotated frames sent to `results` queue.

4.  **Streaming Service (Presentation)**
    * **Function:** Web Server (Flask).
    * **Responsibility:** Consumes processed frames and serves them via HTTP (MJPEG) to the frontend. Exposes REST endpoints for statistics.

5.  **Database Service**
    * **Function:** Persistent storage.
    * **Technology:** PostgreSQL.
    * **Data:** Stores violation timestamps, frame IDs, and file paths to evidence images.

---

## 3. Violation Detection Logic

The core logic is implemented as a state machine within the **Detection Service**. It monitors the interaction between **Hands**, **Scoopers**, and **ROIs** (Protein Containers).

### 3.1 State Machine Definitions
* **ROI (Region of Interest):** A polygon defined around the ingredient container (e.g., protein cargo).
* **Safe Condition:** A `Hand` is detected intersecting with a `Scooper` (Distance < Threshold).
* **Violation Condition:** A `Hand` enters the `ROI` **without** intersecting a `Scooper`.

### 3.2 Logic Flow
For every frame:
1.  **Object Tracking:** Update positions of all Hands and Scoopers.
2.  **Credit Check:** If `Distance(Hand, Scooper) < Threshold`, grant **"Scooper Credit"** (Worker is holding the tool).
3.  **ROI Check:**
    * **IF** Hand is inside ROI **AND** has "Scooper Credit":
        * ✅ **Status:** COMPLIANT.
    * **IF** Hand is inside ROI **AND** does NOT have "Scooper Credit":
        * **Linger Timer Starts.**
        * If the hand remains for > 1 second (approx. 30 frames):
        * ❌ **Status:** VIOLATION DETECTED.
4.  **Logging:** If a violation triggers, increment the counter and save the frame to disk.

---

## 4. Database Schema

The system uses a relational database to store violation events.

**Table Name:** `violations`

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `id` | `SERIAL PRIMARY KEY` | Unique identifier for the event. |
| `timestamp` | `TIMESTAMP` | Date and time of the violation. |
| `frame_id` | `INTEGER` | The specific video frame number. |
| `violation_type` | `TEXT` | Type of event (Default: "Bare Hand Contact"). |
| `image_path` | `TEXT` | Local file path to the saved evidence image (`/app/violation_images/...`). |
| `bounding_boxes`| `TEXT` (JSON) | JSON array of bounding boxes involved in the event. |

---

## 5. API Reference

The **Streaming Service** provides the following endpoints for the Frontend UI.

### 5.1 Video Stream
* **Endpoint:** `GET /video_feed`
* **Format:** `multipart/x-mixed-replace; boundary=frame`
* **Description:** Returns a continuous MJPEG stream of processed video frames with bounding boxes drawn.

### 5.2 Statistics
* **Endpoint:** `GET /stats`
* **Format:** `JSON`
* **Response Example:**
    ```json
    {
      "v": 5,
    }
    ```

---

## 6. Deployment Configuration

The system is containerized using **Docker Compose** to ensure consistency across environments.

* **Environment Variables (`.env`):**
    * Confidential data (DB passwords) and dynamic settings (Camera URLs, ROI Coordinates) are injected at runtime.
* **Volumes:**
    * `./violation_images`: Maps the container's storage to the host machine to preserve evidence photos.
    * `postgres_data`: Persists database records even if containers are restarted.

---

## 7. Future Improvements
* **Alerting:** Integrate Email or SMS notifications when a violation occurs.

import pika
import pickle
import cv2
import numpy as np
import time
import sys
import os
from ultralytics import YOLO
from violation_logic import ViolationTracker
from database import init_db, log_violation

# --- STRICT CONFIGURATION (No Defaults) ---
# This will crash with KeyError if variables are missing in .env
RABBITMQ_HOST = os.environ["RABBITMQ_HOST"]
MODEL_PATH = os.environ["MODEL_PATH"]
IMAGE_SAVE_DIR = os.environ["IMAGE_SAVE_DIR"]
ROI_STR = os.environ["ROI_POINTS"]

# Strict ROI Parsing
coords = list(map(int, ROI_STR.split(',')))
ROI_POINTS = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

print(f"🚀 Initializing Detector Service...", flush=True)

# 1. INITIALIZE DATABASE
init_db()

# 2. LOAD MODEL (No Try/Except Safety)
print(f"Loading model from: {MODEL_PATH}", flush=True)
model = YOLO(MODEL_PATH) 
print(f"✅ Model loaded successfully.", flush=True)

# Initialize the logic with the parsed ROI
tracker = ViolationTracker(roi_points=ROI_POINTS)
last_violation_count = 0 

def connect_to_rabbitmq():
    while True:
        try:
            print(f"⏳ Connecting to RabbitMQ at {RABBITMQ_HOST}...", flush=True)
            connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
            print("✅ Connected!", flush=True)
            return connection
        except pika.exceptions.AMQPConnectionError:
            time.sleep(5)
        # Note: General Exception safety kept here only for network retry (standard practice),
        # but let me know if you want this removed too.

connection = connect_to_rabbitmq()
channel = connection.channel()
channel.queue_declare(queue="frames")
channel.queue_declare(queue="results")

def callback(ch, method, properties, body):
    global last_violation_count
    
    try:
        data = pickle.loads(body)
        
        np_arr = np.frombuffer(data["frame_data"], np.uint8)
        full_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        frame_id = data["frame_id"]

        results = model.track(full_frame, persist=True, verbose=False)[0]
        
        detections = []
        current_frame_bboxes = [] 
        current_frame_labels = [] 
        
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy()

            for box, cls, track_id in zip(boxes, classes, track_ids):
                class_name = model.names[int(cls)] 
                bx1, by1, bx2, by2 = map(int, box)
                bbox_tuple = (bx1, by1, bx2, by2)

                detections.append({
                    "class": class_name,
                    "bbox": bbox_tuple,
                    "id": int(track_id)
                })
                current_frame_bboxes.append(bbox_tuple)
                current_frame_labels.append(class_name)

        current_violations, annotated_frame = tracker.process(full_frame, detections)

        if current_violations > last_violation_count:
            diff = current_violations - last_violation_count
            timestamp = int(time.time())
            filename = f"violation_frame_{frame_id}_{timestamp}.jpg"
            save_path = os.path.join(IMAGE_SAVE_DIR, filename)
            
            # Ensure directory exists (Strict: assumes permission exists)
            if not os.path.exists(IMAGE_SAVE_DIR):
                os.makedirs(IMAGE_SAVE_DIR)
                
            cv2.imwrite(save_path, annotated_frame)
            
            for _ in range(diff):
                log_violation(
                    frame_id=frame_id,
                    violation_type="Bare Hand Contact in ROI",
                    image_path=save_path,
                    bboxes=current_frame_bboxes,
                    labels=current_frame_labels
                )
            
            last_violation_count = current_violations

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        channel.basic_publish(
            exchange="",
            routing_key="results",
            body=pickle.dumps({
                "frame_id": frame_id,
                "frame_data": buffer.tobytes(),
                "violations": current_violations
            })
        )
        
    except Exception as e:
        print(f"❌ Error during processing: {e}", flush=True)

channel.basic_consume(queue="frames", on_message_callback=callback, auto_ack=True)
print("👀 Detector started...", flush=True)
channel.start_consuming()
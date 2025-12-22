import pika
import pickle
import cv2
import numpy as np
import time
import sys
from ultralytics import YOLO
from violation_logic import ViolationTracker

# --- CONFIGURATION ---
MODEL_PATH = "model/yolo12m-v2.pt" 

# CROP CALCULATED FROM YOUR ROI POINTSs
# Min X=322, Max X=719 -> Crop 250 to 800
# Min Y=112, Max Y=739 -> Crop 50 to 800
CROP_X1, CROP_Y1 = 250, 50  
CROP_X2, CROP_Y2 = 750, 750 

print(f"🚀 Initializing Detector Service...", flush=True)

try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}", flush=True)
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load model at {MODEL_PATH}", flush=True)
    sys.exit(1)

tracker = ViolationTracker()

def connect_to_rabbitmq():
    while True:
        try:
            print("⏳ Connecting to RabbitMQ...", flush=True)
            connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
            print("✅ Connected!", flush=True)
            return connection
        except pika.exceptions.AMQPConnectionError:
            time.sleep(5)
        except Exception as e:
            time.sleep(5)

connection = connect_to_rabbitmq()
channel = connection.channel()
channel.queue_declare(queue="frames")
channel.queue_declare(queue="results")

def callback(ch, method, properties, body):
    try:
        data = pickle.loads(body)
        
        # Decode FULL ORIGINAL frame
        np_arr = np.frombuffer(data["frame_data"], np.uint8)
        full_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        frame_id = data["frame_id"]

        # --- 1. SMART CROP ---
        h, w, _ = full_frame.shape
        # Safety checks to ensure crop is within image
        x1 = max(0, CROP_X1)
        y1 = max(0, CROP_Y1)
        x2 = min(w, CROP_X2)
        y2 = min(h, CROP_Y2)
        
        cropped_frame = full_frame[y1:y2, x1:x2]

        # --- 2. DETECT ON CROP ---
        # This makes small objects 2x-3x larger for the model!
        results = model.track(cropped_frame, persist=True, verbose=False)[0]
        
        detections = []
        
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy()

            for box, cls, track_id in zip(boxes, classes, track_ids):
                class_name = model.names[int(cls)] 
                
                # --- 3. FIX COORDINATES ---
                # The model sees (10,10). We must add the Crop Offset (250, 50)
                # so the box appears correctly on the full video.
                bx1, by1, bx2, by2 = map(int, box)
                
                detections.append({
                    "class": class_name,
                    "bbox": (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                    "id": int(track_id)
                })

        # --- 4. LOGIC & DRAWING ---
        # Logic works on the FULL frame, using corrected coordinates
        violations, annotated_frame = tracker.process(full_frame, detections)

        # Optional: Draw the Blue "Focus Area" so you know what the model sees
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
        cv2.putText(annotated_frame, "AI FOCUS AREA", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # Publish
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        channel.basic_publish(
            exchange="",
            routing_key="results",
            body=pickle.dumps({
                "frame_id": frame_id,
                "frame_data": buffer.tobytes(),
                "violations": violations
            })
        )
        
    except Exception as e:
        print(f"❌ Error: {e}", flush=True)

channel.basic_consume(queue="frames", on_message_callback=callback, auto_ack=True)
print("👀 Detector started with CROP FOCUS...", flush=True)
channel.start_consuming()
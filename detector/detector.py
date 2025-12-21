import pika
import pickle
import cv2
import numpy as np
from ultralytics import YOLO
from violation_logic import ViolationTracker

# Load model (YOLOv8/v12 supports tracking natively)
model = YOLO("model/yolov12.pt")
tracker = ViolationTracker()

connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
channel = connection.channel()
channel.queue_declare(queue="frames")
channel.queue_declare(queue="results")

def callback(ch, method, properties, body):
    data = pickle.loads(body)
    
    # FIX: Decode the JPEG back to a Numpy Frame
    np_arr = np.frombuffer(data["frame_data"], np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame_id = data["frame_id"]

    # FIX: Use model.track() instead of model() to get object IDs
    # persist=True keeps IDs stable across frames
    results = model.track(frame, persist=True, verbose=False)[0]
    
    detections = []
    
    if results.boxes.id is not None:
        # Extract boxes, classes, AND Tracking IDs
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy()

        for box, cls, track_id in zip(boxes, classes, track_ids):
            class_name = model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            detections.append({
                "class": class_name,
                "bbox": (x1, y1, x2, y2),
                "id": int(track_id)
            })

    # Pass the tracked detections to logic
    violations, annotated_frame = tracker.process(frame, detections)

    # Re-encode frame for the next service (Streaming)
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

channel.basic_consume(queue="frames", on_message_callback=callback, auto_ack=True)
print("Detector service started...")
channel.start_consuming()
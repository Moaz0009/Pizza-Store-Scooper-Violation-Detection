import pika
import pickle
import cv2
from ultralytics import YOLO
from violation_logic import ViolationTracker

model = YOLO("model/yolov12.pt")
tracker = ViolationTracker()

connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
channel = connection.channel()

channel.queue_declare(queue="frames")
channel.queue_declare(queue="results")

def callback(ch, method, properties, body):
    data = pickle.loads(body)
    frame = data["frame"]
    frame_id = data["frame_id"]

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        cls = model.names[int(box.cls)]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((cls, (x1, y1, x2, y2)))

    violations, annotated = tracker.process(frame, detections)

    channel.basic_publish(
        exchange="",
        routing_key="results",
        body=pickle.dumps({
            "frame_id": frame_id,
            "frame": annotated,
            "violations": violations
        })
    )

channel.basic_consume(
    queue="frames",
    on_message_callback=callback,
    auto_ack=True
)

print("Detector service started...")
channel.start_consuming()

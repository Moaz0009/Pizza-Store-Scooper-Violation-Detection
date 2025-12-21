import cv2
import pika
import pickle
import time
import os

VIDEO_PATH = os.getenv("VIDEO_PATH", "video.mp4")

connection = pika.BlockingConnection(pika.ConnectionParameters(host="rabbitmq"))
channel = connection.channel()
channel.queue_declare(queue="frames")

cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0

print("Frame Reader started...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = buffer.tobytes()

    payload = {
        "frame_id": frame_id,
        "timestamp": time.time(),
        "frame_data": jpg_as_text, 
        "shape": frame.shape 
    }

    channel.basic_publish(
        exchange="",
        routing_key="frames",
        body=pickle.dumps(payload)
    )

    frame_id += 1
    time.sleep(0.03) 

cap.release()
print("Frame Reader finished.")
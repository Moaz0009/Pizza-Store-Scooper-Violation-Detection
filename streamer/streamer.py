import cv2
import pika
import pickle
import threading
import numpy as np
import time
from flask import Flask, Response, render_template_string

app = Flask(__name__)

latest_frame = None
current_violations = 0
lock = threading.Lock()

# Initial blank frame
blank = np.zeros((480, 640, 3), np.uint8)
cv2.putText(blank, "Waiting for stream...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
_, buf = cv2.imencode('.jpg', blank)
latest_frame = buf.tobytes()

def consume_results():
    """Reads processed frames from Detector via RabbitMQ"""
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
            channel = connection.channel()
            channel.queue_declare(queue="results")

            def callback(ch, method, properties, body):
                global latest_frame, current_violations
                data = pickle.loads(body)
                
                # Expects 'frame_data' (JPEG bytes) from Detector
                if "frame_data" in data:
                    with lock:
                        latest_frame = data["frame_data"]
                        current_violations = data.get("violations", 0)

            channel.basic_consume(queue="results", on_message_callback=callback, auto_ack=True)
            print("Streamer connected.")
            channel.start_consuming()
        except Exception:
            time.sleep(2) # Retry connection

def generate_frames():
    global latest_frame
    while True:
        with lock:
            if latest_frame is None: continue
            frame_data = latest_frame
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.04)

@app.route('/')
def index():
    return render_template_string('''
        <html>
        <body style="background:#222; color:white; text-align:center; font-family:sans-serif;">
            <h1>EagleVision Monitor</h1>
            <h2 style="color:red">Violations: <span id="cnt">0</span></h2>
            <img src="/video_feed" style="border:2px solid #555; width:80%;">
            <script>
                setInterval(() => {
                    fetch('/stats').then(r=>r.json()).then(d => document.getElementById('cnt').innerText = d.v);
                }, 1000);
            </script>
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return {"v": current_violations}

if __name__ == '__main__':
    t = threading.Thread(target=consume_results)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5000)
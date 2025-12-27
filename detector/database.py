import psycopg2
import datetime
import os
import time
import json

# Configuration from Environment Variables
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_NAME = os.getenv("POSTGRES_DB", "violations_db")
DB_USER = os.getenv("POSTGRES_USER", "admin")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "secret")

def get_connection():
    """Retries connection to Postgres until successful."""
    while True:
        try:
            conn = psycopg2.connect(
                host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS
            )
            return conn
        except psycopg2.OperationalError:
            print("⏳ Database not ready, waiting...", flush=True)
            time.sleep(2)

def init_db():
    """Creates the violations table with columns for images and metadata."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP,
            frame_id INTEGER,
            violation_type TEXT,
            image_path TEXT,
            bounding_boxes TEXT,  -- Stores JSON string
            detected_labels TEXT  -- Stores JSON string
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()
    print("✅ PostgreSQL Database initialized.", flush=True)

def log_violation(frame_id, violation_type, image_path, bboxes, labels):
    """Saves the violation row."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        timestamp = datetime.datetime.now()
        
        # Serialize lists to JSON strings for storage
        bbox_str = json.dumps(bboxes)
        labels_str = json.dumps(labels)
        
        cur.execute('''
            INSERT INTO violations 
            (timestamp, frame_id, violation_type, image_path, bounding_boxes, detected_labels)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (timestamp, frame_id, violation_type, image_path, bbox_str, labels_str))
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"💾 Violation saved: Frame {frame_id}", flush=True)
    except Exception as e:
        print(f"❌ Database Error: {e}", flush=True)
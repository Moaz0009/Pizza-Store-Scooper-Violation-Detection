import cv2
import numpy as np
import math
import time
from collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort


class ViolationTracker:
    def __init__(self):
        # ==============================
        # DEEPSORT
        # ==============================
        self.tracker = DeepSort(
            max_age=30,
            n_init=2,
            max_iou_distance=0.7,
            embedder="mobilenet",
            half=True,
            bgr=True
        )

        # ==============================
        # STATE
        # ==============================
        self.tracks = {}   # deep_sort_id -> state
        self.total_violations = 0
        self.frame_counter = 0

        # ==============================
        # SETTINGS
        # ==============================
        self.ROI = [(454, 388), (514, 397), (526, 351), (466, 340)]
        self.HOLDING_THRESHOLD = 150
        self.GRACE_FRAMES = 10

        # ==============================
        # EVENTS
        # ==============================
        self.event_log = []
        self.start_time = time.time()

        print("✅ Violation Tracker (DeepSORT) Initialized", flush=True)

    # ==============================
    # UTILS
    # ==============================
    def get_center(self, bbox):
        x, y, w, h = bbox
        return int(x + w / 2), int(y + h / 2)

    def get_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def is_inside_roi(self, point):
        return cv2.pointPolygonTest(
            cv2.convexHull(np.array(self.ROI, np.int32)),
            point,
            False
        ) >= 0

    def log_event(self, event, track_id, pos):
        self.event_log.append({
            "time_sec": round(time.time() - self.start_time, 2),
            "frame": self.frame_counter,
            "track_id": track_id,
            "event": event,
            "x": pos[0],
            "y": pos[1]
        })

    # ==============================
    # MAIN PROCESS
    # ==============================
    def process(self, frame, detections):
        self.frame_counter += 1

        # --------------------------------
        # PREPARE DEEPSORT INPUT (HANDS ONLY)
        # --------------------------------
        ds_inputs = []
        scoopers = []

        for d in detections:
            if d["class"].lower() == "hand":
                x1, y1, x2, y2 = d["bbox"]
                ds_inputs.append((
                    [x1, y1, x2 - x1, y2 - y1],
                    d.get("confidence", 0.9),
                    "hand"
                ))
            elif d["class"].lower() == "scooper":
                scoopers.append(d)

        # --------------------------------
        # UPDATE TRACKER
        # --------------------------------
        tracks = self.tracker.update_tracks(ds_inputs, frame=frame)

        # --------------------------------
        # PROCESS TRACKS
        # --------------------------------
        for t in tracks:
            if not t.is_confirmed():
                continue

            track_id = t.track_id
            bbox = t.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            center = self.get_center([x1, y1, x2 - x1, y2 - y1])

            # INIT TRACK STATE
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    "history": deque(maxlen=5),
                    "state": "outside",
                    "grace_timer": 0,
                    "entered_roi": False,
                    "exit_pos": None
                }

            data = self.tracks[track_id]
            data["history"].append(center)

            if len(data["history"]) < 2:
                continue

            prev_pos = data["history"][-2]
            curr_pos = center

            prev_in = self.is_inside_roi(prev_pos)
            curr_in = self.is_inside_roi(curr_pos)

            # --------------------------------
            # VISUAL
            # --------------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"H{track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --------------------------------
            # ENTER ROI
            # --------------------------------
            if not prev_in and curr_in:
                data["state"] = "inside"
                data["entered_roi"] = True
                data["grace_timer"] = 0
                self.log_event("hand_enter_roi", track_id, curr_pos)

            if not data["entered_roi"]:
                continue

            # --------------------------------
            # EXIT ROI → CHECK
            # --------------------------------
            if prev_in and not curr_in and data["state"] == "inside":
                data["state"] = "checking"
                data["grace_timer"] = self.GRACE_FRAMES
                data["exit_pos"] = curr_pos
                self.log_event("hand_exit_roi", track_id, curr_pos)

            # --------------------------------
            # CHECK SCOOPER (SAME HAND ONLY)
            # --------------------------------
            if data["state"] == "checking":
                check_pos = data["exit_pos"]

                min_dist = min(
                    [self.get_distance(
                        check_pos,
                        self.get_center([
                            s["bbox"][0],
                            s["bbox"][1],
                            s["bbox"][2] - s["bbox"][0],
                            s["bbox"][3] - s["bbox"][1]
                        ])
                    ) for s in scoopers] or [1e9]
                )

                if min_dist < self.HOLDING_THRESHOLD:
                    data["state"] = "outside"
                    data["entered_roi"] = False
                    data["exit_pos"] = None
                    self.log_event("safe_exit_scooper_used", track_id, check_pos)

                    cv2.putText(frame, "SAFE", check_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                else:
                    data["grace_timer"] -= 1
                    cv2.putText(frame, f"Checking {data['grace_timer']}",
                                check_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                    if data["grace_timer"] <= 0:
                        self.total_violations += 1
                        data["state"] = "violation"
                        data["entered_roi"] = False
                        data["exit_pos"] = None
                        self.log_event("violation_confirmed", track_id, check_pos)

            # --------------------------------
            # VIOLATION VISUAL
            # --------------------------------
            if data["state"] == "violation":
                cv2.rectangle(frame,
                              (curr_pos[0] - 50, curr_pos[1] - 50),
                              (curr_pos[0] + 50, curr_pos[1] + 50),
                              (0, 0, 255), 4)
                cv2.putText(frame, "VIOLATION",
                            (curr_pos[0] - 60, curr_pos[1] - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # --------------------------------
        # DRAW ROI
        # --------------------------------
        cv2.polylines(frame, [np.array(self.ROI, np.int32)],
                      True, (0, 255, 255), 2)

        return self.total_violations, frame

import cv2
import numpy as np
import math

class ViolationTracker:
    def __init__(self):
        # SYSTEM STATE
        self.state = "IDLE"
        self.verification_timer = 0
        self.total_violations = 0
        self.frame_counter = 0
        
        # LOGIC FLAGS
        self.scooper_memory = False   # The "Credit"
        self.bare_hand_timer = 0      # Counts frames where hand is separated from scooper
        
        # CONSTANTS
        self.STABILITY_THRESHOLD = 10 
        self.empty_frames_count = 0 
        
        # SETTINGS
        self.HOLDING_THRESHOLD = 30   # Pixel distance to count as "Holding"
        self.SEPARATION_LIMIT = 30    # If separated for ~1 sec (20 frames), it's a violation
        self.VERIFY_DURATION = 20
        
        # --- ROI ---
        raw_roi = [(454, 388), (514, 397), (526, 351), (466, 340)]
        OFFSET_X = 0  
        OFFSET_Y = 0
        
        self.ROI = []
        for (x, y) in raw_roi:
            self.ROI.append((x + OFFSET_X, y + OFFSET_Y))

        print(f"✅ LOGIC STARTED. Separation Limit: {self.SEPARATION_LIMIT} frames.", flush=True)

    def get_center(self, bbox):
        return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)

    def is_inside(self, point):
        return cv2.pointPolygonTest(
            cv2.convexHull(np.array(self.ROI, np.int32)), 
            point, False
        ) >= 0

    def get_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def check_holding_scooper(self, hands, scoopers):
        if not hands or not scoopers:
            return False, 9999
            
        min_dist = 9999
        for hand in hands:
            h_c = self.get_center(hand["bbox"])
            for scooper in scoopers:
                s_c = self.get_center(scooper["bbox"])
                dist = self.get_distance(h_c, s_c)
                if dist < min_dist: min_dist = dist
        
        is_holding = (min_dist < self.HOLDING_THRESHOLD)
        return is_holding, min_dist

    def process(self, frame, detections):
        self.frame_counter += 1
        
        hands = [d for d in detections if d["class"].lower() == "hand"]
        scoopers = [d for d in detections if d["class"].lower() == "scooper"]
        
        # --- 1. MEMORY UPDATE (CREDIT) ---
        is_holding_global, _ = self.check_holding_scooper(hands, scoopers)
        if is_holding_global:
            self.scooper_memory = True
            self.bare_hand_timer = 0 

        # --- VISUALS: SCOOPERS (Blue) ---
        if self.scooper_memory:
            cv2.putText(frame, "CREDIT: ACTIVE", (50, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        for scooper in scoopers:
            x1, y1, x2, y2 = scooper["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Scooper", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # --- VISUALS: HANDS (Red/Green) ---
        hands_inside_roi = 0
        for hand in hands:
            center = self.get_center(hand["bbox"])
            x1, y1, x2, y2 = hand["bbox"]
            
            if self.is_inside(center):
                hands_inside_roi += 1
                # Red Box & Dot for Inside
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1) 
            else:
                # Green Box & Dot for Outside
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 255, 0), -1) 

        # ==========================
        # STATE MACHINE
        # ==========================

        if self.state == "IDLE":
            if hands_inside_roi > 0:
                print(f"➡️ [ENTRY] Monitoring...", flush=True)
                self.state = "OCCUPIED"
                self.empty_frames_count = 0
                self.bare_hand_timer = 0

        elif self.state == "OCCUPIED":
            
            # --- ACTIVE SEPARATION CHECK ---
            if hands_inside_roi > 0 and len(scoopers) > 0:
                is_holding_now, dist = self.check_holding_scooper(hands, scoopers)
                
                if not is_holding_now:
                    # Seen separated -> Linger Timer
                    self.bare_hand_timer += 1
                    cv2.putText(frame, f"BARE HAND TIMER: {self.bare_hand_timer}", (50, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if self.bare_hand_timer > self.SEPARATION_LIMIT:
                        print("   ❌ [VIOLATION] Linger detected! Hand separated from scooper for too long.", flush=True)
                        self.total_violations += 1
                        self.state = "VIOLATION_SHOW"
                        self.verification_timer = 30
                        self.scooper_memory = False # Revoke credit
                else:
                    # Seen together -> Reset Timer
                    self.bare_hand_timer = 0
            
            # Handle Exit Logic
            if hands_inside_roi == 0:
                self.empty_frames_count += 1
                if self.empty_frames_count > self.STABILITY_THRESHOLD:
                    self.state = "VERIFYING"
                    self.verification_timer = self.VERIFY_DURATION
            else:
                self.empty_frames_count = 0

        elif self.state == "VERIFYING":
            self.verification_timer -= 1
            is_holding_now, _ = self.check_holding_scooper(hands, scoopers)
            
            cv2.putText(frame, f"VERIFYING... {self.verification_timer}", (50, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            if is_holding_now:
                self.state = "IDLE"
                cv2.putText(frame, "SAFE (HOLDING)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            elif self.verification_timer <= 0:
                if self.scooper_memory:
                    print("   ✅ [SAFE] Returned (Credit Used).", flush=True)
                    self.scooper_memory = False 
                    self.state = "IDLE"
                    cv2.putText(frame, "SAFE (RETURNED)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                else:
                    self.total_violations += 1
                    print("   ❌ [VIOLATION] No scooper credit.", flush=True)
                    self.state = "VIOLATION_SHOW" 
                    self.verification_timer = 30 

        elif self.state == "VIOLATION_SHOW":
            self.verification_timer -= 1
            h, w, _ = frame.shape
            cv2.rectangle(frame, (20, 20), (w-20, h-20), (0, 0, 255), 10)
            cv2.putText(frame, "VIOLATION!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            if self.verification_timer <= 0:
                self.state = "IDLE"

        roi_color = (0, 255, 255)
        if self.state == "OCCUPIED": roi_color = (0, 0, 255)
        elif self.state == "VERIFYING": roi_color = (0, 165, 255)
        cv2.polylines(frame, [np.array(self.ROI, np.int32)], True, roi_color, 2)
        
        return self.total_violations, frame
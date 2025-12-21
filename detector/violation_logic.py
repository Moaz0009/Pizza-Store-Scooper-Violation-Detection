import cv2

class ViolationTracker:
    def __init__(self):
        # FIX: State is now a dictionary: { hand_id: { "in_roi": bool, "used_scooper": bool } }
        self.hand_states = {}
        self.total_violations = 0
        
        # Define ROI (Example: Protein Container)
        # In production, this should be configurable
        self.roi = [(391, 688), (448, 696), (523, 273), (484, 269)]

    def point_in_roi(self, bbox):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return cv2.pointPolygonTest(
            cv2.convexHull(cv2.UMat.from_array(self.roi).get()), 
            (cx, cy), False
        ) >= 0

    def intersects(self, box_a, box_b):
        # Standard intersection check
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

    def process(self, frame, detections):
        # Separate detections by type
        hands = []
        pizzas = []
        scoopers = []

        for d in detections:
            if d["class"] == "Hand":
                hands.append(d)
            elif d["class"] == "Pizza":
                pizzas.append(d["bbox"])
            elif d["class"] == "Scooper":
                scoopers.append(d["bbox"])

        # Logic: Check each detected hand individually
        current_frame_hand_ids = set()

        for hand in hands:
            h_id = hand["id"]
            h_box = hand["bbox"]
            current_frame_hand_ids.add(h_id)

            # Initialize state for new hands
            if h_id not in self.hand_states:
                self.hand_states[h_id] = {"in_roi": False, "used_scooper": False}

            # 1. Check if Hand is in ROI (Ingredients)
            if self.point_in_roi(h_box):
                self.hand_states[h_id]["in_roi"] = True
                
                # While in ROI, check if holding scooper
                # If they hold a scooper at ANY point while in ROI, they are safe
                for s_box in scoopers:
                    if self.intersects(h_box, s_box):
                        self.hand_states[h_id]["used_scooper"] = True

            # 2. Check if Hand touches Pizza
            for p_box in pizzas:
                if self.intersects(h_box, p_box):
                    # Trigger Violation Condition:
                    # Visited ROI (True) AND Did NOT use scooper (False)
                    state = self.hand_states[h_id]
                    
                    if state["in_roi"] and not state["used_scooper"]:
                        self.total_violations += 1
                        print(f"VIOLATION DETECTED on Hand ID {h_id}!")
                        
                        # Visual Alert
                        cv2.rectangle(frame, (h_box[0], h_box[1]), (h_box[2], h_box[3]), (0, 0, 255), 3)
                        cv2.putText(frame, "VIOLATION", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        # Reset state to prevent infinite counting for same event
                        self.hand_states[h_id] = {"in_roi": False, "used_scooper": False}
                    
                    # If they touched pizza safely (or after violation), reset the cycle
                    # so they can go back to ingredients for a new task
                    elif state["in_roi"] and state["used_scooper"]:
                         self.hand_states[h_id] = {"in_roi": False, "used_scooper": False}

        # Visualization: Draw ROI
        cv2.polylines(frame, [np.array(self.roi, np.int32)], True, (255, 255, 0), 2)
        
        # Visualization: Draw Violation Count
        cv2.putText(frame, f"Violations: {self.total_violations}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return self.total_violations, frame
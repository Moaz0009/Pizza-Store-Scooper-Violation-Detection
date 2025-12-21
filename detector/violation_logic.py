import cv2

class ViolationTracker:
    def __init__(self):
        self.state = {
            "in_roi": False,
            "used_scooper": False
        }
        self.violations = 0

        self.roi = [(200,100),(500,100),(500,300),(200,300)]

    def point_in_roi(self, bbox):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        return cv2.pointPolygonTest(
            cv2.convexHull(
                cv2.UMat.from_array(self.roi).get()
            ), (cx, cy), False
        ) >= 0

    def intersects(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

    def process(self, frame, detections):
        hand = pizza = scooper = None

        for cls, box in detections:
            if cls == "Hand":
                hand = box
            elif cls == "Pizza":
                pizza = box
            elif cls == "Scooper":
                scooper = box

        if hand:
            if self.point_in_roi(hand):
                self.state["in_roi"] = True
                if scooper and self.intersects(hand, scooper):
                    self.state["used_scooper"] = True

            elif self.state["in_roi"] and pizza and self.intersects(hand, pizza):
                if not self.state["used_scooper"]:
                    self.violations += 1
                    cv2.putText(frame, "VIOLATION", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0,0,255), 2)
                self.state = {"in_roi": False, "used_scooper": False}

        cv2.polylines(
            frame,
            [cv2.convexHull(cv2.UMat.from_array(self.roi).get())],
            True, (255,255,0), 2
        )

        return self.violations, frame

"""
Real-time OpenCV Hand Landmark and Skeleton Visualizer for MediaPipe Hand Landmarker.

Renders:
- 21 3D hand landmarks per detected hand.
- Anatomical finger joint connections (skeleton).
- Distinct color coding and badges for Left and Right hands.
- Fingertip emphasis and high-contrast outline rendering.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


# MediaPipe 21 hand landmark connections
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm base / knuckle connections
    (5, 9), (9, 13), (13, 17),
]

FINGERTIP_INDICES: set = {4, 8, 12, 16, 20}
WRIST_INDEX: int = 0


class HandVisualizer:
    """
    Renders hand landmarks, anatomical skeleton lines, and handedness tags
    directly onto OpenCV BGR image frames.
    """

    def __init__(
        self,
        line_thickness: int = 2,
        joint_radius: int = 4,
        tip_radius: int = 6,
        show_labels: bool = True,
    ):
        self.line_thickness = line_thickness
        self.joint_radius = joint_radius
        self.tip_radius = tip_radius
        self.show_labels = show_labels

        # BGR Color schemes
        self.styles = {
            "Right": {
                "bone_color": (0, 220, 120),       # Vibrant Emerald Green
                "joint_color": (0, 165, 255),      # Bright Orange
                "tip_color": (255, 105, 180),      # Hot Pink
                "badge_bg": (20, 80, 20),
                "badge_text": (255, 255, 255),
            },
            "Left": {
                "bone_color": (255, 200, 0),       # Cyan / Sky Blue
                "joint_color": (0, 255, 255),      # Neon Yellow
                "tip_color": (0, 255, 128),        # Bright Mint
                "badge_bg": (80, 40, 20),
                "badge_text": (255, 255, 255),
            },
            "Unknown": {
                "bone_color": (200, 200, 200),
                "joint_color": (255, 255, 255),
                "tip_color": (0, 220, 255),
                "badge_bg": (40, 40, 40),
                "badge_text": (255, 255, 255),
            },
        }

    def draw_landmarks_on_frame(
        self,
        frame: np.ndarray,
        hand_result: Any,
    ) -> np.ndarray:
        """
        Draw landmarks and skeleton connections from a MediaPipe HandLandmarkerResult.
        Modifies frame in-place and returns it.
        """
        if hand_result is None or not hasattr(hand_result, "hand_landmarks") or not hand_result.hand_landmarks:
            return frame

        h, w = frame.shape[:2]

        for h_idx, landmarks in enumerate(hand_result.hand_landmarks):
            if not landmarks or len(landmarks) < 21:
                continue

            # Determine handedness and score
            side = "Unknown"
            score = 1.0
            if hasattr(hand_result, "handedness") and hand_result.handedness and h_idx < len(hand_result.handedness):
                categories = hand_result.handedness[h_idx]
                if categories and len(categories) > 0:
                    side = categories[0].category_name
                    score = float(categories[0].score)

            style = self.styles.get(side, self.styles["Unknown"])

            # Convert normalized coordinates to pixel coords
            pixel_points: List[Tuple[int, int]] = []
            for lm in landmarks:
                px = int(np.clip(lm.x * w, 0, w - 1))
                py = int(np.clip(lm.y * h, 0, h - 1))
                pixel_points.append((px, py))

            # 1. Draw skeleton bones (lines)
            for start_idx, end_idx in HAND_CONNECTIONS:
                pt1 = pixel_points[start_idx]
                pt2 = pixel_points[end_idx]
                # High-contrast background stroke
                cv2.line(frame, pt1, pt2, (10, 10, 10), self.line_thickness + 2, cv2.LINE_AA)
                # Colored skeleton bone
                cv2.line(frame, pt1, pt2, style["bone_color"], self.line_thickness, cv2.LINE_AA)

            # 2. Draw joints (circles)
            for idx, pt in enumerate(pixel_points):
                if idx in FINGERTIP_INDICES:
                    r = self.tip_radius
                    color = style["tip_color"]
                elif idx == WRIST_INDEX:
                    r = self.tip_radius + 1
                    color = (255, 255, 255)
                else:
                    r = self.joint_radius
                    color = style["joint_color"]

                # Dark border for contrast
                cv2.circle(frame, pt, r + 1, (0, 0, 0), -1, cv2.LINE_AA)
                # Inner joint dot
                cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)

            # 3. Handedness badge near wrist
            if self.show_labels:
                wrist_pt = pixel_points[WRIST_INDEX]
                label_text = f"{side} ({score:.0%})"
                (lw, lh), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

                badge_x = max(5, min(w - lw - 10, wrist_pt[0] - lw // 2))
                badge_y = max(lh + 5, min(h - 5, wrist_pt[1] + 25))

                cv2.rectangle(
                    frame,
                    (badge_x - 3, badge_y - lh - 4),
                    (badge_x + lw + 3, badge_y + baseline + 2),
                    style["badge_bg"],
                    -1,
                )
                cv2.rectangle(
                    frame,
                    (badge_x - 3, badge_y - lh - 4),
                    (badge_x + lw + 3, badge_y + baseline + 2),
                    style["bone_color"],
                    1,
                )
                cv2.putText(
                    frame,
                    label_text,
                    (badge_x, badge_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    style["badge_text"],
                    1,
                    cv2.LINE_AA,
                )

        return frame

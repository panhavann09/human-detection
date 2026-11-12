"""Detect upright, full-body people with OpenCV's built-in HOG model."""

import cv2


class HumanDetector:
    def __init__(self):
        # Initialize the model once, rather than for every video frame.
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        """Return boxes (x, y, width, height) and raw detector scores."""
        height, width = frame.shape[:2]

        # The default HOG model uses a 64-by-128 detection window.
        if width < 64 or height < 128:
            return []

        boxes, scores = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        return [
            {
                "box": tuple(int(value) for value in box),
                # This score is not a probability or percentage.
                "score": float(score),
            }
            for box, score in zip(boxes, scores)
        ]

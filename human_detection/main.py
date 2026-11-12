"""Run with: python -m human_detection.main"""

import argparse

import cv2

from .camera import open_camera
from .detector import HumanDetector


def main():
    parser = argparse.ArgumentParser(description="Detect people using a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    detector = HumanDetector()
    try:
        camera = open_camera(args.camera)
    except RuntimeError as error:
        print(error)
        return 1

    window_name = "Human Detection"
    print("Click the video window and press Q or Escape to quit.")
    print("For best results, keep your full body visible in the camera.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the camera.")
                return 1

            # Some cameras ignore the requested resolution. Limit processing size.
            height, width = frame.shape[:2]
            if width > 640:
                frame = cv2.resize(
                    frame,
                    (640, max(1, round(height * 640 / width))),
                    interpolation=cv2.INTER_AREA,
                )

            detections = detector.detect(frame)
            for detection in detections:
                x, y, width, height = detection["box"]
                cv2.rectangle(
                    frame, (x, y), (x + width, y + height), (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    "Person",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

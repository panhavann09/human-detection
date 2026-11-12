"""Open and configure the webcam."""

import cv2


def open_camera(index=0):
    """Return an open camera, or raise an error if it is unavailable."""
    camera = cv2.VideoCapture(index)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Cannot open camera {index}. Check its connection, Windows camera "
            "permissions, and whether another application is using it."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

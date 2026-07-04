#!/usr/bin/env python3
"""
Real-time object detection using YOLOv8 and webcam.
Press 'q' to quit.
"""

import cv2
from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        annotated = results.plot()

        cv2.imshow("Object Detection (YOLO)", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
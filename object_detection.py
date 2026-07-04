#!/usr/bin/env python3
"""
Real-time object detection using YOLOv8 and webcam.
Press 'q' to quit.
"""

import cv2
from ultralytics import YOLO


def list_cameras(max_to_check=10):
    available = []
    for idx in range(max_to_check):
        cap = cv2.VideoCapture(idx + cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                backend_name = cap.getBackendName()
                available.append((idx, backend_name))
            cap.release()
    return available


def select_camera(available):
    print("\nAvailable cameras:")
    for cam_id, backend in available:
        print(f"  [{cam_id}] Camera {cam_id} ({backend})")
    print(f"  [c] Cancel")

    while True:
        choice = input("\nSelect camera index: ").strip()
        if choice.lower() == "c":
            return None
        try:
            idx = int(choice)
            if idx in [cam[0] for cam in available]:
                return idx
            print(f"Camera {idx} not available.")
        except ValueError:
            print("Invalid input.")


def main():
    model = YOLO("yolo11n.pt")

    cameras = list_cameras()
    if not cameras:
        print("No cameras found.")
        return

    selected = select_camera(cameras)
    if selected is None:
        print("Cancelled.")
        return

    cap = cv2.VideoCapture(selected + cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"Error: Could not open camera {selected}")
        return

    print(f"\nCamera {selected} started. Press 'q' to quit.")

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
#!/usr/bin/env python3
"""
License plate detection + OCR + wanted-checking.

Live mode: real-time detection with bounding boxes.
Snapshot mode (SPACE): captures frame and runs deep analysis (harsher preprocessing,
full-text OCR scan on the whole image) to identify plates in detail.

wanted_plates.txt (one plate per line) defines the searched plates.
Press 'q' to quit, SPACE for snapshot.
"""

import cv2
import easyocr
import re
import os
import numpy as np
from datetime import datetime

WANTED_FILE = os.path.join(os.path.dirname(__file__), "wanted_plates.txt")
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
POSSIBLE_PATTERN = re.compile(r"^[A-Z0-9]{4,10}$")


def load_wanted_plates(path):
    plates = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                stripped = line.strip().upper()
                if stripped:
                    plates.add(stripped)
    print(f"Loaded {len(plates)} wanted plate(s)")
    return plates


def refine_text(raw):
    cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper())
    return cleaned


def plate_matches_format(text):
    return bool(POSSIBLE_PATTERN.match(text))


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
    print("  [c] Cancel")

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


def detect_plate_regions(gray, area_min=800, area_max=None):
    candidates = []

    for blur_sigma in [1.6, 2.4]:
        blurred = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
        edges = cv2.Canny(blurred, 40, 160)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / float(h) if h > 0 else 0
            area = w * h
            if not (2.0 <= ratio <= 6.5):
                continue
            if area < area_min:
                continue
            if area_max and area > area_max:
                continue
            approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
            if len(approx) < 4:
                continue
            candidates.append((area, (x, y, w, h)))

    if candidates:
        areas = [a for a, _ in candidates]
        mean_a = np.mean(areas)
        candidates = [(a, b) for a, b in candidates if a >= mean_a * 0.2]

    candidates.sort(key=lambda t: t[0], reverse=True)
    seen = set()
    unique = []
    for area, bbox in candidates:
        key = tuple(round(v / 10) * 10 for v in bbox)
        if key not in seen:
            seen.add(key)
            unique.append(bbox)
    return unique[:6]


def deep_preprocess_for_ocr(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(equalized, None, 10, 7, 21)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def ocr_region(reader, crop_bgr, deep=False):
    if deep:
        processed = deep_preprocess_for_ocr(crop_bgr)
        results = reader.readtext(processed, detail=1, paragraph=False)
    else:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray, detail=1, paragraph=False)

    all_text = []
    best_conf = 0
    best_text = ""
    for bbox, text, conf in results:
        if conf > 0.2:
            all_text.append(text)
        if conf > best_conf:
            best_conf = conf
            best_text = text

    raw = " ".join(all_text)
    clean = refine_text(raw)
    if plate_matches_format(clean):
        return clean, best_conf

    if best_text and plate_matches_format(refine_text(best_text)):
        return refine_text(best_text), best_conf

    return None, 0


def draw_result(frame, x, y, w, h, plate, is_wanted, conf):
    color = (0, 0, 255) if is_wanted else (0, 255, 0)
    status = "WANTED" if is_wanted else "OK"
    label = f"{status}: {plate} ({conf:.0%})"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x, y - th - baseline - 8), (x + tw + 8, y), color, cv2.FILLED)
    cv2.putText(frame, label, (x + 4, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def snapshot_mode(frame, reader, wanted):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    print(f"\n📍 Snapshot saved: {path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    regions = detect_plate_regions(gray, area_min=500)

    if not regions:
        full_text = reader.readtext(frame, detail=0, paragraph=False)
        raw = " ".join(full_text)
        clean = refine_text(raw)
        if clean and plate_matches_format(clean):
            print(f"🔍 Full-frame OCR found: {clean}")
            regions = [(0, 0, frame.shape[1], frame.shape[0])]

    if not regions:
        print("❌ No plate-like region found in snapshot.")
        return frame

    annotated = frame.copy()
    found = []
    for x, y, w, h in regions:
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        plate, conf = ocr_region(reader, crop, deep=True)
        if plate:
            is_wanted = plate in wanted
            draw_result(annotated, x, y, w, h, plate, is_wanted, conf)
            found.append((plate, is_wanted, conf))

    if not found:
        print("❌ Could not read any plate text from regions.")
        return annotated

    print("\n══════════ SNAPSHOT RESULTS ══════════")
    for plate, is_wanted, conf in sorted(found, key=lambda t: t[2], reverse=True):
        status = "🚨 WANTED" if is_wanted else "✅ OK"
        print(f"  {status}  |  {plate}  |  confidence: {conf:.1%}")
    print("═══════════════════════════════════════\n")

    result_path = os.path.join(SNAPSHOT_DIR, f"result_{timestamp}.jpg")
    cv2.imwrite(result_path, annotated)
    print(f"Annotated result saved: {result_path}")

    cv2.imshow("Snapshot Result", annotated)
    print("Press any key to return to live feed...")
    cv2.waitKey(0)
    cv2.destroyWindow("Snapshot Result")

    return frame


def main():
    reader = easyocr.Reader(["en"], gpu=True)
    wanted = load_wanted_plates(WANTED_FILE)

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

    print(f"\nCamera {selected} started.")
    print("Controls:  SPACE = snapshot analysis  |  q = quit")
    print("Live detection running...\n")

    live_skip = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            snapshot_mode(frame, reader, wanted)
            continue

        if key == ord("q"):
            break

        live_skip += 1
        if live_skip % 3 != 0:
            cv2.imshow("License Plate Detection (LIVE)", frame)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        regions = detect_plate_regions(gray, area_min=1000)

        for x, y, w, h in regions:
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            plate, conf = ocr_region(reader, crop, deep=False)
            if plate:
                is_wanted = plate in wanted
                draw_result(frame, x, y, w, h, plate, is_wanted, conf)

        cv2.imshow("License Plate Detection (LIVE)", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
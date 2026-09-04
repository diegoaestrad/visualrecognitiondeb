#!/usr/bin/env python3
"""
Periodic-capture variant of detect_license.py.

Instead of scanning every live frame, this grabs a full-resolution still
every 10-15s (randomized), saves it to captures/ with a timestamped name,
and runs the fast EasyOCR pipeline against that still image ("offline",
after the fact) — printing whatever it finds to the console. Results go
to analysis_log.txt.

On startup, any images left over in captures/ from a previous run trigger
a prompt:
  - answer no  -> they stay in captures/ and get analyzed in a background
                  thread using a local Ollama vision model (qwen2.5vl:7b),
                  which reads plate text plus any other notable elements
                  in more detail than the live EasyOCR pass. Results go to
                  ai_analysis_log.txt. This runs concurrently with the live
                  capture loop below, on its own thread.
  - answer yes -> they're moved to archive/ untouched, to be analyzed later

Press 'q' in the preview window to quit.
"""

import base64
import os
import random
import threading
import time
from datetime import datetime

import cv2
import easyocr
import requests

from detect_license import (
    WANTED_FILE,
    detect_plate_regions,
    list_cameras,
    load_wanted_plates,
    ocr_region,
    plate_matches_format,
    refine_text,
    select_camera,
)

BASE_DIR = os.path.dirname(__file__)
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
LOG_FILE = os.path.join(BASE_DIR, "analysis_log.txt")
AI_LOG_FILE = os.path.join(BASE_DIR, "ai_analysis_log.txt")

CAPTURE_INTERVAL_RANGE = (10, 15)  # seconds
RESOLUTION_CANDIDATES = [(4096, 2160), (3840, 2160), (1920, 1080), (1280, 720)]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DUPLICATE_SIMILARITY_THRESHOLD = 0.70  # skip saving a capture this close to the previous one

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_MODEL = "qwen2.5vl:7b"
OLLAMA_PROMPT = (
    "You are reviewing a photo from a security camera watching a window/street. "
    "Read any visible license plate text exactly as printed (letters and numbers only). "
    "Also briefly note any other notable elements: vehicles, people, objects, signage or text. "
    "Respond in exactly this format:\n"
    "PLATE: <plate text, or NONE>\n"
    "NOTES: <one short sentence about other notable elements/text, or NONE>"
)


def frame_similarity(frame_a, frame_b, size=128, pixel_diff_threshold=25):
    gray_a = cv2.cvtColor(cv2.resize(frame_a, (size, size)), cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(cv2.resize(frame_b, (size, size)), cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    return float((diff < pixel_diff_threshold).sum()) / diff.size


def set_max_resolution(cap):
    best = (0, 0)
    for w, h in RESOLUTION_CANDIDATES:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual[0] * actual[1] > best[0] * best[1]:
            best = actual
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, best[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best[1])
    return int(best[0]), int(best[1])


def gather_existing_images():
    if not os.path.isdir(CAPTURES_DIR):
        return []
    return sorted(f for f in os.listdir(CAPTURES_DIR) if f.lower().endswith(IMAGE_EXTENSIONS))


def handle_existing_images():
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    existing = gather_existing_images()
    if not existing:
        return []

    print(f"\nFound {len(existing)} image(s) left over from a previous run in '{CAPTURES_DIR}'.")
    answer = input("Delete them now? They'll be archived, not analyzed this run. [y/N]: ").strip().lower()

    if answer == "y":
        for name in existing:
            os.replace(os.path.join(CAPTURES_DIR, name), os.path.join(ARCHIVE_DIR, name))
        print(f"Moved {len(existing)} image(s) to '{ARCHIVE_DIR}' for later analysis.\n")
        return []

    print(f"Keeping them — they'll be analyzed as part of this run.\n")
    return existing


def analyze_image(reader, wanted, image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    regions = detect_plate_regions(gray, area_min=500)

    findings = []
    for x, y, w, h in regions:
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        plate, conf = ocr_region(reader, crop, deep=True)
        if plate:
            findings.append((plate, plate in wanted, conf))

    if not findings:
        full_text = reader.readtext(frame, detail=0, paragraph=False)
        clean = refine_text(" ".join(full_text))
        if clean and plate_matches_format(clean):
            findings.append((clean, clean in wanted, 0.0))

    return findings


def log_findings(image_name, findings):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if findings:
        parts = [f"{plate} ({'WANTED' if is_wanted else 'ok'}, {conf:.0%})" for plate, is_wanted, conf in findings]
        line = f"[{timestamp}] {image_name} -> {', '.join(parts)}"
    else:
        line = f"[{timestamp}] {image_name} -> no plate-like text detected"

    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    print(line)


def ollama_model_available(model):
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=5)
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", [])]
        return model in names
    except requests.RequestException:
        return False


def ai_analyze_image(image_path):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": OLLAMA_PROMPT,
            "images": [image_b64],
            "stream": False,
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def log_ai_finding(image_name, response_text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {image_name} ->\n{response_text}\n"
    with open(AI_LOG_FILE, "a") as f:
        f.write(line + "\n")
    print(f"[AI] {image_name} analyzed (see {os.path.basename(AI_LOG_FILE)})")


def ai_backlog_worker(image_names):
    print(f"[AI] Background thread starting: analyzing {len(image_names)} old image(s) with {OLLAMA_MODEL}...")
    for name in image_names:
        path = os.path.join(CAPTURES_DIR, name)
        try:
            response_text = ai_analyze_image(path)
        except requests.RequestException as exc:
            response_text = f"AI analysis failed: {exc}"
        log_ai_finding(name, response_text)
    print("[AI] Background analysis of old images finished.")


def main():
    reader = easyocr.Reader(["en"], gpu=True)
    wanted = load_wanted_plates(WANTED_FILE)

    queued = handle_existing_images()
    if queued:
        if ollama_model_available(OLLAMA_MODEL):
            threading.Thread(target=ai_backlog_worker, args=(queued,), daemon=True).start()
        else:
            print(
                f"[AI] Model '{OLLAMA_MODEL}' isn't available in Ollama yet "
                f"(still pulling it? run 'ollama pull {OLLAMA_MODEL}') — "
                f"skipping detailed background analysis of the {len(queued)} old image(s) for now.\n"
            )

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

    width, height = set_max_resolution(cap)
    lo, hi = CAPTURE_INTERVAL_RANGE
    print(f"\nCamera {selected} started at {width}x{height}.")
    print(f"Capturing a frame every {lo}-{hi}s. Press 'q' in the preview window to quit.\n")

    next_capture = time.time() + random.uniform(lo, hi)
    last_capture_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture Preview (press q to quit)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

        if time.time() >= next_capture:
            if last_capture_frame is not None:
                similarity = frame_similarity(frame, last_capture_frame)
            else:
                similarity = 0.0

            if similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
                print(f"Skipped capture — {similarity:.0%} similar to the previous one, kept only that one.")
            else:
                filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                path = os.path.join(CAPTURES_DIR, filename)
                cv2.imwrite(path, frame)

                log_findings(filename, analyze_image(reader, wanted, path))
                last_capture_frame = frame.copy()

            next_capture = time.time() + random.uniform(lo, hi)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

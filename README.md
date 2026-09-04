# Visual Recognition Toolkit

A small collection of Python scripts for real-time and offline visual recognition using a webcam: generic object detection (YOLO11), license-plate reading with a "wanted list", an unattended offline capture mode, and an optional cloud-based image description script. Everything that touches OCR/plate detection runs locally by default; only `app.py` calls out to a cloud API.

## Features

- **Live object detection** (`object_detection.py`) - real-time YOLO11 inference on a webcam feed, with bounding boxes and class labels drawn live.
- **License plate detection & wanted-list matching** (`detect_license.py`) - live detection with on-screen bounding boxes/labels (WANTED in red / OK in green + confidence), plus a snapshot mode (SPACE) that runs a deeper OCR pass (CLAHE + denoising + full-frame OCR) and saves an annotated result image.
- **Offline periodic-capture mode** (`capture_analyze.py`) - instead of scanning every live frame, grabs a full-resolution still every 10-15s (randomized), runs local OCR against it, and logs results - designed to run unattended, without needing an internet connection for detection. Leftover images from a previous run can optionally be analyzed in the background with a local vision model served by Ollama, for a more detailed read.
- **Cloud image description demo** (`app.py`) - press SPACE to send the current frame to an NVIDIA NIM multimodal model and print a natural-language description. Requires your own API key and an internet connection; not used by the offline/local scripts above.

## Requirements

- Python 3.8+
- A working webcam
- Python packages: `opencv-python`, `ultralytics`, `easyocr`, `numpy`, `requests` (see `requirements.txt`)
- Optional, only for the background AI analysis in `capture_analyze.py`: [Ollama](https://ollama.com) running locally with the `qwen2.5vl:7b` model pulled (`ollama pull qwen2.5vl:7b`)
- Optional, only for `app.py`: an NVIDIA NIM API key

## Installation

```bash
git clone https://github.com/diegoaestrad/visualrecognitiondeb.git
cd visualrecognitiondeb
python3 -m venv objdetect_venv
source objdetect_venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the virtual environment with `objdetect_venv\Scripts\activate` instead.

## Usage

### Live object detection

```bash
python object_detection.py
```

- Lists connected cameras and lets you pick one.
- Press `q` to quit.

### License plate detection (live + snapshot)

```bash
python detect_license.py
```

- Select a camera, then live detection runs continuously with bounding boxes.
- Press `SPACE` for a deeper snapshot analysis (saved to `snapshots/`), `q` to quit.
- Optional: create a `wanted_plates.txt` file (one plate per line) in the project root to flag matches in red as WANTED.

### Offline periodic capture

```bash
./run_capture_analyze.sh
# or directly: python capture_analyze.py
```

- Before running the shell script, update the `cd` path inside `run_capture_analyze.sh` to match where you cloned the project.
- Captures a still every 10-15s to `captures/`, OCRs it, and appends results to `analysis_log.txt`.
- On startup, if images remain in `captures/` from a previous run, you'll be asked whether to archive them (answer `y` moves them to `archive/`) or keep analyzing them in the background with a local Ollama model, logged to `ai_analysis_log.txt` (any other answer).
- Press `q` in the preview window to quit.

### Cloud image description (optional, requires internet)

```bash
python app.py
```

- Replace `API_KEY` in `app.py` with your own NVIDIA NIM key first - never commit a real key to the repository.
- Press `SPACE` to analyze the current frame, `q` to quit.

## Configuration

| Setting | Where | Default |
| --- | --- | --- |
| Capture interval | `CAPTURE_INTERVAL_RANGE` in `capture_analyze.py` | 10-15s (randomized) |
| Duplicate-frame skip threshold | `DUPLICATE_SIMILARITY_THRESHOLD` in `capture_analyze.py` | 0.70 |
| Local vision model | `OLLAMA_MODEL` in `capture_analyze.py` | `qwen2.5vl:7b` |
| Wanted plates | `wanted_plates.txt` (create it yourself, one plate per line) | none / optional |

## Output & data files

`captures/`, `archive/`, `snapshots/`, `analysis_log.txt` and `ai_analysis_log.txt` are all git-ignored - they can contain real photos and license-plate data and must never be committed.

## Project structure

| File | Purpose |
| --- | --- |
| `object_detection.py` | Live generic object detection with YOLO11 |
| `detect_license.py` | Live + snapshot license plate detection with wanted-list matching |
| `capture_analyze.py` | Offline periodic-capture detection mode |
| `run_capture_analyze.sh` | Convenience launcher for the script above |
| `app.py` | Optional cloud-based image description demo (NVIDIA NIM) |
| `yolo11n.pt` | YOLO11 model weights used by `object_detection.py` |
| `wanted_plates.txt` | Optional, user-created - plates to flag as WANTED |

## Known limitations

- Camera selection uses `cv2.CAP_AVFOUNDATION`, which is macOS-specific; on Windows/Linux, change or remove that backend flag.
- `easyocr.Reader(..., gpu=True)` assumes a CUDA-capable GPU is available; set `gpu=False` if you don't have one.
- The plate-format check is a loose alphanumeric pattern and may produce false positives; treat matches as a starting point, not ground truth.

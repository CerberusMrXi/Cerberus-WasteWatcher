<div align="center">

<img src="https://img.shields.io/badge/🚧%20PROJECT%20STATUS-UNDER%20MAINTENANCE-orange?style=for-the-badge" alt="Under Maintenance">

<br>

<h2>🚧 Project Under Maintenance 🚧</h2>

<p>
This project is currently undergoing major improvements, bug fixes, performance optimizations,
and feature enhancements.
</p>

<p>
⚠️ Some features may be unstable or temporarily unavailable.
</p>

<p>
Thank you for your patience and support.
</p>

</div>

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1a2e,100:16213e&height=200&section=header&text=CERBERUS%20WASTEWATCHER&fontSize=42&fontColor=FFD700&fontAlignY=38&desc=Industry-Level%20AI%20Garbage%20and%20People%20Detection%20System&descColor=00FF88&descAlignY=58&descSize=16" width="100%"/>

<br/>

<p>
  <img src="https://img.shields.io/badge/Version-10.0.0-FFD700?style=for-the-badge&labelColor=0d1117"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=0d1117"/>
  <img src="https://img.shields.io/badge/YOLOv8s-Ultralytics-FF6B35?style=for-the-badge&labelColor=0d1117"/>
  <img src="https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white&labelColor=0d1117"/>
  <img src="https://img.shields.io/badge/License-MIT-00C896?style=for-the-badge&labelColor=0d1117"/>
  <img src="https://img.shields.io/badge/Status-Active-00FF88?style=for-the-badge&labelColor=0d1117"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Real--Time-Detection-FF4757?style=for-the-badge&labelColor=0d1117"/>
  <img src="https://img.shields.io/badge/IoU-Tracking-9B59B6?style=for-the-badge&labelColor=0d1117"/>
  <img src="https://img.shields.io/badge/COCO-80%20Classes-2ECC71?style=for-the-badge&labelColor=0d1117"/>
  <img src="https://img.shields.io/badge/RTSP-Ready-1ABC9C?style=for-the-badge&labelColor=0d1117"/>
</p>

<br/>

> **Built by [Sudeepa Wanigarathna](https://github.com/cerberusmrxi) — Cerberus Project Series**
>
> *An industry-grade AI surveillance system that detects people, identifies garbage and litter in real-time, tracks carry events across frames, and alerts on illegal dumping — all running on a standard webcam or IP camera.*

<br/>

```
╔══════════════════════════════════════════════════════════════════════╗
   🟢 Person Detected  |  🟡 Garbage/Litter  |  🟠 Bag/Backpack      
   🔴 Carrying Garbage |  🟣 Litter Drop Alert                        
╚══════════════════════════════════════════════════════════════════════╝
```

</div>

---

## 📌 Table of Contents

- [✨ What is Cerberus WasteWatcher?](#-what-is-cerberus-wastewatcher)
- [🏗️ System Architecture](#️-system-architecture)
- [🔬 Detection Engine](#-detection-engine)
- [⚡ Key Features](#-key-features)
- [🚀 Quick Start](#-quick-start)
- [🎛️ CLI Reference](#️-cli-reference)
- [📊 Detection Classes](#-detection-classes)
- [🧠 How It Works](#-how-it-works)
- [📁 Project Structure](#-project-structure)
- [📈 Performance](#-performance)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [👤 Author](#-author)

---

## ✨ What is Cerberus WasteWatcher?

**Cerberus WasteWatcher** is an AI-powered real-time surveillance system engineered for urban waste management and environmental monitoring. Using **YOLOv8s** deep learning inference paired with a custom **IoU-based multi-object tracker**, it can:

- 🎯 **Precisely detect people** in live camera feeds — fixing the common pitfall of aspect-ratio filters that reject standing humans
- 🗑️ **Identify garbage, litter, bags, bottles, and containers** using COCO class IDs (not broken string matching)
- 🔗 **Track carry events temporally** — a person must hold an item for ≥15 consecutive frames before an alert fires (eliminates false positives)
- 📍 **Detect illegal litter drops** — identifies when a garbage item lingers in a location for 45+ frames after being placed down
- 📸 **Auto-capture evidence** — saves annotated event images and face crops to disk
- 📋 **Log all events to CSV** — timestamped records with person ID, item class, confidence, and image path

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   CERBERUS WASTEWATCHER                      │
│                   Industry Edition v10.0                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │   Camera / RTSP │  ← Webcam, IP Cam, Video File
              │   Frame Source  │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  YOLOv8s Model  │  ← Ultralytics — COCO 80 classes
              │  (Inference)    │     conf=0.35, iou=0.45
              └────────┬────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
 ┌────────▼────────┐     ┌──────────▼──────────┐
 │ Person Tracker  │     │  Garbage Tracker     │
 │ (IoU — stable  │     │  (IoU — COCO IDs)    │
 │  IDs per frame)│     │  Bottles, Bags, Cups │
 └────────┬────────┘     └──────────┬──────────┘
          │                         │
          └───────────┬─────────────┘
                      │
             ┌────────▼────────┐
             │  CarryTracker   │  ← 15-frame temporal threshold
             │  (Temporal)     │     proximity + upper-body check
             └────────┬────────┘
                      │
             ┌────────▼────────┐
             │ LitterDropTrack │  ← 45-frame lingering detection
             │ (Drop Detect)   │
             └────────┬────────┘
                      │
          ┌───────────┴────────────┐
          │                        │
 ┌────────▼────────┐    ┌──────────▼──────────┐
 │   Alert Engine  │    │   HUD Renderer       │
 │ Image Save +    │    │  Bounding Boxes +    │
 │ Face Crop + CSV │    │  Track IDs + Legend  │
 └─────────────────┘    └──────────────────────┘
```

---

## 🔬 Detection Engine

### Why YOLOv8s?

| Model | mAP50 | Speed (CPU) | Size |
|-------|-------|-------------|------|
| YOLOv8n (old) | 37.3 | Fast | 6 MB |
| **YOLOv8s (new)** | **44.9** | Medium | 22 MB |
| YOLOv8m | 50.2 | Slow | 52 MB |

> WasteWatcher uses **YOLOv8s** — the optimal trade-off between accuracy and real-time speed.

### The COCO Class ID Fix

The original version attempted string matching against class names like `'trash'`, `'garbage'` — **none of which exist in COCO**. This was the primary reason garbage detection produced zero results.

```python
# ❌ OLD — broken, classes never matched
GARBAGE_CLASSES = ['trash', 'waste', 'garbage', 'sack']

# ✅ NEW — exact COCO integer IDs
GARBAGE_IDS = {
    39,  # bottle        41,  # cup
    24,  # backpack      26,  # handbag
    28,  # suitcase      45,  # bowl
    42,  # fork          43,  # knife
    73,  # book          79,  # toothbrush
    ...
}
```

---

## ⚡ Key Features

<table>
<tr>
<td width="50%">

### 🎯 Detection
- ✅ YOLOv8s — small model, high accuracy
- ✅ 80 COCO classes (real class ID matching)
- ✅ Bottles, cups, bags, backpacks, suitcases
- ✅ Adjustable confidence threshold
- ✅ Configurable IoU suppression
- ✅ 1280×720 HD resolution support

</td>
<td width="50%">

### 🔗 Tracking
- ✅ IoU-based multi-object tracker
- ✅ Stable person IDs across frames
- ✅ Temporal carry verification (15 frames)
- ✅ Upper-body proximity check (not feet)
- ✅ Litter-drop lingering detection (45 frames)
- ✅ Smart decay — avoids false positives

</td>
</tr>
<tr>
<td>

### 🚨 Alerts & Evidence
- ✅ Auto-saves annotated event images
- ✅ Face crop capture (Haar cascade)
- ✅ CSV event log with timestamps
- ✅ 3-second alert cooldown (no spam)
- ✅ Console alert with person ID + item info
- ✅ Unique person ID tracking across session

</td>
<td>

### 🖥️ HUD & Controls
- ✅ Real-time FPS counter
- ✅ Live person / carrying / litter counts
- ✅ Color-coded bounding boxes
- ✅ Track ID labels on every detection
- ✅ Carry confidence score display
- ✅ Keyboard: `q` quit `s` snapshot `r` reset

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
A webcam, IP camera, or video file
```

### 1. Clone the Repository

```bash
git clone https://github.com/cerberusmrxi/cerberus-wastewatcher.git
cd cerberus-wastewatcher
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, `yolov8s.pt` (~22MB) will be auto-downloaded by Ultralytics.

### 4. Run

```bash
# Default webcam (camera 0)
python3 WasteWatcher.py

# Watch! Green boxes = people, Yellow = garbage, Red = CARRYING ALERT 🔴
```

---

## 🎛️ CLI Reference

```bash
python3 WasteWatcher.py [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--camera` | `int` | `0` | Webcam device ID |
| `--source` | `str` | `None` | Video file path or RTSP URL |
| `--width` | `int` | `1280` | Frame width (pixels) |
| `--height` | `int` | `720` | Frame height (pixels) |
| `--conf` | `float` | `0.35` | Detection confidence threshold |
| `--model` | `str` | `yolov8s.pt` | YOLO model path or name |

### Examples

```bash
# Use external USB camera
python3 WasteWatcher.py --camera 1

# Analyze a recorded video
python3 WasteWatcher.py --source footage/mall_cctv.mp4

# Connect to an IP camera via RTSP
python3 WasteWatcher.py --source rtsp://admin:pass@192.168.1.10:554/stream

# More sensitive detection (lower threshold)
python3 WasteWatcher.py --conf 0.25

# Use a different model (nano for weaker hardware)
python3 WasteWatcher.py --model yolov8n.pt

# Full custom run
python3 WasteWatcher.py --source rtsp://... --conf 0.30 --width 1920 --height 1080
```

### Keyboard Controls (while running)

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Save a snapshot of the current frame |
| `r` | Reset all stats and counters |

---

## 📊 Detection Classes

### 🟠 Bag / Container Classes (Orange)
| COCO ID | Class | Detection Note |
|---------|-------|----------------|
| 24 | `backpack` | School bags, hiking packs |
| 26 | `handbag` | Purses, tote bags |
| 28 | `suitcase` | Luggage, carry-on bags |

### 🟡 Garbage / Litter Classes (Yellow)
| COCO ID | Class | Detection Note |
|---------|-------|----------------|
| 39 | `bottle` | Plastic bottles, water bottles |
| 40 | `wine glass` | Glass waste |
| 41 | `cup` | Coffee cups, disposable cups |
| 42 | `fork` | Cutlery litter |
| 43 | `knife` | Cutlery litter |
| 44 | `spoon` | Cutlery litter |
| 45 | `bowl` | Food containers |
| 73 | `book` | Paper waste |
| 75 | `vase` | Ceramic waste |
| 79 | `toothbrush` | Bathroom waste |

---

## 🧠 How It Works

### Step 1 — Detection
Every frame is passed through **YOLOv8s**. Detections are filtered by class ID (COCO integers) and by person aspect ratio `H/W > 1.3` (standing humans are tall, not wide — the old filter `0.3–1.0` rejected everyone).

### Step 2 — Tracking
Two independent **IoU trackers** maintain stable integer IDs for persons and garbage items across frames. When a detection overlaps with an existing track (IoU ≥ 0.30), the track ID is preserved.

### Step 3 — Carry Detection (Temporal)
For every `(person_id, garbage_id)` pair:
- The distance between their centers is computed
- If within `200px` AND the item is in the **upper 75% of the person's bounding box** (not on the ground), a frame counter increments
- After **15 consecutive frames** → the pair is declared **CARRYING**

```
Person bbox:
┌───────────────┐  ← y1
│  Head/Torso   │
│               │
│  [CARRY ZONE] │  ← 75% of height = upper body
│               │
├───────────────┤  ← 75% mark
│  Legs/Feet    │  ← item here = on ground, not carrying
└───────────────┘  ← y2
```

### Step 4 — Litter Drop Detection
Items NOT being carried that remain in the scene for **45+ frames** are flagged as **DROPPED LITTER** (magenta). This catches the case where someone placed garbage and walked away.

### Step 5 — Alert & Evidence
When a CARRYING or LITTER DROP event is confirmed:
1. Annotated event image saved to `Cerberus_Data/events/`
2. Face crop saved to `Cerberus_Data/faces/`
3. Event written to CSV log
4. Console alert with person ID, item, confidence, and timestamp

---

## 📁 Project Structure

```
Cerberus WasteWatcher/
│
├── WasteWatcher.py              ← Main application (all-in-one)
├── requirements.txt             ← Clean pip dependencies
├── yolov8s.pt                   ← Auto-downloaded on first run
│
└── Cerberus_Data/               ← Auto-created output directory
    ├── events/                  ← Annotated alert images (JPG)
    ├── faces/                   ← Captured face crops (JPG)
    └── logs/                    ← CSV event logs
        └── log_YYYYMMDD_HHMMSS.csv
```

### CSV Log Format

```csv
Timestamp,Event,PersonID,GarbageClass,GarbageConf,IsLitterDrop,ImagePath
2026-07-13 19:45:02,CARRYING,3,bottle,0.812,False,Cerberus_Data/events/event_20260713_194502.jpg
2026-07-13 19:47:31,LITTER_DROP,7,cup,0.741,True,Cerberus_Data/events/event_20260713_194731.jpg
```

---

## 📈 Performance

| Hardware | Resolution | FPS | Notes |
|----------|-----------|-----|-------|
| Intel i5 (CPU only) | 640×480 | ~8–12 | Acceptable for surveillance |
| Intel i7 (CPU only) | 1280×720 | ~15–20 | Smooth real-time |
| NVIDIA GPU (CUDA) | 1280×720 | ~45–60 | Full real-time HD |
| NVIDIA GPU (CUDA) | 1920×1080 | ~25–35 | Full HD |

> **GPU Tip:** Uncomment the `cupy-cudaXX` line in `requirements.txt` matching your CUDA version for 3–5× speedup.

---

## 🛠️ Troubleshooting

<details>
<summary><b>❌ Camera not opening</b></summary>

Try different camera IDs:
```bash
python3 WasteWatcher.py --camera 0
python3 WasteWatcher.py --camera 1
python3 WasteWatcher.py --camera 2
```
On Linux, verify devices: `ls /dev/video*`
</details>

<details>
<summary><b>❌ No people being detected</b></summary>

Lower the confidence threshold:
```bash
python3 WasteWatcher.py --conf 0.20
```
The person aspect-ratio filter requires `H/W > 1.3` — ensure the camera captures full or half body, not just a head.
</details>

<details>
<summary><b>❌ No garbage detected</b></summary>

Garbage detection uses COCO classes — test with a **bottle, cup, backpack, or handbag** held in front of the camera. Lower confidence if needed:
```bash
python3 WasteWatcher.py --conf 0.20
```
</details>

<details>
<summary><b>❌ pip install fails</b></summary>

Ensure you are using the venv and a Python 3.8+ environment:
```bash
python3 --version
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
</details>

<details>
<summary><b>❌ YOLO model not found</b></summary>

The model auto-downloads on first run. If behind a firewall, manually download:
```bash
# Using ultralytics CLI
pip install ultralytics
yolo export model=yolov8s.pt   # triggers download
```
</details>

---

## 🏆 Version History

| Version | Highlights |
|---------|-----------|
| **10.0.0** | IoU tracker, COCO class IDs, temporal carry logic, litter drop detection, bug fixes |
| 9.0.0 | Bag detection, motion tracking, multi-color HUD |
| 8.x | Initial YOLO integration, face detection |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

<div align="center">

<img src="https://avatars.githubusercontent.com/cerberusmrxi" width="100" style="border-radius:50%"/>

### Sudeepa Wanigarathna

*Computer Vision Engineer · AI Developer · Cerberus Project Series*

[![GitHub](https://img.shields.io/badge/GitHub-cerberusmrxi-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/cerberusmrxi)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/sudeepa-wanigarathna)

</div>

---

<div align="center">

**Part of the Cerberus Project Series** — Building intelligent AI surveillance tools.

*Cerberus WasteWatcher · Cerberus Future · Cerberus Chart Oracle · Cerberus 30*

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:16213e,50:1a1a2e,100:0d1117&height=100&section=footer" width="100%"/>

</div>

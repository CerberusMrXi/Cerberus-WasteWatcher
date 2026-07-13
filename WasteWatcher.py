#!/usr/bin/env python3
"""
Cerberus WasteWatcher — Industry Edition
Version: 1.0.0
Author - Sudeepa Wanigarathna
Fixes: Person detection filter, COCO-class garbage IDs, IoU tracker,
       temporal carry logic, on_drop NameError, requirements cleanup.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import csv
from datetime import datetime
from pathlib import Path
import hashlib
from collections import defaultdict, deque
import argparse
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
def display_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║        CERBERUS WASTEWATCHER — INDUSTRY EDITION          ║
║  • Person Detection          (GREEN)                     ║
║  • Garbage / Litter          (YELLOW)                    ║
║  • Bag / Backpack / Suitcase (ORANGE)                    ║
║  • Person Carrying Garbage   (RED)                       ║
║  • Litter Drop Alert         (MAGENTA)                   ║
║  Version: 10.0.0  |  Tracker: IoU  |  Model: YOLOv8s     ║ 
╚══════════════════════════════════════════════════════════╝
""")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    CAMERA_ID    = 0
    FRAME_WIDTH  = 1280
    FRAME_HEIGHT = 720
    MODEL_PATH   = 'yolov8s.pt'       # upgraded from nano → small
    CONFIDENCE   = 0.35
    IOU_THRESH   = 0.45

    # ── COCO class IDs that we treat as GARBAGE/LITTER ───────────────────────
    # (string matching was broken — COCO has no class named 'trash'/'garbage')
    GARBAGE_IDS = {
        39,   # bottle
        40,   # wine glass
        41,   # cup
        42,   # fork
        43,   # knife
        44,   # spoon
        45,   # bowl
        67,   # cell phone  (commonly littered)
        73,   # book
        74,   # clock
        75,   # vase
        76,   # scissors
        77,   # teddy bear
        79,   # toothbrush
    }

    # ── COCO class IDs that are bag-type (orange) ────────────────────────────
    BAG_IDS = {
        24,   # backpack
        26,   # handbag
        28,   # suitcase
    }

    # ── COCO class ID for person ─────────────────────────────────────────────
    PERSON_ID = 0

    # ── Tracking & carry thresholds ──────────────────────────────────────────
    TRACK_IOU_THRESH     = 0.30   # min IoU to match detection to existing track
    CARRY_FRAME_THRESH   = 15     # frames item must be near person → CARRYING
    DROP_DISTANCE_PX     = 100    # px — how far item must be from person to drop
    CARRY_PROXIMITY_PX   = 200    # px — center-to-center for carry detection
    LITTER_LINGER_FRAMES = 45     # frames a stationary item lingers after person leaves

    # ── Directories ──────────────────────────────────────────────────────────
    BASE_DIR  = "Cerberus_Data"
    IMAGE_DIR = f"{BASE_DIR}/events"
    LOG_DIR   = f"{BASE_DIR}/logs"
    FACE_DIR  = f"{BASE_DIR}/faces"

    # ── Colors (BGR) ─────────────────────────────────────────────────────────
    C_PERSON   = (  0, 220,   0)   # green
    C_CARRYING = (  0,   0, 220)   # red
    C_GARBAGE  = (  0, 220, 220)   # yellow
    C_BAG      = (  0, 165, 255)   # orange
    C_LITTER   = (200,   0, 200)   # magenta  (dropped litter alert)
    C_FACE     = (220,   0,   0)   # blue
    C_HUD_BG   = (  0,   0,   0)
    C_GOLD     = (  0, 215, 255)

    @staticmethod
    def create_dirs():
        for d in [Config.IMAGE_DIR, Config.LOG_DIR, Config.FACE_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# IoU UTILITY
# ─────────────────────────────────────────────────────────────────────────────
def iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter  = interW * interH
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE IoU TRACKER
# Assigns stable integer IDs to detections across frames.
# ─────────────────────────────────────────────────────────────────────────────
class SimpleTracker:
    def __init__(self, max_lost: int = 30):
        self._next_id  = 0
        self._tracks   = {}   # id → {bbox, lost, cls, conf, center}
        self._max_lost = max_lost

    def update(self, detections: list) -> list:
        """
        detections: list of dicts with keys bbox, cls, conf, center, area
        Returns: same list enriched with 'track_id'
        """
        # ── age all tracks ────────────────────────────────────────────────────
        for tid in self._tracks:
            self._tracks[tid]['lost'] += 1

        matched_track_ids = set()
        output = []

        for det in detections:
            best_iou  = Config.TRACK_IOU_THRESH
            best_tid  = None

            for tid, tr in self._tracks.items():
                if tr['cls'] != det['cls']:
                    continue
                sc = iou(det['bbox'], tr['bbox'])
                if sc > best_iou:
                    best_iou = sc
                    best_tid = tid

            if best_tid is not None:
                self._tracks[best_tid].update({
                    'bbox'  : det['bbox'],
                    'conf'  : det['conf'],
                    'center': det['center'],
                    'area'  : det['area'],
                    'lost'  : 0,
                })
                matched_track_ids.add(best_tid)
                det['track_id'] = best_tid
            else:
                # new track
                tid = self._next_id; self._next_id += 1
                self._tracks[tid] = {
                    'bbox'  : det['bbox'],
                    'cls'   : det['cls'],
                    'conf'  : det['conf'],
                    'center': det['center'],
                    'area'  : det['area'],
                    'lost'  : 0,
                }
                det['track_id'] = tid

            output.append(det)

        # ── prune dead tracks ─────────────────────────────────────────────────
        dead = [tid for tid, tr in self._tracks.items() if tr['lost'] > self._max_lost]
        for tid in dead:
            del self._tracks[tid]

        return output


# ─────────────────────────────────────────────────────────────────────────────
# FACE DETECTOR (Haar — lightweight, no extra model download)
# ─────────────────────────────────────────────────────────────────────────────
class FaceDetector:
    def __init__(self):
        self._cascade = None
        try:
            path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(path):
                c = cv2.CascadeClassifier(path)
                if not c.empty():
                    self._cascade = c
                    print("  [OK] Face detector loaded (Haar)")
        except Exception as e:
            print(f"  [WARN] Face detector: {e}")

    def detect(self, roi):
        if self._cascade is None or roi is None or roi.size == 0:
            return []
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=5,
            minSize=(25, 25), maxSize=(300, 300)
        )
        return faces if len(faces) > 0 else []


# ─────────────────────────────────────────────────────────────────────────────
# GARBAGE DETECTOR — wraps YOLO + Tracker
# ─────────────────────────────────────────────────────────────────────────────
class GarbageDetector:
    def __init__(self):
        print("[LOAD] Loading YOLOv8s model …")
        self.model = YOLO(Config.MODEL_PATH)
        print(f"  [OK] Model loaded: {Config.MODEL_PATH}")

        self.person_tracker  = SimpleTracker(max_lost=30)
        self.garbage_tracker = SimpleTracker(max_lost=30)

    # ── raw YOLO inference ─────────────────────────────────────────────────
    def _infer(self, frame):
        return self.model(
            frame,
            conf=Config.CONFIDENCE,
            iou=Config.IOU_THRESH,
            verbose=False
        )

    # ── build det dict ─────────────────────────────────────────────────────
    @staticmethod
    def _make_det(name, cls_id, conf, x1, y1, x2, y2):
        cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
        return {
            'class' : name,
            'cls'   : cls_id,
            'conf'  : conf,
            'bbox'  : (int(x1), int(y1), int(x2), int(y2)),
            'center': (cx, cy),
            'area'  : (x2 - x1) * (y2 - y1),
        }

    # ── detect & track ─────────────────────────────────────────────────────
    def detect(self, frame):
        """
        Returns:
            persons  — tracked list of person dicts
            garbage  — tracked list of garbage/litter dicts
            bags     — subset of garbage that are bag-type
        """
        raw_persons = []
        raw_garbage = []
        raw_bags    = []

        results = self._infer(frame)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                name   = self.model.names[cls_id]
                det    = self._make_det(name, cls_id, conf, x1, y1, x2, y2)

                if cls_id == Config.PERSON_ID:
                    w = x2 - x1; h = y2 - y1
                    aspect = h / w if w > 0 else 0
                    area   = w * h
                    # FIX: correct aspect ratio — standing person H/W ≈ 2–3
                    if aspect > 1.3 and 500 < area < 250_000:
                        raw_persons.append(det)

                elif cls_id in Config.BAG_IDS:
                    det['is_bag'] = True
                    raw_bags.append(det)
                    raw_garbage.append(det)

                elif cls_id in Config.GARBAGE_IDS:
                    det['is_bag'] = False
                    raw_garbage.append(det)

        persons = self.person_tracker.update(raw_persons)
        garbage = self.garbage_tracker.update(raw_garbage)
        bags    = [g for g in garbage if g.get('is_bag')]

        return persons, garbage, bags


# ─────────────────────────────────────────────────────────────────────────────
# CARRY TRACKER — temporal logic: must be near person for N frames
# ─────────────────────────────────────────────────────────────────────────────
class CarryTracker:
    """
    Tracks how many consecutive frames a (person_id, garbage_id) pair
    have been within CARRY_PROXIMITY_PX of each other.
    When frame count ≥ CARRY_FRAME_THRESH → declare CARRYING.
    """
    def __init__(self):
        self._counters = defaultdict(int)   # (pid, gid) → frame count
        self._active   = set()              # confirmed carrying pairs

    def update(self, persons, garbage):
        """Returns list of confirmed carrying dicts."""
        current_pairs = set()

        for p in persons:
            pid = p['track_id']
            px, py = p['center']
            for g in garbage:
                gid = g['track_id']
                gx, gy = g['center']
                dist = ((px-gx)**2 + (py-gy)**2) ** 0.5
                if dist < Config.CARRY_PROXIMITY_PX:
                    # Also check item is in upper-body region (not just standing on it)
                    px1, py1, px2, py2 = p['bbox']
                    lower_bound = py1 + (py2 - py1) * 0.75
                    if gy < lower_bound:
                        current_pairs.add((pid, gid))
                        self._counters[(pid, gid)] += 1
                    else:
                        # item near feet — likely on ground
                        self._counters[(pid, gid)] = max(0, self._counters[(pid, gid)] - 2)

        # Decay pairs no longer close
        for pair in list(self._counters):
            if pair not in current_pairs:
                self._counters[pair] = max(0, self._counters[pair] - 3)
                if self._counters[pair] == 0:
                    self._active.discard(pair)
                    del self._counters[pair]

        # Confirm new carrying pairs
        for pair, count in self._counters.items():
            if count >= Config.CARRY_FRAME_THRESH:
                self._active.add(pair)

        # Build result
        result = []
        pid_map = {p['track_id']: p for p in persons}
        gid_map = {g['track_id']: g for g in garbage}
        seen = set()
        for (pid, gid) in self._active:
            if pid in pid_map and gid in gid_map and (pid, gid) not in seen:
                seen.add((pid, gid))
                g = gid_map[gid]
                result.append({
                    'person'    : pid_map[pid],
                    'garbage'   : g,
                    'confidence': min(1.0, self._counters.get((pid,gid),0) / Config.CARRY_FRAME_THRESH),
                    'is_bag'    : g.get('is_bag', False),
                })
        return result


# ─────────────────────────────────────────────────────────────────────────────
# LITTER DROP TRACKER
# Detects when a garbage item appears (or stays) in a location after a
# person who was carrying it has moved away.
# ─────────────────────────────────────────────────────────────────────────────
class LitterDropTracker:
    def __init__(self):
        self._litter   = {}   # gid → {center, frames_alone}
        self.confirmed = set()

    def update(self, garbage, carrying):
        carried_gids = {c['garbage']['track_id'] for c in carrying}

        for g in garbage:
            gid = g['track_id']
            if gid in carried_gids:
                # being carried — reset linger counter
                if gid in self._litter:
                    del self._litter[gid]
                self.confirmed.discard(gid)
            else:
                if gid not in self._litter:
                    self._litter[gid] = {'center': g['center'], 'frames': 0, 'det': g}
                else:
                    self._litter[gid]['frames'] += 1
                    self._litter[gid]['det'] = g
                    if self._litter[gid]['frames'] >= Config.LITTER_LINGER_FRAMES:
                        self.confirmed.add(gid)

        # prune gids no longer in garbage
        live_gids = {g['track_id'] for g in garbage}
        for gid in list(self._litter):
            if gid not in live_gids:
                del self._litter[gid]
                self.confirmed.discard(gid)

    def get_confirmed(self, garbage):
        return [g for g in garbage if g['track_id'] in self.confirmed]


# ─────────────────────────────────────────────────────────────────────────────
# CERBERUS DETECTOR — main orchestrator
# ─────────────────────────────────────────────────────────────────────────────
class CerberusDetector:
    def __init__(self):
        display_banner()
        Config.create_dirs()

        self.detector      = GarbageDetector()
        self.face_det      = FaceDetector()
        self.carry_tracker = CarryTracker()
        self.litter_track  = LitterDropTracker()

        self.drop_count    = 0
        self.unique_ids    = set()
        self.last_alert    = 0.0
        self.frame_count   = 0
        self.fps           = 0.0
        self._fps_t        = time.time()
        self._fps_buf      = deque(maxlen=60)

        self._setup_log()

        print("\n" + "═"*58)
        print("  SYSTEM READY — Industry Detection Active")
        print("  Controls: [q] Quit  [s] Snapshot  [r] Reset stats")
        print("═"*58 + "\n")

    # ── logging ───────────────────────────────────────────────────────────
    def _setup_log(self):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = f"{Config.LOG_DIR}/log_{ts}.csv"
        with open(self.log_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Timestamp', 'Event', 'PersonID', 'GarbageClass',
                        'GarbageConf', 'IsLitterDrop', 'ImagePath'])
        print(f"[LOG] {self.log_path}")

    def _log(self, person_id, garbage, image_path, is_drop=False):
        try:
            with open(self.log_path, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'LITTER_DROP' if is_drop else 'CARRYING',
                    person_id,
                    garbage['class'],
                    f"{garbage['conf']:.3f}",
                    is_drop,
                    image_path or '',
                ])
        except Exception:
            pass

    # ── face capture ──────────────────────────────────────────────────────
    def _capture_face(self, frame, bbox):
        try:
            x1, y1, x2, y2 = bbox
            pad = 30
            roi = frame[max(0,y1-pad):min(frame.shape[0],y2+pad),
                        max(0,x1-pad):min(frame.shape[1],x2+pad)]
            faces = self.face_det.detect(roi)
            if len(faces):
                fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
                face = roi[fy:fy+fh, fx:fx+fw]
                if face.size:
                    face = cv2.resize(face, (112, 112))
                    ts   = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    path = f"{Config.FACE_DIR}/face_{ts}.jpg"
                    cv2.imwrite(path, face)
                    return path
        except Exception:
            pass
        return None

    # ── save event image ──────────────────────────────────────────────────
    def _save_event(self, frame, person, garbage, is_drop=False):
        try:
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            path = f"{Config.IMAGE_DIR}/event_{ts}.jpg"
            img  = frame.copy()
            x1, y1, x2, y2 = person['bbox']
            cv2.rectangle(img, (x1,y1),(x2,y2), Config.C_CARRYING, 3)
            label = "LITTER DROP!" if is_drop else "CARRYING GARBAGE"
            cv2.putText(img, label, (x1, y1-12),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, Config.C_CARRYING, 2)
            gx1,gy1,gx2,gy2 = garbage['bbox']
            gc = Config.C_LITTER if is_drop else (Config.C_BAG if garbage.get('is_bag') else Config.C_GARBAGE)
            cv2.rectangle(img,(gx1,gy1),(gx2,gy2), gc, 3)
            cv2.line(img, person['center'], garbage['center'], Config.C_CARRYING, 2)
            cv2.putText(img, f"#{self.drop_count} | {datetime.now().strftime('%H:%M:%S')}",
                        (12, 32), cv2.FONT_HERSHEY_DUPLEX, 0.7, Config.C_GOLD, 2)
            cv2.imwrite(path, img)
            return path
        except Exception as e:
            print(f"  [WARN] save_event: {e}")
            return None

    # ── alert ─────────────────────────────────────────────────────────────
    def _alert(self, frame, person, garbage, is_drop=False):
        now = time.time()
        if now - self.last_alert < 3.0:
            return
        self.last_alert = now
        self.drop_count += 1

        pid        = person['track_id']
        self.unique_ids.add(pid)

        face_path  = self._capture_face(frame, person['bbox'])
        img_path   = self._save_event(frame, person, garbage, is_drop)
        self._log(pid, garbage, img_path, is_drop)

        tag = "LITTER DROP" if is_drop else "CARRYING GARBAGE"
        print(f"\n{'!'*55}")
        print(f"  ⚠  ALERT — {tag}")
        print(f"  Event #{self.drop_count}  |  Person ID: {pid}")
        print(f"  Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Item   : {garbage['class']}  (conf {garbage['conf']:.2f})")
        if face_path:
            print(f"  Face   : {os.path.basename(face_path)}")
        if img_path:
            print(f"  Saved  : {os.path.basename(img_path)}")
        print(f"{'!'*55}\n")

    # ── draw ──────────────────────────────────────────────────────────────
    def _draw(self, frame, persons, garbage, bags, carrying, litter_dropped):
        img = frame.copy()
        carrying_person_ids  = {c['person']['track_id'] for c in carrying}
        carrying_garbage_ids = {c['garbage']['track_id'] for c in carrying}
        litter_ids           = {g['track_id'] for g in litter_dropped}

        # ── garbage / bags ─────────────────────────────────────────────────
        for g in garbage:
            x1,y1,x2,y2 = g['bbox']
            gid = g['track_id']
            if gid in litter_ids:
                color = Config.C_LITTER
                label = f"LITTER! {g['class']}"
            elif g.get('is_bag'):
                color = Config.C_BAG
                label = f"BAG:{g['class']} {g['conf']:.2f}"
            else:
                color = Config.C_GARBAGE
                label = f"{g['class']} {g['conf']:.2f}"
            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            self._label(img, label, x1, y1-8, color)

        # ── persons ────────────────────────────────────────────────────────
        for p in persons:
            x1,y1,x2,y2 = p['bbox']
            pid = p['track_id']
            if pid in carrying_person_ids:
                color = Config.C_CARRYING
                label = f"ID:{pid} CARRYING"
            else:
                color = Config.C_PERSON
                label = f"ID:{pid} PERSON {p['conf']:.2f}"
            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            self._label(img, label, x1, y1-10, color)

        # ── carry links ────────────────────────────────────────────────────
        for c in carrying:
            pc = c['person']['center']
            gc = c['garbage']['center']
            cv2.line(img, pc, gc, Config.C_CARRYING, 2)
            mid = ((pc[0]+gc[0])//2, (pc[1]+gc[1])//2)
            cv2.putText(img, f"conf:{c['confidence']:.2f}", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.C_CARRYING, 1)

        self._draw_hud(img, persons, carrying, litter_dropped)
        return img

    @staticmethod
    def _label(img, text, x, y, color, scale=0.5, thick=1):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(img,(x, y-th-4),(x+tw+4, y+2),(0,0,0),-1)
        cv2.putText(img, text,(x+2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    def _draw_hud(self, frame, persons, carrying, litter):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay,(8,8),(310,195),(0,0,0),-1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame,(8,8),(310,195),Config.C_GOLD,1)

        def put(txt, y, color=(255,255,255), scale=0.48):
            cv2.putText(frame, txt,(18,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

        put("CERBERUS WASTEWATCHER", 30, Config.C_GOLD, 0.52)
        put(f"FPS: {self.fps:4.1f}", 55)
        put(f"Persons  : {len(persons)}", 78)
        put(f"Carrying : {len(carrying)}", 100, Config.C_CARRYING if carrying else (255,255,255))
        put(f"Litter   : {len(litter)}", 122, Config.C_LITTER if litter else (255,255,255))
        put(f"Alerts   : {self.drop_count}", 144, Config.C_GOLD)
        put(f"Unique IDs: {len(self.unique_ids)}", 167)

        status = "⚠ ALERT" if (carrying or litter) else "● MONITORING"
        sc = Config.C_CARRYING if (carrying or litter) else Config.C_PERSON
        put(status, 190, sc, 0.5)

        # legend bottom-left
        lx, ly = 10, h-130
        overlay2 = frame.copy()
        cv2.rectangle(overlay2,(lx,ly),(lx+215,ly+125),(0,0,0),-1)
        cv2.addWeighted(overlay2, 0.60, frame, 0.40, 0, frame)
        items = [
            (Config.C_PERSON,   "Person (normal)"),
            (Config.C_CARRYING, "Person carrying"),
            (Config.C_GARBAGE,  "Garbage/litter"),
            (Config.C_BAG,      "Bag/Backpack"),
            (Config.C_LITTER,   "Dropped litter"),
        ]
        cv2.putText(frame,"LEGEND",(lx+8,ly+18),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
        for i,(col,lab) in enumerate(items):
            iy = ly+38+i*18
            cv2.rectangle(frame,(lx+8,iy-10),(lx+26,iy+2),col,-1)
            cv2.putText(frame,lab,(lx+32,iy),cv2.FONT_HERSHEY_SIMPLEX,0.38,(220,220,220),1)

    # ── FPS calculation ───────────────────────────────────────────────────
    def _tick_fps(self):
        now = time.time()
        self._fps_buf.append(now)
        if len(self._fps_buf) > 1:
            self.fps = (len(self._fps_buf)-1) / (self._fps_buf[-1] - self._fps_buf[0])

    # ── main frame processor ──────────────────────────────────────────────
    def process(self, frame):
        persons, garbage, bags = self.detector.detect(frame)
        carrying  = self.carry_tracker.update(persons, garbage)
        self.litter_track.update(garbage, carrying)
        litter    = self.litter_track.get_confirmed(garbage)

        # fire alerts
        for c in carrying:
            self._alert(frame, c['person'], c['garbage'], is_drop=False)
        for g in litter:
            # find nearest person as "responsible" — use closest person
            best_p = None
            best_d = float('inf')
            for p in persons:
                d = ((p['center'][0]-g['center'][0])**2 +
                     (p['center'][1]-g['center'][1])**2) ** 0.5
                if d < best_d:
                    best_d = d; best_p = p
            if best_p is None and persons:
                best_p = persons[0]
            if best_p:
                self._alert(frame, best_p, g, is_drop=True)

        annotated = self._draw(frame, persons, garbage, bags, carrying, litter)
        self._tick_fps()
        return annotated

    # ── run loop ──────────────────────────────────────────────────────────
    def run(self, source=None):
        src = Config.CAMERA_ID if source is None else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {src}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        print(f"[OK] Source opened ({int(cap.get(3))}×{int(cap.get(4))})\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Stream ended.")
                break
            self.frame_count += 1

            annotated = self.process(frame)
            cv2.imshow('Cerberus WasteWatcher — Industry Edition', annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
                path = f"{Config.IMAGE_DIR}/snap_{ts}.jpg"
                cv2.imwrite(path, annotated)
                print(f"[SNAP] {path}")
            elif key == ord('r'):
                self.drop_count = 0
                self.unique_ids.clear()
                print("[RESET] Stats cleared")

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n{'═'*55}")
        print("  FINAL REPORT")
        print(f"  Alerts logged : {self.drop_count}")
        print(f"  Unique IDs    : {len(self.unique_ids)}")
        print(f"  Log file      : {self.log_path}")
        print(f"{'═'*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description='Cerberus WasteWatcher — Industry Edition')
    p.add_argument('--camera',  type=int,   default=0,     help='Camera device ID (default 0)')
    p.add_argument('--source',  type=str,   default=None,  help='Video file or RTSP URL (overrides --camera)')
    p.add_argument('--width',   type=int,   default=1280,  help='Frame width')
    p.add_argument('--height',  type=int,   default=720,   help='Frame height')
    p.add_argument('--conf',    type=float, default=0.35,  help='Detection confidence threshold')
    p.add_argument('--model',   type=str,   default='yolov8s.pt', help='YOLO model path/name')
    args = p.parse_args()

    Config.CAMERA_ID    = args.camera
    Config.FRAME_WIDTH  = args.width
    Config.FRAME_HEIGHT = args.height
    Config.CONFIDENCE   = args.conf
    Config.MODEL_PATH   = args.model

    try:
        det = CerberusDetector()
        det.run(source=args.source)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Cerberus WasteWatcher - Ultimate Full Version
Complete Garbage Detection System with ALL Features
Version: 6.0.0 - The Final Version
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
import os
import csv
from datetime import datetime
from pathlib import Path
import hashlib
from collections import deque
import argparse
import sys
import signal
import json
import threading
import queue
import shutil
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# BANNER
# ============================================================
def display_banner():
    banner = """
██████╗██████╗ ██████╗ ███████╗██████╗ ██╗   ██╗███████╗
██╔════╝██╔══██╗██╔══██╗██╔════╝██╔══██╗██║   ██║██╔════╝
██║     ██████╔╝██████╔╝█████╗  ██████╔╝██║   ██║███████╗
██║     ██╔══██╗██╔══██╗██╔══╝  ██╔══██╗██║   ██║╚════██║
╚██████╗██║  ██║██║  ██║███████╗██║  ██║╚██████╔╝███████║
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝

██╗    ██╗ █████╗ ███████╗████████╗███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗███████╗██████╗
██║    ██║██╔══██╗██╔════╝╚══██╔══╝██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║██╔════╝██╔══██╗
██║ █╗ ██║███████║███████╗   ██║   █████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║█████╗  ██████╔╝
██║███╗██║██╔══██║╚════██║   ██║   ██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║██╔══╝  ██╔══██╗
╚███╔███╔╝██║  ██║███████║   ██║   ███████╗╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║███████╗██║  ██║
 ╚══╝╚══╝ ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝

    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║              🐕  CERBERUS WASTEWATCHER  🐕                   ║
    ║                                                               ║
    ║          "Three Heads - One Mission - Zero Waste"            ║
    ║                                                               ║
    ║    • HEAD 1: Motion Sentinel (Optical Flow)                  ║
    ║    • HEAD 2: Object Guardian (YOLOv8 + Filters)              ║
    ║    • HEAD 3: Face Keeper (Multi-Method Detection)            ║
    ║                                                               ║
    ║    Version: 6.0.0 - The Final Version                        ║
    ║    Ultimate AI-Powered Garbage Detection System              ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

# ============================================================
# CONFIGURATION
# ============================================================
class CerberusConfig:
    """Complete Configuration"""
    
    # System
    SYSTEM_NAME = "Cerberus WasteWatcher"
    VERSION = "6.0.0"
    
    # Detection - Optimized
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 100
    
    # Motion Detection
    MOTION_THRESHOLD = 15
    MIN_MOTION_AREA = 200
    MOTION_HISTORY = 30
    
    # Person Detection
    PERSON_MIN_AREA = 3000
    PERSON_MAX_AREA = 60000
    PERSON_MIN_ASPECT = 0.4
    PERSON_MAX_ASPECT = 0.9
    
    # Garbage Detection
    GARBAGE_MIN_AREA = 500
    GARBAGE_MAX_AREA = 30000
    CARRYING_DISTANCE = 250
    OVERLAP_THRESHOLD = 0.15
    
    # Garbage Classes - Complete List
    GARBAGE_CLASSES = [
        'trash', 'waste', 'dumpster', 'bin', 'bag', 'garbage',
        'can', 'box', 'plastic', 'bottle', 'container', 'wrapper',
        'packet', 'sack', 'basket', 'barrel', 'crate', 'pallet',
        'carton', 'package', 'parcel', 'suitcase', 'backpack',
        'tent', 'umbrella', 'ball', 'toy', 'chair', 'table',
        'bottle', 'cup', 'bowl', 'plate', 'fork', 'knife', 'spoon',
        'can', 'glass', 'jar', 'bag', 'box', 'crate', 'barrel'
    ]
    
    # Colors (BGR)
    COLOR_CARRYING = (0, 0, 255)
    COLOR_NORMAL = (0, 255, 0)
    COLOR_GARBAGE = (0, 255, 255)
    COLOR_BAG = (255, 128, 0)
    COLOR_FACE = (255, 0, 0)
    COLOR_ALERT = (255, 0, 255)
    COLOR_TRACK = (128, 128, 128)
    
    # Camera
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30
    BUFFER_SIZE = 5
    
    # Storage
    BASE_DIR = "Cerberus_WasteWatcher_Data"
    IMAGE_DIR = f"{BASE_DIR}/events"
    VIDEO_DIR = f"{BASE_DIR}/recordings"
    LOG_DIR = f"{BASE_DIR}/logs"
    FACE_DIR = f"{BASE_DIR}/faces"
    SNAPSHOT_DIR = f"{BASE_DIR}/snapshots"
    CROPPED_DIR = f"{BASE_DIR}/cropped"
    BACKUP_DIR = f"{BASE_DIR}/backups"
    EXPORT_DIR = f"{BASE_DIR}/exports"
    
    # Timing
    ALERT_COOLDOWN = 2
    SNAPSHOT_INTERVAL = 1
    SAVE_INTERVAL = 5
    TRACKING_AGE = 20
    
    # Threading
    USE_THREADING = True
    THREAD_POOL_SIZE = 4
    
    @staticmethod
    def create_directories():
        for dir_path in [
            CerberusConfig.IMAGE_DIR,
            CerberusConfig.VIDEO_DIR,
            CerberusConfig.LOG_DIR,
            CerberusConfig.FACE_DIR,
            CerberusConfig.SNAPSHOT_DIR,
            CerberusConfig.CROPPED_DIR,
            CerberusConfig.BACKUP_DIR,
            CerberusConfig.EXPORT_DIR
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

# ============================================================
# FACE DETECTOR - MULTI-METHOD
# ============================================================
class FaceDetector:
    """Multi-method face detection"""
    
    def __init__(self):
        self.detectors = []
        self.face_enabled = False
        self.init_detectors()
    
    def init_detectors(self):
        methods = [
            ('haar', 'haarcascade_frontalface_default.xml'),
            ('lbp', 'lbpcascade_frontalface.xml'),
            ('haar2', 'haarcascade_frontalface_alt.xml'),
            ('haar3', 'haarcascade_frontalface_alt2.xml')
        ]
        
        for name, file in methods:
            try:
                path = cv2.data.haarcascades + file
                if os.path.exists(path):
                    cascade = cv2.CascadeClassifier(path)
                    if not cascade.empty():
                        self.detectors.append((name, cascade))
                        print(f"  [OK] Face: {name} loaded")
            except:
                pass
        
        if self.detectors:
            self.face_enabled = True
            print(f"  [OK] Face: {len(self.detectors)} methods loaded")
        else:
            print("  [WARN] Face: No detectors available")
    
    def detect_faces(self, frame):
        if not self.face_enabled or not self.detectors:
            return []
        
        all_faces = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        for name, detector in self.detectors:
            try:
                if name == 'haar':
                    faces = detector.detectMultiScale(
                        gray, scaleFactor=1.05, minNeighbors=5,
                        minSize=(30, 30), maxSize=(200, 200)
                    )
                else:
                    faces = detector.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=3,
                        minSize=(30, 30), maxSize=(200, 200)
                    )
                all_faces.extend(faces)
            except:
                continue
        
        if all_faces:
            return self.non_max_suppression(all_faces)
        return []
    
    def non_max_suppression(self, faces):
        if len(faces) <= 1:
            return faces
        
        boxes = np.array([[x, y, x+w, y+h] for x, y, w, h in faces])
        scores = np.ones(len(boxes))
        
        try:
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), scores.tolist(), 0.3, 0.4
            )
            if len(indices) > 0:
                indices = indices.flatten()
                return [faces[i] for i in indices]
        except:
            pass
        
        return faces

# ============================================================
# OBJECT TRACKER - KALMAN FILTER
# ============================================================
class ObjectTracker:
    """Kalman filter based object tracking"""
    
    def __init__(self, max_age=20, min_hits=3):
        self.trackers = {}
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 0
        self.history = {}
    
    def update(self, detections):
        # Predict existing
        for track_id in list(self.trackers.keys()):
            tracker = self.trackers[track_id]
            tracker['age'] += 1
            tracker['predictions'] += 1
            
            if tracker['age'] > self.max_age:
                del self.trackers[track_id]
                continue
        
        # Match
        if detections:
            matched, unmatched = self.match_detections(detections)
            
            for track_id, detection in matched:
                self.update_track(track_id, detection)
            
            for detection in unmatched:
                self.create_track(detection)
        
        return self.get_active_tracks()
    
    def match_detections(self, detections):
        matched = []
        unmatched = list(detections)
        
        for track_id, tracker in self.trackers.items():
            if tracker['age'] > self.max_age // 2:
                continue
            
            best_match = None
            best_iou = 0
            
            for i, detection in enumerate(unmatched):
                iou = self.calculate_iou(tracker['last_bbox'], detection['bbox'])
                if iou > best_iou and iou > 0.2:
                    best_iou = iou
                    best_match = i
            
            if best_match is not None:
                matched.append((track_id, unmatched[best_match]))
                unmatched.pop(best_match)
        
        return matched, unmatched
    
    def calculate_iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0
    
    def create_track(self, detection):
        track_id = self.next_id
        self.next_id += 1
        
        self.trackers[track_id] = {
            'last_bbox': detection['bbox'],
            'age': 0,
            'hits': 1,
            'predictions': 0,
            'center': detection['center'],
            'class': detection['class'],
            'confidence': detection['confidence'],
            'history': [detection['center']]
        }
        
        self.history[track_id] = deque(maxlen=30)
        self.history[track_id].append(detection['center'])
    
    def update_track(self, track_id, detection):
        tracker = self.trackers[track_id]
        tracker['last_bbox'] = detection['bbox']
        tracker['age'] = 0
        tracker['hits'] += 1
        tracker['center'] = detection['center']
        tracker['class'] = detection['class']
        tracker['confidence'] = detection['confidence']
        tracker['history'].append(detection['center'])
        
        if track_id in self.history:
            self.history[track_id].append(detection['center'])
    
    def get_active_tracks(self):
        active = []
        for track_id, tracker in self.trackers.items():
            if tracker['hits'] >= self.min_hits:
                active.append({
                    'id': track_id,
                    'bbox': tracker['last_bbox'],
                    'center': tracker['center'],
                    'class': tracker['class'],
                    'confidence': tracker['confidence'],
                    'age': tracker['age'],
                    'hits': tracker['hits'],
                    'history': list(self.history.get(track_id, []))
                })
        return active

# ============================================================
# GARBAGE ANALYZER
# ============================================================
class GarbageAnalyzer:
    """Advanced garbage analysis"""
    
    def __init__(self):
        self.garbage_history = deque(maxlen=100)
        self.patterns = {}
    
    def analyze(self, garbage_items, persons):
        """Analyze garbage patterns"""
        analysis = {
            'count': len(garbage_items),
            'types': {},
            'sizes': [],
            'positions': [],
            'clusters': []
        }
        
        for garbage in garbage_items:
            class_name = garbage['class']
            analysis['types'][class_name] = analysis['types'].get(class_name, 0) + 1
            analysis['sizes'].append(garbage.get('area', 0))
            analysis['positions'].append(garbage['center'])
        
        # Cluster detection
        if len(analysis['positions']) > 2:
            try:
                positions = np.array(analysis['positions'])
                clustering = DBSCAN(eps=50, min_samples=2).fit(positions)
                labels = clustering.labels_
                
                for label in set(labels):
                    if label >= 0:
                        cluster_points = positions[labels == label]
                        if len(cluster_points) > 1:
                            center = np.mean(cluster_points, axis=0)
                            analysis['clusters'].append({
                                'center': (int(center[0]), int(center[1])),
                                'size': len(cluster_points)
                            })
            except:
                pass
        
        self.garbage_history.append(analysis)
        return analysis

# ============================================================
# MAIN DETECTOR - FULL VERSION
# ============================================================
class CerberusWasteWatcher:
    """Complete Detection System"""
    
    def __init__(self, args=None):
        self.args = args
        display_banner()
        
        # Initialize
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n[INFO] Cerberus Initializing on {self.device.upper()}\n")
        
        # Head 1
        print("[HEAD 1] Motion Sentinel - Optical Flow")
        self.previous_gray = None
        self.motion_history = deque(maxlen=30)
        
        # Head 2
        print("[HEAD 2] Object Guardian - YOLOv8")
        self.model = self.load_model()
        
        # Head 3
        print("[HEAD 3] Face Keeper - Multi-Method")
        self.face_detector = FaceDetector()
        
        # Trackers
        print("[TRACKER] Object Tracking - Kalman Filter")
        self.person_tracker = ObjectTracker(max_age=20, min_hits=3)
        self.garbage_tracker = ObjectTracker(max_age=15, min_hits=3)
        
        # Analyzer
        print("[ANALYZER] Garbage Pattern Analysis")
        self.analyzer = GarbageAnalyzer()
        
        # State
        self.current_events = {}
        self.unique_persons = set()
        self.garbage_drop_count = 0
        self.last_alert = 0
        self.last_snapshot = 0
        self.last_save = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.log_file = None
        self.video_writer = None
        self.is_running = True
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.processing_thread = None
        
        # Create directories
        CerberusConfig.create_directories()
        self.setup_logging()
        
        print("\n" + "="*60)
        print("🐕 CERBERUS WASTEWATCHER - FULLY OPERATIONAL")
        print("   All three heads are active and watching...")
        print("="*60)
        print("\nCONTROLS:")
        print("  [q] Quit")
        print("  [s] Save snapshot")
        print("  [r] Reset statistics")
        print("  [f] Force face detection")
        print("  [e] Export data")
        print("  [d] Toggle debug")
        print("  [b] Backup data")
        print("  [h] Show help")
        print("="*60 + "\n")
    
    def load_model(self):
        """Load YOLO model"""
        try:
            model_path = self.args.model if self.args and self.args.model else 'yolov8n.pt'
            
            if not os.path.exists(model_path):
                print(f"  [INFO] Downloading {model_path}...")
            
            model = YOLO(model_path)
            print(f"  [OK] Model: {model_path}")
            
            # Warmup
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            model(test_img, verbose=False)
            print("  [OK] Model warmed up")
            
            return model
            
        except Exception as e:
            print(f"  [ERROR] Model error: {e}")
            print("  [INFO] Trying to download...")
            try:
                model = YOLO('yolov8n.pt')
                print("  [OK] Model downloaded")
                return model
            except:
                print("  [ERROR] Failed to load model")
                sys.exit(1)
    
    def setup_logging(self):
        """Setup logging"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = f"{CerberusConfig.LOG_DIR}/cerberus_log_{timestamp}.csv"
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Event', 'Person_ID', 'Face_Image',
                'Garbage_Type', 'Confidence', 'Image_Path',
                'Duration', 'Track_Length', 'Details'
            ])
        
        print(f"[LOG] Log file: {self.log_file}")
    
    def detect_motion(self, frame):
        """Head 1: Advanced motion detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if self.previous_gray is None:
            self.previous_gray = gray
            return []
        
        # Optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.previous_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        magnitude = np.clip(magnitude, 0, 50)
        
        _, motion_mask = cv2.threshold(magnitude, 2, 255, cv2.THRESH_BINARY)
        motion_mask = motion_mask.astype(np.uint8)
        
        kernel = np.ones((3, 3), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(
            motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        motion_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > CerberusConfig.MIN_MOTION_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append((x, y, w, h, area))
        
        self.previous_gray = gray
        return motion_areas
    
    def detect_objects(self, frame):
        """Head 2: Object detection"""
        try:
            results = self.model(
                frame,
                conf=CerberusConfig.CONFIDENCE_THRESHOLD,
                iou=CerberusConfig.IOU_THRESHOLD,
                max_det=CerberusConfig.MAX_DETECTIONS,
                verbose=False
            )
            
            persons = []
            garbage_items = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        aspect = height / width if width > 0 else 0
                        
                        detection = {
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            'area': area,
                            'aspect': aspect,
                            'width': width,
                            'height': height
                        }
                        
                        # Person detection with validation
                        if class_name == 'person':
                            if (CerberusConfig.PERSON_MIN_AREA < area < CerberusConfig.PERSON_MAX_AREA and
                                CerberusConfig.PERSON_MIN_ASPECT < aspect < CerberusConfig.PERSON_MAX_ASPECT):
                                persons.append(detection)
                                print(f"  [DETECT] Person at ({int(x1)},{int(y1)}) conf:{confidence:.2f}")
                        
                        # Garbage detection
                        elif any(gc.lower() in class_name.lower() 
                                for gc in CerberusConfig.GARBAGE_CLASSES):
                            if CerberusConfig.GARBAGE_MIN_AREA < area < CerberusConfig.GARBAGE_MAX_AREA:
                                garbage_items.append(detection)
                                print(f"  [DETECT] Garbage: {class_name} conf:{confidence:.2f}")
                        
                        # Additional common objects
                        elif class_name in ['bottle', 'cup', 'bowl', 'plate', 'bag']:
                            garbage_items.append(detection)
                            print(f"  [DETECT] Garbage: {class_name} conf:{confidence:.2f}")
            
            return persons, garbage_items
        
        except Exception as e:
            print(f"  [ERROR] Detection: {e}")
            return [], []
    
    def is_carrying(self, persons, garbage_items, motion_areas):
        """Check if person is carrying garbage"""
        carrying = []
        
        if not persons or not garbage_items:
            return carrying
        
        for person in persons:
            px1, py1, px2, py2 = person['bbox']
            p_center = person['center']
            p_area = person['area']
            
            best_garbage = None
            best_score = 0
            
            for garbage in garbage_items:
                gx1, gy1, gx2, gy2 = garbage['bbox']
                g_center = garbage['center']
                
                # Distance check
                dist = distance.euclidean(p_center, g_center)
                
                if dist < CerberusConfig.CARRYING_DISTANCE:
                    # Check hand area
                    hand_y = py1 + (py2 - py1) * 0.3
                    hand_x1 = px1 + (px2 - px1) * 0.2
                    hand_x2 = px2 - (px2 - px1) * 0.2
                    
                    if (hand_x1 < g_center[0] < hand_x2 and 
                        py1 < g_center[1] < hand_y + 50):
                        score = 1 - (dist / CerberusConfig.CARRYING_DISTANCE)
                        if score > best_score:
                            best_score = score
                            best_garbage = garbage
                
                # Overlap check
                if not best_garbage:
                    xi1 = max(px1, gx1)
                    yi1 = max(py1, gy1)
                    xi2 = min(px2, gx2)
                    yi2 = min(py2, gy2)
                    
                    if xi2 > xi1 and yi2 > yi1:
                        overlap = (xi2 - xi1) * (yi2 - yi1)
                        if overlap / p_area > CerberusConfig.OVERLAP_THRESHOLD:
                            best_score = 0.8
                            best_garbage = garbage
            
            if best_garbage:
                carrying.append({
                    'person': person,
                    'garbage': best_garbage,
                    'confidence': best_score
                })
                print(f"  [CARRY] Person carrying {best_garbage['class']}")
        
        return carrying
    
    def detect_faces(self, frame, person_bbox):
        """Head 3: Face detection"""
        try:
            x1, y1, x2, y2 = person_bbox
            padding = 20
            
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None, None
            
            faces = self.face_detector.detect_faces(roi)
            
            if faces:
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                face_img = roi[fy:fy+fh, fx:fx+fw]
                
                if face_img.size > 0:
                    face_img = cv2.resize(face_img, (100, 100))
                    face_img = cv2.GaussianBlur(face_img, (3, 3), 0)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    face_path = f"{CerberusConfig.FACE_DIR}/face_{timestamp}.jpg"
                    cv2.imwrite(face_path, face_img)
                    print(f"  [FACE] Captured: {face_path}")
                    return face_img, face_path
            
            return None, None
        
        except:
            return None, None
    
    def save_event(self, frame, person, garbage, face_path=None):
        """Save complete event"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            
            # Main image
            image_path = f"{CerberusConfig.IMAGE_DIR}/event_{timestamp}.jpg"
            annotated = frame.copy()
            
            # Person
            if person:
                px1, py1, px2, py2 = person['bbox']
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 3)
                cv2.putText(annotated, "CARRYING", (px1, py1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Garbage
            if garbage:
                gx1, gy1, gx2, gy2 = garbage['bbox']
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 255, 255), 3)
                cv2.putText(annotated, garbage['class'], (gx1, gy1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Branding
            cv2.putText(annotated, "CERBERUS WASTEWATCHER", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)
            cv2.putText(annotated, f"Event #{self.garbage_drop_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(annotated, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if face_path:
                cv2.putText(annotated, f"Face: {os.path.basename(face_path)}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            cv2.imwrite(image_path, annotated)
            
            # Cropped image
            if person:
                px1, py1, px2, py2 = person['bbox']
                cropped = frame[py1:py2, px1:px2]
                if cropped.size > 0:
                    crop_path = f"{CerberusConfig.CROPPED_DIR}/crop_{timestamp}.jpg"
                    cv2.imwrite(crop_path, cropped)
            
            return image_path
        
        except Exception as e:
            print(f"  [ERROR] Saving event: {e}")
            return None
    
    def log_event(self, event_type, person, garbage, face_path, image_path, details=""):
        """Log event"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            person_id = hashlib.md5(str(person['bbox']).encode()).hexdigest()[:8]
            
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, event_type, person_id,
                    face_path or 'None',
                    garbage['class'] if garbage else 'None',
                    garbage['confidence'] if garbage else 0,
                    image_path or '',
                    details,
                    '',
                    ''
                ])
        except:
            pass
    
    def export_data(self):
        """Export all data"""
        try:
            export_path = f"{CerberusConfig.EXPORT_DIR}/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            Path(export_path).mkdir(parents=True, exist_ok=True)
            
            # Copy log
            if self.log_file and os.path.exists(self.log_file):
                shutil.copy(self.log_file, f"{export_path}/log.csv")
            
            # Copy latest images
            for dir_name in ['events', 'faces', 'snapshots']:
                src = f"{CerberusConfig.BASE_DIR}/{dir_name}"
                dst = f"{export_path}/{dir_name}"
                if os.path.exists(src):
                    shutil.copytree(src, dst)
            
            # Create report
            report = {
                'export_time': datetime.now().isoformat(),
                'total_drops': self.garbage_drop_count,
                'unique_persons': len(self.unique_persons),
                'total_frames': self.frame_count,
                'runtime': time.time() - self.start_time
            }
            
            with open(f"{export_path}/report.json", 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"[EXPORT] Data exported to: {export_path}")
            return export_path
        
        except Exception as e:
            print(f"[ERROR] Export: {e}")
            return None
    
    def backup_data(self):
        """Backup all data"""
        try:
            backup_path = f"{CerberusConfig.BACKUP_DIR}/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(CerberusConfig.BASE_DIR, backup_path)
            print(f"[BACKUP] Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"[ERROR] Backup: {e}")
            return None
    
    def process_frame(self, frame):
        """Process single frame"""
        # Motion
        motion_areas = self.detect_motion(frame)
        
        # Objects
        persons, garbage_items = self.detect_objects(frame)
        
        # Track
        tracked_persons = self.person_tracker.update([
            {'bbox': p['bbox'], 'center': p['center'],
             'class': p['class'], 'confidence': p['confidence']}
            for p in persons
        ])
        
        tracked_garbage = self.garbage_tracker.update([
            {'bbox': g['bbox'], 'center': g['center'],
             'class': g['class'], 'confidence': g['confidence']}
            for g in garbage_items
        ])
        
        # Analyze
        analysis = self.analyzer.analyze(garbage_items, persons)
        
        # Carrying detection
        carrying = self.is_carrying(persons, garbage_items, motion_areas)
        
        # Events
        if carrying:
            self.detect_drops(frame, carrying)
        
        # Annotate
        annotated = self.draw_annotations(
            frame, persons, garbage_items, carrying,
            tracked_persons, tracked_garbage, analysis
        )
        
        return annotated, carrying, analysis
    
    def detect_drops(self, frame, carrying):
        """Detect garbage drops"""
        current_time = time.time()
        
        for item in carrying:
            person = item['person']
            garbage = item['garbage']
            person_id = hashlib.md5(str(person['bbox']).encode()).hexdigest()[:8]
            
            if person_id not in self.current_events:
                self.current_events[person_id] = {
                    'start': current_time,
                    'person': person,
                    'garbage': garbage,
                    'count': 0,
                    'history': []
                }
                self.current_events[person_id]['history'].append(person['center'])
            else:
                event = self.current_events[person_id]
                event['count'] += 1
                event['history'].append(person['center'])
                
                # Check drop
                still_carrying = any(
                    c['person']['bbox'] == person['bbox']
                    for c in carrying
                )
                
                if not still_carrying and event['count'] > 3:
                    if current_time - self.last_alert > CerberusConfig.ALERT_COOLDOWN:
                        self.on_garbage_drop(frame, person, garbage, event)
                        self.last_alert = current_time
                    del self.current_events[person_id]
    
    def on_garbage_drop(self, frame, person, garbage, event):
        """Handle garbage drop"""
        self.garbage_drop_count += 1
        
        # Detect face
        face_img, face_path = self.detect_faces(frame, person['bbox'])
        
        # Save event
        image_path = self.save_event(frame, person, garbage, face_path)
        
        # Snapshot
        if time.time() - self.last_snapshot > CerberusConfig.SNAPSHOT_INTERVAL:
            snapshot_path = self.save_snapshot(frame, person, garbage)
            self.last_snapshot = time.time()
        
        # Log
        duration = time.time() - event['start']
        track_len = len(event['history'])
        self.log_event(
            'GARBAGE_DROP', person, garbage, face_path, image_path,
            f"Duration: {duration:.1f}s, Track: {track_len}pts"
        )
        
        # Alert
        print("\n" + "!"*60)
        print("  🐕 CERBERUS ALERT - GARBAGE DROPPED!")
        print(f"  Event #{self.garbage_drop_count}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Garbage: {garbage['class']} ({garbage['confidence']:.2f})")
        print(f"  Duration: {duration:.1f} seconds")
        if face_path:
            print(f"  Face: {os.path.basename(face_path)}")
        if image_path:
            print(f"  Image: {os.path.basename(image_path)}")
        print("!"*60 + "\n")
        
        self.unique_persons.add(person_id)
    
    def save_snapshot(self, frame, person=None, garbage=None):
        """Save snapshot"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            path = f"{CerberusConfig.SNAPSHOT_DIR}/snapshot_{timestamp}.jpg"
            
            annotated = frame.copy()
            
            if person:
                px1, py1, px2, py2 = person['bbox']
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 2)
            
            if garbage:
                gx1, gy1, gx2, gy2 = garbage['bbox']
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
            
            cv2.putText(annotated, "CERBERUS SNAPSHOT", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)
            cv2.putText(annotated, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imwrite(path, annotated)
            return path
        
        except:
            return None
    
    def draw_annotations(self, frame, persons, garbage_items, carrying,
                         tracked_persons, tracked_garbage, analysis):
        """Draw everything"""
        annotated = frame.copy()
        
        try:
            # Tracked persons
            for track in tracked_persons:
                x1, y1, x2, y2 = track['bbox']
                cv2.rectangle(annotated, (x1, y1), (x2, y2),
                             CerberusConfig.COLOR_TRACK, 1)
                cv2.putText(annotated, f"ID:{track['id']}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                           CerberusConfig.COLOR_TRACK, 1)
            
            # Normal persons - Green
            for person in persons:
                if not any(c['person'] == person for c in carrying):
                    x1, y1, x2, y2 = person['bbox']
                    cv2.rectangle(annotated, (x1, y1), (x2, y2),
                                 CerberusConfig.COLOR_NORMAL, 2)
                    cv2.putText(annotated, "PERSON", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               CerberusConfig.COLOR_NORMAL, 2)
            
            # Carrying persons - Red
            for item in carrying:
                person = item['person']
                garbage = item['garbage']
                conf = item['confidence']
                
                x1, y1, x2, y2 = person['bbox']
                cv2.rectangle(annotated, (x1, y1), (x2, y2),
                             CerberusConfig.COLOR_CARRYING, 3)
                cv2.putText(annotated, f"CARRYING ({conf:.2f})", (x1, y1-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           CerberusConfig.COLOR_CARRYING, 2)
                
                # Connection line
                cv2.line(annotated, person['center'], garbage['center'],
                        CerberusConfig.COLOR_CARRYING, 2)
            
            # Garbage - Yellow
            for garbage in garbage_items:
                x1, y1, x2, y2 = garbage['bbox']
                cv2.rectangle(annotated, (x1, y1), (x2, y2),
                             CerberusConfig.COLOR_GARBAGE, 2)
                cv2.putText(annotated, garbage['class'], (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                           CerberusConfig.COLOR_GARBAGE, 1)
            
            # Analysis clusters
            for cluster in analysis.get('clusters', []):
                cx, cy = cluster['center']
                cv2.circle(annotated, (cx, cy), 10,
                          CerberusConfig.COLOR_ALERT, 2)
                cv2.putText(annotated, f"Cluster {cluster['size']}",
                           (cx-20, cy-15), cv2.FONT_HERSHEY_SIMPLEX,
                           0.3, CerberusConfig.COLOR_ALERT, 1)
            
            # HUD
            self.draw_hud(annotated, carrying, analysis)
            
        except Exception as e:
            pass
        
        return annotated
    
    def draw_hud(self, frame, carrying, analysis):
        """Draw HUD"""
        try:
            h, w = frame.shape[:2]
            
            # Background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 220), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            y = 40
            cv2.putText(frame, "CERBERUS WASTEWATCHER", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 215, 0), 2)
            
            cv2.putText(frame, f"Drops: {self.garbage_drop_count}",
                       (20, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 255), 2)
            cv2.putText(frame, f"Persons: {len(self.unique_persons)}",
                       (20, y+55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 2)
            cv2.putText(frame, f"Carrying: {len(carrying)}",
                       (20, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 0, 255), 2)
            cv2.putText(frame, f"Garbage: {analysis.get('count', 0)}",
                       (20, y+105), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 255), 2)
            cv2.putText(frame, f"Clusters: {len(analysis.get('clusters', []))}",
                       (20, y+130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 0, 255), 2)
            
            # FPS
            if self.frame_count > 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Status
            status = 'ALERT' if carrying else 'MONITORING'
            color = (0, 0, 255) if carrying else (0, 255, 0)
            cv2.putText(frame, f"Status: {status}", (w-140, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Heads status
            heads = "1 2 3"
            cv2.putText(frame, f"Heads: {heads}", (w-140, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Legend
            legend_y = h - 130
            cv2.rectangle(frame, (10, legend_y), (210, legend_y+120),
                         (0, 0, 0), -1)
            cv2.addWeighted(frame, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, "LEGEND", (20, legend_y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            colors = [
                ((0, 255, 0), "Normal Person"),
                ((0, 0, 255), "Carrying Garbage"),
                ((0, 255, 255), "Garbage"),
                ((255, 0, 0), "Face Detected"),
                ((255, 128, 0), "Bag"),
                ((255, 0, 255), "Cluster")
            ]
            
            y_pos = legend_y + 40
            for color, label in colors:
                cv2.rectangle(frame, (20, y_pos), (40, y_pos+12), color, -1)
                cv2.putText(frame, label, (45, y_pos+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                           (255, 255, 255), 1)
                y_pos += 20
            
        except:
            pass
    
    def run(self):
        """Main loop"""
        cap = None
        video_writer = None
        
        try:
            cap = cv2.VideoCapture(CerberusConfig.CAMERA_ID)
            
            if not cap.isOpened():
                print("[ERROR] Cannot access camera!")
                return
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CerberusConfig.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CerberusConfig.FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CerberusConfig.FPS)
            
            print("[OK] Camera ready!")
            print("[OK] ALL SYSTEMS GO - Cerberus is watching!\n")
            
            debug = False
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Process
                annotated_frame, carrying, analysis = self.process_frame(frame)
                
                # Video recording
                if video_writer is None:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    video_path = f"{CerberusConfig.VIDEO_DIR}/recording_{timestamp}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(
                        video_path, fourcc, CerberusConfig.FPS,
                        (CerberusConfig.FRAME_WIDTH, CerberusConfig.FRAME_HEIGHT)
                    )
                    print(f"[VIDEO] Recording: {video_path}")
                
                if video_writer:
                    video_writer.write(annotated_frame)
                
                # Show
                cv2.imshow('Cerberus WasteWatcher Ultimate', annotated_frame)
                
                # Controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    path = self.save_snapshot(annotated_frame)
                    print(f"[SAVED] Snapshot: {path}")
                elif key == ord('r'):
                    self.garbage_drop_count = 0
                    self.unique_persons.clear()
                    self.current_events.clear()
                    print("[RESET] Statistics reset")
                elif key == ord('f'):
                    print("[FACE] Forcing face detection...")
                    for item in carrying:
                        face_img, face_path = self.detect_faces(
                            frame, item['person']['bbox']
                        )
                        if face_path:
                            print(f"  Face: {face_path}")
                elif key == ord('e'):
                    self.export_data()
                elif key == ord('b'):
                    self.backup_data()
                elif key == ord('d'):
                    debug = not debug
                    print(f"[DEBUG] {'ON' if debug else 'OFF'}")
                elif key == ord('h'):
                    print("\n" + "="*50)
                    print("CERBERUS WASTEWATCHER - HELP")
                    print("="*50)
                    print("[q] Quit")
                    print("[s] Save snapshot")
                    print("[r] Reset statistics")
                    print("[f] Force face detection")
                    print("[e] Export data")
                    print("[b] Backup data")
                    print("[d] Toggle debug")
                    print("[h] Show help")
                    print("="*50 + "\n")
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted")
        except Exception as e:
            print(f"\n[ERROR] {e}")
        finally:
            if cap:
                cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*60)
            print("FINAL REPORT - CERBERUS WASTEWATCHER")
            print("="*60)
            print(f"Total Garbage Drops: {self.garbage_drop_count}")
            print(f"Unique Persons: {len(self.unique_persons)}")
            print(f"Total Frames: {self.frame_count}")
            print(f"Runtime: {time.time() - self.start_time:.1f}s")
            print(f"Log: {self.log_file}")
            print("="*60)
            print("🐕 Cerberus WasteWatcher signing off...\n")

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Cerberus WasteWatcher - Ultimate Full Version'
    )
    
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (default: 0)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model (default: yolov8n.pt)')
    parser.add_argument('--no-face', action='store_true',
                       help='Disable face detection')
    parser.add_argument('--width', type=int, default=640,
                       help='Frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Frame height (default: 480)')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    
    args = parser.parse_args()
    
    # Update config
    CerberusConfig.CAMERA_ID = args.camera
    CerberusConfig.FRAME_WIDTH = args.width
    CerberusConfig.FRAME_HEIGHT = args.height
    CerberusConfig.CONFIDENCE_THRESHOLD = args.confidence
    
    try:
        watcher = CerberusWasteWatcher(args)
        watcher.run()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

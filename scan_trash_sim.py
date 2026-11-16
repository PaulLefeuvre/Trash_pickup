import argparse
import math
import os
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
import lerobot.robots.so_follower as so_follower
from ultralytics import YOLO

from NewArmHoldController import (
    ArmHoldController,
    HOME_POSE,
    PICK_AND_DROP_SEQUENCE,
    STOW_HOLD_POSE,
)

# ==============================================================================
# 1. TARGET PROMPTS, BIN ROUTING & SWEEP PARAMETERS
# ==============================================================================
PROMPT_TO_CANONICAL = {
    "plastic bottle": "plastic bottle",
    "transparent water bottle": "plastic bottle",
    "clear bottle": "plastic bottle",
    "aluminium can": "aluminium can",
    "drink can": "aluminium can",
    "soda can": "aluminium can",
}

LABEL_TO_BIN = {
    "plastic bottle": 0,
    "aluminium can": 1,
}
DEFAULT_BIN_INDEX = 0

# Homing pose moved to on program stop (Ctrl+C, Q, ESC) before motors disconnect.
CUSTOM_HOME_POSE = {
    **HOME_POSE,
}
HOMING_DURATION_SEC = 1.5        # Smooth interpolation duration to CUSTOM_HOME_POSE on exit

# Smooth pan sweep & auto-pick parameters (tuned for zero-overshoot settle)
CONTROL_FPS = 30
PAN_SIGN = 1.0                   # Flip between -1.0 and +1.0 if pan rotates away from target
PAN_KP = 0.95                    # Proportional gain (deg/s per degree of visual error)
PAN_KD = 0.18                    # Derivative damping to prevent overshoot on approach
SWEEP_MAX_SPEED_DEG_S = 15.0     # Maximum pan sweep speed (deg/s)
SWEEP_MAX_ACCEL_DEG_S2 = 45.0    # Smooth ramp-up acceleration (deg/s^2)
SWEEP_MAX_DECEL_DEG_S2 = 120.0   # Fast braking deceleration when entering deadband (deg/s^2)
PAN_DEADBAND_DEG = 2.0           # Stop pan motor inside +/-2.0 deg (must be <= ANGLE_ZERO_TOL_DEG)
ANGLE_ZERO_TOL_DEG = 2.5         # Alignment window around 0.0 deg to confirm pick
ALIGN_CONFIRM_SEC = 0.5          # Duration |angle| must stay <= ANGLE_ZERO_TOL_DEG before picking
ALIGN_JITTER_GRACE_SEC = 0.15    # Tolerates brief 1-2 frame bbox jitter without resetting timer
LOST_RETURN_WAIT_SEC = 4.0       # Wait time (seconds) before returning to default shoulder_pan when lost
POST_PICK_COOLDOWN_SEC = 0.8     # Settling pause after returning to stow before scanning again

# ==============================================================================
# 2. WORKSPACE & DETECTOR PARAMETERS
# ==============================================================================
WEIGHTS_PATH = "yolov8n_taco.pt"
CONF_THRESH = 0.28
IMGSZ = 512
TRACK_SCALE = 0.5
MIN_BOX_AREA = 1600.0
MIN_BOX_SIDE = 25.0
USE_CLAHE = False

# Large red circle outline (shoulder pan path)
PAN_CIRCLE_CENTER = (334, 374)   # (x, y) pixel center of the shoulder pan arc
PAN_CIRCLE_RADIUS = 143          # Radius in pixels
PAN_CIRCLE_THICKNESS = 2         # Positive integer = outline only
PAN_CIRCLE_COLOR = (0, 0, 255)   # BGR: Red

# Radial distance tolerance from the red circle circumference (in pixels).
MAX_CIRCLE_DIST_PX = 70.0

# Thin vertical centerline & movable dot along that line
CENTER_LINE_X = 334              # None = exact horizontal middle (w // 2), or int
CENTER_LINE_THICKNESS = 1
CENTER_LINE_COLOR = (220, 220, 220)

LINE_DOT_Y = 231                 # Vertical Y position of the dot along the line
LINE_DOT_RADIUS = 5
LINE_DOT_COLOR = (255, 255, 0)   # BGR: Cyan


# ==============================================================================
# 3. SO-101 ROBOT CONNECTION & SIMULATION FALLBACK
# ==============================================================================
class SimulatedArmHoldController:
    """Drop-in dry-run controller when no physical SO-101 arm is connected."""

    def __init__(self, hold_pose: dict, sweep_max_speed: float = SWEEP_MAX_SPEED_DEG_S, lost_grace_sec: float = LOST_RETURN_WAIT_SEC):
        self._stow_pan = float(hold_pose.get("shoulder_pan", 0.0))
        self.current_pan = self._stow_pan
        self._sweep_max_speed = sweep_max_speed
        self._lost_grace_sec = lost_grace_sec
        self._busy_until = 0.0
        self._lost_since: Optional[float] = None
        self._last_update = time.perf_counter()

    def start(self) -> None:
        print("[SIM] Running in vision-only simulation mode (no SO-101 connected).")

    @property
    def is_busy(self) -> bool:
        return time.perf_counter() < self._busy_until

    @property
    def is_aligned(self) -> bool:
        return not self.is_busy

    @property
    def lost_elapsed_sec(self) -> Optional[float]:
        if self._lost_since is None:
            return None
        return time.perf_counter() - self._lost_since

    def update_visual_angle(self, angle_rad: Optional[float]) -> None:
        now = time.perf_counter()
        dt = min(now - self._last_update, 0.1)
        self._last_update = now

        if self.is_busy:
            return

        if angle_rad is None:
            if self._lost_since is None:
                self._lost_since = now
            elif (now - self._lost_since) >= self._lost_grace_sec:
                err = self._stow_pan - self.current_pan
                step = max(-self._sweep_max_speed * dt, min(self._sweep_max_speed * dt, err))
                self.current_pan += step
            return

        self._lost_since = None
        err_deg = math.degrees(angle_rad)
        if abs(err_deg) > PAN_DEADBAND_DEG:
            vel = max(-self._sweep_max_speed, min(self._sweep_max_speed, PAN_SIGN * PAN_KP * err_deg))
            self.current_pan += vel * dt

    def reset_to_stow_pan(self, wait: bool = False) -> None:
        self.current_pan = self._stow_pan
        self._lost_since = None

    def send_trajectory(
        self,
        sequence,
        bin_index: int = 0,
        pan_angle: float = 0.0,
        return_to_hold: bool = True,
        wait: bool = False,
    ) -> None:
        print(f"[SIM] Simulated pick-and-drop -> bin={bin_index}, pan={pan_angle:+.1f} deg")
        self._busy_until = time.perf_counter() + 2.0
        if return_to_hold:
            self.current_pan = self._stow_pan

    def shutdown_to_home(self, home_pose: dict, duration: float = 1.5) -> None:
        print("[SIM] Shutdown complete (no hardware to home).")


def connect_so101(port: str, robot_id: str = "robodog_follower", force_sim: bool = False):
    """Connects to an SO-101 follower arm, returning None if unavailable."""
    if force_sim:
        print("[INFO] --sim enabled: skipping SO-101 hardware connection.")
        return None

    try:
        if hasattr(so_follower, "SO101Follower") and hasattr(so_follower, "SO101FollowerConfig"):
            config = so_follower.SO101FollowerConfig(port=port, id=robot_id)
            robot = so_follower.SO101Follower(config)
        else:
            config = so_follower.SOFollowerConfig(port=port, id=robot_id)
            robot = so_follower.SOFollower(config)

        robot.connect()
        print(f"[INFO] Connected to SO-101 on {port}.")
        return robot
    except Exception as exc:
        print(f"[WARN] Could not connect to SO-101 on {port!r} ({exc}).")
        print("[WARN] Continuing without robot hardware (simulation mode).")
        return None


# ==============================================================================
# 4. MODEL, CAMERA, AND TRACKER HELPERS
# ==============================================================================
def load_detector(weights_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TACO] Inference device: {device.upper()}")

    if os.path.exists(weights_path):
        print(f"[TACO] Loading fine-tuned weights: {weights_path}")
        return YOLO(weights_path), False, device

    prompts = list(PROMPT_TO_CANONICAL.keys())
    print("[TACO] Loading yolov8s-worldv2.pt for single-object precision tracking...")
    model = YOLO("yolov8s-worldv2.pt")
    model.set_classes(prompts)
    return model, True, device


def create_csrt_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT not found. Install: pip install --upgrade opencv-contrib-python")


def open_source(src: str, width: int = 640, height: int = 480, fps: int = CONTROL_FPS) -> cv2.VideoCapture:
    if src.isdigit():
        if sys.platform.startswith("linux"):
            cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/video source: {src!r}")
    return cap


def preprocess_for_transparent(frame_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_eq, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def is_valid_box_size(xyxy: np.ndarray, min_area: float, min_side: float) -> bool:
    bw = max(0.0, float(xyxy[2] - xyxy[0]))
    bh = max(0.0, float(xyxy[3] - xyxy[1]))
    return bw >= min_side and bh >= min_side and (bw * bh) >= min_area


def distance_from_circle_line(u: float, v: float) -> float:
    """Returns radial pixel distance between point (u, v) and the pan circle circumference."""
    radial_dist = math.hypot(u - PAN_CIRCLE_CENTER[0], v - PAN_CIRCLE_CENTER[1])
    return abs(radial_dist - PAN_CIRCLE_RADIUS)


def is_near_circle_line(xyxy: np.ndarray, max_circle_dist: float) -> bool:
    u = 0.5 * (float(xyxy[0]) + float(xyxy[2]))
    v = 0.5 * (float(xyxy[1]) + float(xyxy[3]))
    return distance_from_circle_line(u, v) <= max_circle_dist


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def xyxy_to_xywh_clamped(xyxy: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    x = max(0, min(int(round(xyxy[0])), w - 4))
    y = max(0, min(int(round(xyxy[1])), h - 4))
    bw = max(4, min(int(round(xyxy[2] - xyxy[0])), w - x))
    bh = max(4, min(int(round(xyxy[3] - xyxy[1])), h - y))
    return (x, y, bw, bh)


# ==============================================================================
# 5. ASYNCHRONOUS YOLOV8S WORKER THREAD
# ==============================================================================
class AsyncYoloWorker:
    def __init__(
        self,
        model,
        device: str,
        conf_thresh: float,
        imgsz: int,
        map_synonyms: bool,
        min_area: float,
        min_side: float,
    ):
        self.model = model
        self.device = device
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.map_synonyms = map_synonyms
        self.min_area = min_area
        self.min_side = min_side

        self._lock = threading.Lock()
        self._pending_frame: np.ndarray | None = None
        self._latest_dets: list[dict] | None = None
        self._ready = False
        self._running = True
        self.infer_ms = 0.0

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def submit(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            if self._pending_frame is None:
                self._pending_frame = frame_bgr.copy()

    def poll(self) -> list[dict] | None:
        with self._lock:
            if not self._ready:
                return None
            self._ready = False
            return self._latest_dets

    def _worker_loop(self) -> None:
        while self._running:
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                time.sleep(0.002)
                continue

            t0 = time.perf_counter()
            results = self.model.predict(
                frame,
                conf=self.conf_thresh,
                imgsz=self.imgsz,
                iou=0.45,
                device=self.device,
                verbose=False,
            )
            self.infer_ms = (time.perf_counter() - t0) * 1000.0

            boxes = results[0].boxes
            detections = []
            if boxes is not None and len(boxes) > 0:
                xyxy_arr = boxes.xyxy.cpu().numpy().astype(float)
                conf_arr = boxes.conf.cpu().numpy()
                cls_arr = boxes.cls.cpu().numpy().astype(int)
                for xyxy, conf, cls_id in zip(xyxy_arr, conf_arr, cls_arr):
                    if not is_valid_box_size(xyxy, self.min_area, self.min_side):
                        continue
                    raw_label = str(self.model.names[cls_id])
                    norm_key = raw_label.strip().lower()
                    label = PROMPT_TO_CANONICAL.get(norm_key, norm_key if self.map_synonyms else raw_label)
                    detections.append({"label": label, "conf": float(conf), "xyxy": xyxy})

            with self._lock:
                self._latest_dets = detections
                self._ready = True

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)


# ==============================================================================
# 6. SINGLE-TARGET PRECISION CSRT TRACKER (WITH CIRCLE-LINE FILTER)
# ==============================================================================
@dataclass
class TargetLock:
    tracker: object
    bbox_xyxy: np.ndarray
    label_history: deque = field(default_factory=lambda: deque(maxlen=15))
    conf_ema: float = 0.0
    hits: int = 1
    misses: int = 0

    @property
    def stable_label(self) -> str:
        if not self.label_history:
            return "unknown"
        return Counter(self.label_history).most_common(1)[0][0]

    @property
    def center_uv(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return int(round(0.5 * (x1 + x2))), int(round(0.5 * (y1 + y2)))


class SingleObjectPrecisionTracker:
    def __init__(
        self,
        max_circle_dist: float = MAX_CIRCLE_DIST_PX,
        iou_thresh: float = 0.30,
        reinit_iou: float = 0.72,
        min_hits: int = 2,
        max_misses: int = 6,
        ema_alpha: float = 0.65,
        min_area: float = MIN_BOX_AREA,
        min_side: float = MIN_BOX_SIDE,
    ):
        self.max_circle_dist = max_circle_dist
        self.iou_thresh = iou_thresh
        self.reinit_iou = reinit_iou
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.ema_alpha = ema_alpha
        self.min_area = min_area
        self.min_side = min_side
        self.target: TargetLock | None = None

    def predict_csrt(self, track_frame: np.ndarray, scale: float, orig_w: int, orig_h: int) -> None:
        if self.target is None:
            return

        ok, xywh = self.target.tracker.update(track_frame)
        if ok:
            x, y, bw, bh = xywh
            pred_xyxy = np.array([x, y, x + bw, y + bh], dtype=float) / scale
            pred_xyxy[0::2] = np.clip(pred_xyxy[0::2], 0, orig_w - 1)
            pred_xyxy[1::2] = np.clip(pred_xyxy[1::2], 0, orig_h - 1)

            if not is_valid_box_size(pred_xyxy, self.min_area, self.min_side) or not is_near_circle_line(
                pred_xyxy, self.max_circle_dist
            ):
                self.target = None
                return

            self.target.bbox_xyxy = (
                self.ema_alpha * pred_xyxy + (1.0 - self.ema_alpha) * self.target.bbox_xyxy
            )
        else:
            self.target.misses += 1
            if self.target.misses > self.max_misses:
                self.target = None

    def update_with_detections(self, track_frame: np.ndarray, scale: float, detections: list[dict]) -> None:
        th, tw = track_frame.shape[:2]

        # Ignore any detections too far from the red circle line
        valid_dets = [d for d in detections if is_near_circle_line(d["xyxy"], self.max_circle_dist)]

        if self.target is not None:
            best_det = None
            best_iou = 0.0
            for det in valid_dets:
                iou = compute_iou(self.target.bbox_xyxy, det["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            if best_det is not None and best_iou >= self.iou_thresh:
                self.target.bbox_xyxy = (
                    self.ema_alpha * best_det["xyxy"] + (1.0 - self.ema_alpha) * self.target.bbox_xyxy
                )
                self.target.label_history.append(best_det["label"])
                self.target.conf_ema = 0.6 * best_det["conf"] + 0.4 * self.target.conf_ema
                self.target.hits += 1
                self.target.misses = 0

                if best_iou < self.reinit_iou:
                    csrt = create_csrt_tracker()
                    csrt.init(track_frame, xyxy_to_xywh_clamped(self.target.bbox_xyxy * scale, tw, th))
                    self.target.tracker = csrt
                return

            self.target.misses += 1
            if self.target.misses <= self.max_misses:
                return
            self.target = None

        if valid_dets:
            top = max(valid_dets, key=lambda d: d["conf"])
            csrt = create_csrt_tracker()
            csrt.init(track_frame, xyxy_to_xywh_clamped(top["xyxy"] * scale, tw, th))
            lock = TargetLock(
                tracker=csrt,
                bbox_xyxy=top["xyxy"].copy(),
                conf_ema=top["conf"],
            )
            lock.label_history.append(top["label"])
            self.target = lock

    def confirmed_target(self) -> TargetLock | None:
        if self.target is not None and self.target.hits >= self.min_hits:
            u, v = self.target.center_uv
            if distance_from_circle_line(u, v) <= self.max_circle_dist:
                return self.target
        return None


# ==============================================================================
# 7. MAIN LOOP
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 Stow-Hold Visual Pan Aligner & Auto-Picker")
    parser.add_argument("--port", default="/dev/ttyACM0", help="SO-101 serial port")
    parser.add_argument("--source", default="0", help="Camera index or video file path")
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode without connecting to SO-101")
    args = parser.parse_args()

    _ = create_csrt_tracker()
    model, map_synonyms, device = load_detector(WEIGHTS_PATH)
    cap = open_source(args.source, fps=CONTROL_FPS)

    robot = connect_so101(args.port, force_sim=args.sim)
    if robot is not None:
        arm = ArmHoldController(
            robot=robot,
            hold_pose=STOW_HOLD_POSE,
            home_pose=CUSTOM_HOME_POSE,
            fps=CONTROL_FPS,
            pan_sign=PAN_SIGN,
            pan_kp=PAN_KP,
            pan_kd=PAN_KD,
            sweep_max_speed=SWEEP_MAX_SPEED_DEG_S,
            sweep_max_accel=SWEEP_MAX_ACCEL_DEG_S2,
            sweep_max_decel=SWEEP_MAX_DECEL_DEG_S2,
            pan_deadband_deg=PAN_DEADBAND_DEG,
            angle_zero_tol_deg=ANGLE_ZERO_TOL_DEG,
            lost_grace_sec=LOST_RETURN_WAIT_SEC,
        )
    else:
        arm = SimulatedArmHoldController(
            hold_pose=STOW_HOLD_POSE,
            sweep_max_speed=SWEEP_MAX_SPEED_DEG_S,
            lost_grace_sec=LOST_RETURN_WAIT_SEC,
        )
    arm.start()

    worker = AsyncYoloWorker(
        model=model,
        device=device,
        conf_thresh=CONF_THRESH,
        imgsz=IMGSZ,
        map_synonyms=map_synonyms,
        min_area=MIN_BOX_AREA,
        min_side=MIN_BOX_SIDE,
    )
    tracker = SingleObjectPrecisionTracker(max_circle_dist=MAX_CIRCLE_DIST_PX)

    prev_time = time.perf_counter()
    fps_smooth = 0.0
    aligned_since: Optional[float] = None
    last_aligned_time: float = 0.0
    cooldown_until: float = 0.0
    was_busy: bool = False
    mode_prefix = "" if robot is not None else "[SIM] "

    win_name = "SO-101 Stow-Hold Pan Aligner & Auto-Picker"
    print("Running... Controls: [P] pick now | [R] reset pan now | [Q]/[ESC]/[Ctrl+C] move to HOME_POSE & quit.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print(f"[ERROR] Camera read failed on source={args.source!r}.")
                break

            now = time.perf_counter()
            h, w = frame_bgr.shape[:2]
            proc_frame = preprocess_for_transparent(frame_bgr) if USE_CLAHE else frame_bgr

            # Draw workspace guides & circle-line acceptance band
            line_x = int(CENTER_LINE_X) if CENTER_LINE_X is not None else w // 2
            cx, cy = int(PAN_CIRCLE_CENTER[0]), int(PAN_CIRCLE_CENTER[1])
            cv2.circle(frame_bgr, (cx, cy), int(PAN_CIRCLE_RADIUS), PAN_CIRCLE_COLOR, PAN_CIRCLE_THICKNESS)
            cv2.circle(frame_bgr, (cx, cy), max(1, int(PAN_CIRCLE_RADIUS - MAX_CIRCLE_DIST_PX)), (80, 80, 160), 1)
            cv2.circle(frame_bgr, (cx, cy), int(PAN_CIRCLE_RADIUS + MAX_CIRCLE_DIST_PX), (80, 80, 160), 1)
            cv2.line(frame_bgr, (line_x, 0), (line_x, h), CENTER_LINE_COLOR, CENTER_LINE_THICKNESS)
            cv2.circle(frame_bgr, (line_x, int(LINE_DOT_Y)), int(LINE_DOT_RADIUS), LINE_DOT_COLOR, -1)

            force_pick_now = False
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                tracker.target = None
                aligned_since = None
                arm.reset_to_stow_pan(wait=False)
                print("[INFO] Lock cleared; returning pan to STOW_HOLD_POSE.")
            if key == ord("p"):
                force_pick_now = True

            if arm.is_busy:
                was_busy = True
                aligned_since = None
                tracker.target = None
                arm.update_visual_angle(None)
                _ = worker.poll()
                target_hud = "EXECUTING PICK & DROP..."
            elif now < cooldown_until:
                aligned_since = None
                tracker.target = None
                arm.update_visual_angle(None)
                _ = worker.poll()
                target_hud = f"SETTLING ({cooldown_until - now:.1f}s)"
            else:
                if was_busy:
                    was_busy = False
                    cooldown_until = now + POST_PICK_COOLDOWN_SEC
                    aligned_since = None
                    tracker.target = None
                    arm.update_visual_angle(None)
                    target_hud = f"SETTLING ({POST_PICK_COOLDOWN_SEC:.1f}s)"
                else:
                    worker.submit(proc_frame)

                    if TRACK_SCALE != 1.0:
                        track_frame = cv2.resize(
                            proc_frame,
                            (int(round(w * TRACK_SCALE)), int(round(h * TRACK_SCALE))),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        track_frame = proc_frame

                    tracker.predict_csrt(track_frame, TRACK_SCALE, w, h)

                    dets = worker.poll()
                    if dets is not None:
                        tracker.update_with_detections(track_frame, TRACK_SCALE, dets)

                    locked = tracker.confirmed_target()
                    if locked is not None:
                        x1, y1, x2, y2 = [int(round(c)) for c in locked.bbox_xyxy]
                        u, v = locked.center_uv

                        du = u - PAN_CIRCLE_CENTER[0]
                        dv = PAN_CIRCLE_CENTER[1] - v
                        angle = 0.0 if (du == 0 and dv == 0) else math.atan2(du, dv)
                        angle_deg = math.degrees(angle)

                        arm.update_visual_angle(angle)

                        within_tol = abs(angle_deg) <= ANGLE_ZERO_TOL_DEG and arm.is_aligned
                        if within_tol:
                            if aligned_since is None:
                                aligned_since = now
                            last_aligned_time = now
                            hold_elapsed = now - aligned_since
                        elif aligned_since is not None and (now - last_aligned_time) <= ALIGN_JITTER_GRACE_SEC:
                            hold_elapsed = now - aligned_since
                        else:
                            aligned_since = None
                            hold_elapsed = 0.0

                        is_coasting = locked.misses > 0
                        if within_tol:
                            color = (255, 200, 0)
                            tag = f"ALIGNED {hold_elapsed:.2f}/{ALIGN_CONFIRM_SEC:.2f}s | pan: {arm.current_pan:+.1f}"
                        else:
                            color = (0, 200, 255) if is_coasting else (0, 220, 0)
                            tag = f"angle: {angle_deg:+.1f} deg | pan: {arm.current_pan:+.1f}"

                        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                        cv2.drawMarker(frame_bgr, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
                        cv2.putText(frame_bgr, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                        cv2.line(frame_bgr, PAN_CIRCLE_CENTER, (u, v), (0, 255, 0), 2)

                        if (within_tol and hold_elapsed >= ALIGN_CONFIRM_SEC) or force_pick_now:
                            pick_pan = arm.current_pan
                            target_label = locked.stable_label
                            bin_idx = LABEL_TO_BIN.get(target_label, DEFAULT_BIN_INDEX)

                            print(
                                f"[AUTO-PICK] Confirmed '{target_label}' "
                                f"(err={angle_deg:+.2f} deg, pan={pick_pan:+.2f} deg) -> Bin {bin_idx}"
                            )
                            arm.send_trajectory(
                                PICK_AND_DROP_SEQUENCE,
                                bin_index=bin_idx,
                                pan_angle=pick_pan,
                                return_to_hold=True,
                                wait=False,
                            )
                            tracker.target = None
                            arm.update_visual_angle(None)
                            aligned_since = None
                            was_busy = True
                            target_hud = f"PICKING {target_label} -> BIN {bin_idx}"
                        elif within_tol:
                            target_hud = f"CONFIRMING: {locked.stable_label} ({hold_elapsed:.1f}/{ALIGN_CONFIRM_SEC:.1f}s)"
                        else:
                            target_hud = f"SWEEPING: {locked.stable_label} (err={angle_deg:+.1f} deg)"
                    else:
                        aligned_since = None
                        arm.update_visual_angle(None)
                        lost_elapsed = arm.lost_elapsed_sec
                        if lost_elapsed is not None and lost_elapsed < LOST_RETURN_WAIT_SEC:
                            rem = LOST_RETURN_WAIT_SEC - lost_elapsed
                            target_hud = f"LOST - HOLDING PAN ({rem:.1f}s, pan={arm.current_pan:+.1f})"
                        else:
                            target_hud = f"SEARCHING (pan={arm.current_pan:+.1f} deg)"

            inst_fps = 1.0 / max(now - prev_time, 1e-6)
            fps_smooth = inst_fps if fps_smooth == 0.0 else (0.85 * fps_smooth + 0.15 * inst_fps)
            prev_time = now

            hud = f"{mode_prefix}FPS: {fps_smooth:4.1f} | YOLO: {worker.infer_ms:4.0f} ms | {target_hud}"
            cv2.putText(frame_bgr, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)

            cv2.imshow(win_name, frame_bgr)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected — shutting down safely to HOME_POSE...")
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        arm.shutdown_to_home(home_pose=CUSTOM_HOME_POSE, duration=HOMING_DURATION_SEC)
        if robot is not None:
            try:
                robot.disconnect()
                print("[INFO] Arm homed and disconnected.")
            except Exception as exc:
                print(f"[WARN] Error during robot disconnect: {exc}")


if __name__ == "__main__":
    main()
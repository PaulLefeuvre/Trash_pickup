"""Capture images from the arm camera for Roboflow labeling / YOLO11 fine-tuning.

Keys (focus the preview window):
    p      save frame to <out>/plastic/
    c      save frame to <out>/can/
    u      undo (delete) the last saved image
    q/ESC  quit

Saved images are the raw camera frames (no overlay), named
<class>_<timestamp>.jpg so they stay unique across sessions.

Usage:
    python record_dataset.py                 # auto-detects the arm camera
    python record_dataset.py --source 4      # or force a /dev/video index
"""

import argparse
import os
import sys
import time

import cv2

# Arm (wrist) camera USB name. The laptop webcam is "USB2.0 HD UVC WebCam",
# so we look the arm camera up by name instead of trusting /dev/videoN order.
ARM_CAMERA_NAME = "USB2.0_CAM1"

CLASS_KEYS = {
    ord("p"): "plastic",
    ord("c"): "can",
}


def find_arm_camera(name: str = ARM_CAMERA_NAME) -> str:
    """Return /dev/videoN of the arm camera's capture node (looked up by USB name)."""
    by_id = "/dev/v4l/by-id"
    if os.path.isdir(by_id):
        for link in sorted(os.listdir(by_id)):
            if name in link and link.endswith("video-index0"):
                return os.path.realpath(os.path.join(by_id, link))
    raise RuntimeError(
        f"Arm camera {name!r} not found. Is it plugged in? Check: ls -l /dev/v4l/by-id/"
    )


def open_source(src: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if src == "arm":
        src = find_arm_camera()
        print(f"Using arm camera: {src}")
    is_device = src.isdigit() or src.startswith("/dev/video")
    if is_device:
        dev = int(src) if src.isdigit() else src
        if sys.platform.startswith("linux"):
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(dev)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/video source: {src!r}")
    return cap


def count_images(folder: str) -> int:
    return sum(1 for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Record plastic/can images for YOLO dataset")
    parser.add_argument("--source", default="arm",
                        help="'arm' (auto-detect arm camera, default), camera index, /dev/videoN, or video file")
    parser.add_argument("--out", default="dataset_raw", help="Output root folder")
    # 1920x1080 is the arm camera's native mode and shows the full field of view.
    # Lower modes (e.g. 640x480) are a center crop of the sensor -> look "zoomed in".
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--preview-width", type=int, default=960, help="Preview window width (display only)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (0-100)")
    args = parser.parse_args()

    dirs = {}
    for name in CLASS_KEYS.values():
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        dirs[name] = d
    counts = {name: count_images(d) for name, d in dirs.items()}

    cap = open_source(args.source, args.width, args.height, args.fps)
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"Capture resolution: {actual[0]}x{actual[1]} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
    win = "Dataset recorder  [p]=plastic  [c]=can  [u]=undo  [q]=quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)  # 1:1 pixels, no window-driven crop/zoom

    last_saved = []          # stack of saved paths for undo
    flash_text, flash_until = "", 0.0

    print(f"Saving to {os.path.abspath(args.out)}  (existing: {counts})")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame grab failed, stopping.")
                break

            ph = int(frame.shape[0] * args.preview_width / frame.shape[1])
            preview = cv2.resize(frame, (args.preview_width, ph), interpolation=cv2.INTER_AREA)
            hud = "  ".join(f"{n}: {c}" for n, c in counts.items())
            cv2.putText(preview, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(preview, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if time.time() < flash_until:
                h, w = preview.shape[:2]
                cv2.rectangle(preview, (0, 0), (w - 1, h - 1), (0, 255, 0), 8)
                cv2.putText(preview, flash_text, (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(win, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if key in CLASS_KEYS:
                name = CLASS_KEYS[key]
                stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
                path = os.path.join(dirs[name], f"{name}_{stamp}.jpg")
                if cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality]):
                    counts[name] += 1
                    last_saved.append((name, path))
                    flash_text, flash_until = f"saved {name} #{counts[name]}", time.time() + 0.4
                    print(f"[{name}] {path}")
                else:
                    print(f"Failed to write {path}")

            elif key == ord("u") and last_saved:
                name, path = last_saved.pop()
                if os.path.exists(path):
                    os.remove(path)
                    counts[name] -= 1
                    flash_text, flash_until = f"deleted {os.path.basename(path)}", time.time() + 0.8
                    print(f"[undo] removed {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Done. Totals: {counts}")


if __name__ == "__main__":
    main()

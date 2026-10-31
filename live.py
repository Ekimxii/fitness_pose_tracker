"""Count reps live from a webcam, and save the recording.

Run it from the project folder with the .venv active:
    python live.py                      # default webcam, squats
    python live.py --camera 1           # another camera, e.g. a phone used as a webcam
    python live.py --video "Videos for Testing/5 squats.mov"   # a recording instead of the camera

1. A preview window opens. Stand where the camera sees your whole body from the side.
2. Click Start (or press S). After a short countdown to get into position, reps are counted live.
3. Click Terminate (or press T) to stop. Saved to "Video Output": the video with the skeleton and
   rep count drawn on, the plain camera video, and a table of the reps.
Press Q or Esc in the preview to quit without recording.
"""
import argparse
import math
import queue
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import kagglehub
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from features import angles_from_points, normalize
from reps import EXERCISES, LiveRepCounter, load_training_data

HERE = Path(__file__).resolve().parent
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_{0}/float16/latest/pose_landmarker_{0}.task")
CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
WINDOW = "Rep counter"
WHITE, BLACK, BLUE, GREEN, RED = (255, 255, 255), (0, 0, 0), (214, 120, 42), (60, 150, 60), (50, 50, 200)  # BGR


def frame_features(landmarks, width, height):
    """One frame's landmarks -> its 12 angles and 33 visibilities, scaled like the Kaggle data (as in pose-detection)."""
    pixels = np.array([[[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks]])
    return angles_from_points(normalize(pixels))[0], np.array([lm.visibility for lm in landmarks])


def text_scale(img):
    return max(0.6, img.shape[0] / 1200)   # bigger text on bigger (e.g. vertical phone) frames


def put_text(img, text, line, size=1.0):
    """White text with a black outline on the given line (0 = top) of the frame."""
    s = text_scale(img) * size
    org = (int(15 * text_scale(img)), int((50 + 45 * line) * text_scale(img)))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, BLACK, max(2, int(5 * s)), cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, WHITE, max(1, int(2 * s)), cv2.LINE_AA)


def draw_skeleton(img, landmarks):
    h, w = img.shape[:2]
    xy = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    seen = [lm.visibility > 0.5 for lm in landmarks]
    for a, b in CONNECTIONS:
        if seen[a] and seen[b]:
            cv2.line(img, xy[a], xy[b], WHITE, 2, cv2.LINE_AA)
    for point, visible in zip(xy, seen):
        if visible:
            cv2.circle(img, point, 4, BLUE, -1, cv2.LINE_AA)


class CameraReader:
    """Reads the camera on its own thread and keeps only the newest frame, so the picture never lags behind."""

    def __init__(self, cap):
        self.cap, self.ok, self.latest = cap, True, None
        self.fresh = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.ok:
            ok, frame = self.cap.read()
            self.latest, self.ok = (frame, time.monotonic()), ok
            self.fresh.set()

    def read(self):
        """The newest frame not yet returned, and when it was captured. (None, None) if the camera stopped."""
        if not self.fresh.wait(timeout=2) or not self.ok:
            return None, None
        self.fresh.clear()
        return self.latest

    def release(self):
        self.ok = False
        self.thread.join(timeout=2)
        self.cap.release()


class BackgroundWriter:
    """Saves video frames on its own thread, so writing the file doesn't slow down counting."""

    def __init__(self, path, fps, size):
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            raise SystemExit(f"Could not create {path}")
        self.frames = queue.Queue(maxsize=120)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while (frame := self.frames.get()) is not None:
            self.writer.write(frame)

    def write(self, frame):
        self.frames.put(frame)

    def release(self):
        self.frames.put(None)
        self.thread.join()
        self.writer.release()


class Session:
    """Pose detection, rep counting, drawing and recording for one run of the app."""

    def __init__(self, landmarker, train, exercise, output_dir, save_raw):
        self.landmarker, self.train, self.exercise = landmarker, train, exercise
        self.output_dir, self.save_raw = output_dir, save_raw
        self.counter = self.start_t = self.count_from = None
        self.writers, self.paths, self.frames_written, self.last_ms = {}, {}, 0, -1
        self.process_times = deque(maxlen=300)

    @property
    def recording(self):
        return self.start_t is not None

    def process(self, frame, t):
        """Find the pose in one frame, count, and record. Returns the annotated frame (what gets saved)."""
        began = time.perf_counter()
        h, w = frame.shape[:2]
        self.last_ms = max(int(t * 1000), self.last_ms + 1)   # MediaPipe needs increasing timestamps
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect_for_video(image, self.last_ms)
        landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        out = frame.copy()
        if landmarks:
            draw_skeleton(out, landmarks)
        if self.recording:
            if landmarks and t >= self.count_from:
                self.counter.update(t - self.start_t, *frame_features(landmarks, w, h))
            self._draw_status(out, t, landmarks is not None)
            self._record(frame, out, t)
        self.process_times.append(time.perf_counter() - began)
        return out

    def _draw_status(self, img, t, person_seen):
        put_text(img, f"Reps: {len(self.counter.reps)}", 0, 1.6)
        if t < self.count_from:
            status = f"Get in position: {math.ceil(self.count_from - t)}"
        elif not person_seen:
            status = "No one in view"
        elif self.counter.phase == "waiting":
            status = f"Get into the {EXERCISES[self.exercise][1]} position to start"
        else:
            status = self.counter.phase.upper()
        put_text(img, status, 1)
        for k, (joint, angle) in enumerate(self.counter.angles.items()):
            put_text(img, f"{joint}: {angle:.0f} deg", 2 + k, 0.8)
        if self.counter.reps:
            last = self.counter.reps[-1]
            joint = next(iter(self.counter.ranges))
            put_text(img, f"Last rep: {joint} {last[f'{joint} at deepest (deg)']:.0f} deg, "
                          f"{last['duration_s']:.1f} s", 2 + len(self.counter.angles), 0.8)

    def _record(self, raw, annotated, t):
        # Write as many copies as real time requires, so the saved video plays at the right speed
        # even when processing runs slower than the video's frame rate.
        due = round((t - self.start_t) * self.fps) + 1
        while self.frames_written < due:
            for name, writer in self.writers.items():
                writer.write(annotated if name == "overlay" else raw)
            self.frames_written += 1

    def start(self, t, fps, size, countdown_s):
        """Start recording and (after the countdown) counting."""
        self.counter = LiveRepCounter(self.train, self.exercise)
        self.start_t, self.count_from, self.fps = t, t + countdown_s, fps
        self.output_dir.mkdir(parents=True, exist_ok=True)
        base = f"{self.exercise}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
        names = ["overlay", "raw"] if self.save_raw else ["overlay"]
        self.paths = {name: self.output_dir / f"{base}_{name}.mp4" for name in names}
        self.paths["reps"] = self.output_dir / f"{base}_reps.csv"
        self.writers = {name: BackgroundWriter(self.paths[name], fps, size) for name in names}
        print(f"Recording started; counting {self.exercise} in {countdown_s:.0f} s")

    def stop(self):
        """Finish the recording, save the rep table, and print a summary."""
        for writer in self.writers.values():
            writer.release()
        summary = self.counter.summary()
        summary.to_csv(self.paths["reps"])
        print(f"\n{len(summary)} {self.exercise} reps")
        if len(summary):
            print(summary.to_string())
        print(f"Processing speed: {1 / np.mean(self.process_times):.0f} frames per second")
        print("Saved:")
        for path in self.paths.values():
            print("  ", path)


def make_display(frame, session, max_w=1280, max_h=800):
    """Fit the frame to the screen and add the Start/Terminate button (buttons aren't saved in the recording)."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    disp = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame.copy()
    dh, dw = disp.shape[:2]
    action, label, color = ("terminate", "Terminate (T)", RED) if session.recording else ("start", "Start (S)", GREEN)
    bw, bh = max(170, dw // 4), max(46, dh // 13)
    x0, y0 = dw - bw - 12, 12
    cv2.rectangle(disp, (x0, y0), (x0 + bw, y0 + bh), color, -1)
    s = bh / 70
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
    cv2.putText(disp, label, (x0 + (bw - tw) // 2, y0 + (bh + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, s, WHITE, 2, cv2.LINE_AA)
    if not session.recording:
        cv2.putText(disp, "Preview (not recording). Q = quit", (12, dh - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, WHITE, 2, cv2.LINE_AA)
    return disp, {action: (x0, y0, x0 + bw, y0 + bh)}


def parse_args():
    p = argparse.ArgumentParser(description="Count reps live from a webcam and save the recording.")
    p.add_argument("--exercise", default="squats", choices=list(EXERCISES))
    p.add_argument("--camera", type=int, default=0, help="camera number: 0 is the default webcam, 1 the next, ...")
    p.add_argument("--video", help="use a recorded video instead of the camera (starts right away, for testing)")
    p.add_argument("--countdown", type=float, default=5, help="seconds to get into position after Start")
    p.add_argument("--pose-model", default="full", choices=["full", "lite"],
                   help="MediaPipe model: full counted the test videos correctly; lite is a bit faster but "
                        "counted an extra rep in one test")
    p.add_argument("--output", default=str(HERE / "Video Output"), help="folder for the saved files")
    p.add_argument("--no-window", action="store_true", help="with --video: run without a window (for automated tests)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_window and not args.video:
        raise SystemExit("--no-window only works with --video (there'd be no way to press Terminate)")
    pose_model = HERE / f"pose_landmarker_{args.pose_model}.task"
    if not pose_model.exists():
        print(f"Downloading MediaPipe's {args.pose_model} pose model (first run only)...")
        urllib.request.urlretrieve(POSE_MODEL_URL.format(args.pose_model), pose_model)
    print("Loading the Kaggle data to train the rep counter...")
    train = load_training_data(kagglehub.dataset_download("muhannadtuameh/exercise-recognition"))

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video or f'camera {args.camera}'}. For a camera, check that no other "
                         "app is using it and that Settings > Privacy & security > Camera allows desktop apps.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 5 <= fps <= 60 else 30
    camera = None if args.video else CameraReader(cap)

    # Load the model from memory: MediaPipe can't open files by full path in folders with non-English names (like 桌面)
    options = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_buffer=pose_model.read_bytes()),
                                           running_mode=vision.RunningMode.VIDEO)
    clicks = deque()
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        session = Session(landmarker, train, args.exercise, Path(args.output), save_raw=camera is not None)
        if not args.no_window:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW, lambda event, x, y, *_: clicks.append((x, y))
                                 if event == cv2.EVENT_LBUTTONUP else None)
            print("Window open: click Start (or press S) when ready")
        t0, i = time.monotonic(), 0
        while True:
            if camera:
                frame, captured = camera.read()
                t = None if frame is None else captured - t0
            else:
                ok, frame = cap.read()
                t, i = i / fps, i + 1   # a recording's own timeline, however fast it's processed
                if not ok:
                    frame = None
            if frame is None:
                break   # end of the video file, or the camera stopped
            if not camera and not session.recording:   # a recording starts right away, no countdown
                session.start(t, fps, (frame.shape[1], frame.shape[0]), countdown_s=0)
            out = session.process(frame, t)
            if args.no_window:
                continue

            display, buttons = make_display(out, session)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            action = {ord("s"): "start", ord("t"): "terminate", ord("q"): "quit", 27: "quit"}.get(key)
            while clicks:
                x, y = clicks.popleft()
                action = next((name for name, (x0, y0, x1, y1) in buttons.items()
                               if x0 <= x <= x1 and y0 <= y <= y1), action)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                action = "quit"   # window closed with its X
            if action == "start" and not session.recording:
                session.start(t, fps, (frame.shape[1], frame.shape[0]), args.countdown)
            elif action == "quit" or (action == "terminate" and session.recording):
                break
        if camera:
            camera.release()
        else:
            cap.release()
        if session.recording:
            session.stop()
        else:
            print("Quit without recording")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

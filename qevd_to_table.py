"""Turn QEVD's squat and push-up clips into dataframes: one row per frame, and one row per clip.

QEVD stays where you downloaded it. This reads the labels from QEVD-FIT-COACH.zip, takes each clip
straight out of combined-part-1.zip, runs MediaPipe on it, and writes:

    qevd_data/clips.csv        one row per clip: exercise, QEVD's labels (as text and as True/False
                               columns, e.g. "shallow"), its train/test split, and how many frames have
                               a person in them
    qevd_data/frames.parquet   one row per frame with a person in it: the 33 landmarks scaled like the
                               Kaggle data, their visibility (vis_*), and the 12 joint angles. Same
                               columns as `poses` in pose-detection.ipynb, plus clip and exercise.

Run it from the project folder with the .venv active. It takes about an hour; it's safe to stop and
run again, because finished clips are skipped.
    python qevd_to_table.py
    python qevd_to_table.py --limit 5          # a quick test: 5 clips per exercise
    python qevd_to_table.py --exercises squats pushups "pushups (on knees)"   # other QEVD exercises
    python qevd_to_table.py --part-zip combined-part-1.zip combined-part-2.zip  # more parts, if downloaded

Then, in a notebook:
    clips = pd.read_csv("qevd_data/clips.csv", dtype={"clip": str})
    frames = pd.read_parquet("qevd_data/frames.parquet")

QEVD's license is research-only and forbids sharing the data or anything containing it, so
qevd_data/ is in .gitignore. Don't upload it anywhere.
"""
import argparse
import json
import os
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from features import ALL_ANGLES, LANDMARK_COLS, LANDMARKS, angles_from_points, normalize

HERE = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "Downloads"
POSE_MODEL = HERE / "pose_landmarker_full.task"
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
VIS_COLS = [f"vis_{name}" for name in LANDMARKS]


def read_labels(coach_zip, exercises, available):
    """One row per labelled clip of `exercises` that is in the downloaded part(s)."""
    with zipfile.ZipFile(coach_zip) as z:
        records = json.loads(z.read("QEVD-FIT-COACH/fine_grained_labels.json"))
    rows = []
    for r in records:
        clip = PurePosixPath(r["video_path"]).stem   # "./00001672.mp4" -> "00001672"
        # Labels look like "squats - shallow"; keep the ones for our exercises
        mine = [(short, long) for short, long in zip(r["labels"], r["labels_descriptive"])
                if short.split(" - ", 1)[0] in exercises]
        if clip in available and mine:
            rows.append({"clip": clip, "exercise": mine[0][0].split(" - ", 1)[0], "split": r["split"],
                         "labels": "; ".join(short.split(" - ", 1)[-1] for short, _ in mine),
                         "labels_descriptive": "; ".join(long.split(" - ", 1)[-1] for _, long in mine)})
    return pd.DataFrame(rows, columns=["clip", "exercise", "split", "labels", "labels_descriptive"])


# Each worker process opens the zips and sets up MediaPipe once, then handles many clips
_worker = {}


def _start_worker(part_zips, model_bytes):
    from mediapipe.tasks.python import BaseOptions, vision

    _worker["zips"] = {path: zipfile.ZipFile(path) for path in part_zips}
    # From memory: MediaPipe can't open files by full path in folders with non-English names (like 桌面)
    _worker["options"] = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_buffer=model_bytes),
                                                      running_mode=vision.RunningMode.VIDEO)


def _extract(task):
    """Run MediaPipe on one clip and save its landmarks to an .npz. Returns (clip, error or None)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    clip, zip_path, member, out_path = task
    # OpenCV only reads videos from files, so the clip sits in a temporary file just while it's read
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, "wb") as f:
        f.write(_worker["zips"][zip_path].read(member))
    cap = cv2.VideoCapture(tmp)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames, points, visibility, size, i = [], [], [], (0, 0), 0
        # A new landmarker per clip: VIDEO mode tracks the person from frame to frame within one clip
        with vision.PoseLandmarker.create_from_options(_worker["options"]) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                size = (frame.shape[1], frame.shape[0])
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(image, int(i * 1000 / fps))
                if result.pose_landmarks:   # frames with nobody in them are skipped
                    lms = result.pose_landmarks[0]
                    frames.append(i)
                    points.append([[lm.x * size[0], lm.y * size[1], lm.z * size[0]] for lm in lms])
                    visibility.append([lm.visibility for lm in lms])
                i += 1
        # Save under a temporary name first, so a run that's stopped mid-save never leaves a broken file
        partial = Path(out_path).with_suffix(".partial.npz")
        np.savez_compressed(partial, frames=np.array(frames, dtype=np.int32),
                            points=np.array(points, dtype=np.float32).reshape(-1, len(LANDMARKS), 3),
                            visibility=np.array(visibility, dtype=np.float32).reshape(-1, len(LANDMARKS)),
                            fps=fps, n_frames=i, width=size[0], height=size[1])
        os.replace(partial, out_path)
        return clip, None
    except Exception as e:
        return clip, f"{type(e).__name__}: {e}"
    finally:
        cap.release()
        os.remove(tmp)


def build_tables(clips, poses_dir):
    """Combine the per-clip .npz files into the frames table, and add per-clip stats to `clips`."""
    parts, stats = [], []
    for clip in clips.itertuples():
        path = poses_dir / f"{clip.clip}.npz"
        if not path.exists():
            stats.append({"fps": np.nan, "n_frames": np.nan, "frames_with_person": 0, "width": np.nan, "height": np.nan})
            continue
        d = np.load(path)
        stats.append({"fps": float(d["fps"]), "n_frames": int(d["n_frames"]), "frames_with_person": len(d["frames"]),
                      "width": int(d["width"]), "height": int(d["height"])})
        if not len(d["frames"]):
            continue
        points = normalize(d["points"])   # same scaling as the Kaggle data
        parts.append(pd.concat([
            pd.DataFrame({"clip": clip.clip, "exercise": clip.exercise, "frame": d["frames"],
                          "time_s": d["frames"] / float(d["fps"])}),
            pd.DataFrame(points.reshape(len(points), -1), columns=LANDMARK_COLS),
            pd.DataFrame(d["visibility"], columns=VIS_COLS),
            pd.DataFrame(angles_from_points(points), columns=list(ALL_ANGLES)),
        ], axis=1))
    frames = pd.concat(parts, ignore_index=True)
    numbers = LANDMARK_COLS + VIS_COLS + list(ALL_ANGLES) + ["time_s"]
    frames[numbers] = frames[numbers].astype(np.float32)   # half the size of float64, plenty precise
    frames["exercise"] = frames["exercise"].astype("category")

    clips = pd.concat([clips.reset_index(drop=True), pd.DataFrame(stats)], axis=1)
    # One True/False column per label, e.g. clips[clips["shallow"]]
    label_sets = clips["labels"].str.split("; ")
    for label in sorted({label for labels in label_sets for label in labels}):
        clips[label] = label_sets.apply(lambda labels: label in labels)
    return clips, frames


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coach-zip", type=Path, default=DOWNLOADS / "QEVD-FIT-COACH.zip",
                   help="QEVD-FIT-COACH.zip, which has the labels")
    p.add_argument("--part-zip", type=Path, nargs="+", default=[DOWNLOADS / "combined-part-1.zip"],
                   help="the merged QEVD-FIT-300k part zip(s), which have the clips")
    p.add_argument("--exercises", nargs="+", default=["squats", "pushups"], help="QEVD exercise names, exactly")
    p.add_argument("--out", type=Path, default=HERE / "qevd_data")
    p.add_argument("--limit", type=int, help="only the first N clips of each exercise (for a quick test)")
    p.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)),
                   help="clips processed at the same time")
    return p.parse_args()


def main():
    args = parse_args()
    poses_dir = args.out / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    members = {}   # clip -> (part zip, file inside it)
    for part in args.part_zip:
        with zipfile.ZipFile(part) as z:
            members |= {PurePosixPath(n).stem: (str(part), n) for n in z.namelist() if n.endswith(".mp4")}
    clips = read_labels(args.coach_zip, set(args.exercises), members)
    if args.limit:
        clips = clips.groupby("exercise", sort=False).head(args.limit)
    print("Labelled clips found:", clips["exercise"].value_counts().to_dict())

    todo = [(c, *members[c], str(poses_dir / f"{c}.npz")) for c in clips["clip"] if not (poses_dir / f"{c}.npz").exists()]
    print(f"{len(clips) - len(todo)} clips already done, {len(todo)} to go, {args.workers} at a time")
    failed = []
    if todo:
        if not POSE_MODEL.exists():
            print("Downloading MediaPipe's pose model (first run only)...")
            urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL)
        began = time.monotonic()
        with ProcessPoolExecutor(args.workers, initializer=_start_worker,
                                 initargs=([str(p) for p in args.part_zip], POSE_MODEL.read_bytes())) as pool:
            for k, (clip, error) in enumerate(pool.map(_extract, todo, chunksize=2), start=1):
                if error:
                    failed.append(clip)
                    print(f"  clip {clip} failed: {error}")
                if k % 50 == 0 or k == len(todo):
                    per_clip = (time.monotonic() - began) / k
                    print(f"  {k}/{len(todo)} clips, about {per_clip * (len(todo) - k) / 60:.0f} min left", flush=True)

    clips, frames = build_tables(clips, poses_dir)
    clips.to_csv(args.out / "clips.csv", index=False)
    frames.to_parquet(args.out / "frames.parquet", index=False)
    print(f"\nclips.csv: {len(clips)} clips; frames.parquet: {len(frames)} frames "
          f"({(args.out / 'frames.parquet').stat().st_size / 1e6:.0f} MB)")
    print("Clips with no person found:", int((clips["frames_with_person"] == 0).sum()), "| failed:", failed or "none")


if __name__ == "__main__":
    main()

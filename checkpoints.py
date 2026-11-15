"""Checkpoint model: which position of which exercise a frame shows, e.g. "squats_down".

Train once (about a minute; needs the Kaggle data, which kagglehub downloads):
    python checkpoints.py
This prints how well the model does and saves models/checkpoint_model.joblib.

Two parts:

- The model: XGBoost, trained on the Kaggle frames, predicting one of the 10 poses (5 exercises, up or
  down). Besides the 12 joint angles from features.py it looks at how the body sits on screen (how far
  the thighs, shins, torso and arms tilt from upright, how high the hips and hands are), which is what
  tells a push-up from a squat. When the exercise is already known, as in live.py, only that exercise's
  up and down are compared.
- CheckpointTracker: follows the model frame by frame and reports each checkpoint as it is reached
  (the bottom of every squat, the top of every push-up), with the pose at that moment.
  issue_flagger.py compares that pose with form_reference_rules.csv.
"""
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import ALL_ANGLES, LANDMARK_COLS, LANDMARKS, angle_joints, angles_from_points
from reps import EXERCISES, REST_BELOW, WORK_ABOVE, RepTracker, joint_ranges, load_training_data, work_position

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "checkpoint_model.joblib"
REST_WINDOW_S = 1.5   # the rest checkpoint (e.g. the top of a push-up) is judged within this long of arriving

# What the model sees besides the 12 joint angles: on-screen (2D) measures. Lengths are in torso lengths,
# so they don't depend on how close the camera is.
TILTS = {   # name: (from, up to): how far that line leans away from upright
    **{f"{side} {name} tilt": (f"{side}_{lower}", f"{side}_{upper}")
       for side in ("left", "right")
       for name, lower, upper in [("thigh", "knee", "hip"), ("shin", "ankle", "knee"),
                                  ("upper arm", "elbow", "shoulder"), ("forearm", "wrist", "elbow")]},
    "torso tilt": ("mid_hip", "mid_shoulder"),
}
SCREEN_ANGLES = {   # name: three joints; the angle at the middle one
    f"{side} {name} (on screen)": tuple(f"{side}_{j}" for j in joints)
    for side in ("left", "right")
    for name, joints in [("elbow", ("shoulder", "elbow", "wrist")), ("knee", ("hip", "knee", "ankle")),
                         ("hip bend", ("shoulder", "hip", "knee"))]
}
HEIGHTS = {   # name: (below, above): how far the second is above the first
    "left wrist above shoulder": ("left_shoulder", "left_wrist"),
    "right wrist above shoulder": ("right_shoulder", "right_wrist"),
    "hips above knees": ("mid_knee", "mid_hip"),
    "hips above ankles": ("mid_ankle", "mid_hip"),
    "shoulders above hips": ("mid_hip", "mid_shoulder"),
}
WIDTHS = {"feet width": "ankle", "hand width": "wrist", "shoulder width": "shoulder"}   # name: left to right at this joint
FEATURES = [*ALL_ANGLES, *TILTS, *SCREEN_ANGLES, *HEIGHTS, *WIDTHS]

MIDPOINTS = ["shoulder", "hip", "knee", "ankle"]   # mid_hip etc.: halfway between the left and right joint
_POINTS = LANDMARKS + [f"mid_{joint}" for joint in MIDPOINTS]


def _columns(names):
    return [_POINTS.index(name) for name in names]


_TILT, _ANGLE, _HEIGHT = ([_columns(names) for names in zip(*table.values())] for table in (TILTS, SCREEN_ANGLES, HEIGHTS))
_WIDTH = [_columns(f"{side}_{joint}" for joint in WIDTHS.values()) for side in ("left", "right")]


def tilt(lower, upper):
    """How far the line from `lower` up to `upper` leans away from upright, in degrees, as seen on screen.

    0 = `upper` is straight above `lower`, 90 = level with it, more than 90 = below it.
    Both are x, y points (or arrays of them) with y pointing down, as in images.
    """
    line = np.asarray(upper) - np.asarray(lower)
    return np.degrees(np.arctan2(np.abs(line[..., 0]), -line[..., 1]))


def screen_angle(a, b, c):
    """Angle a-b-c in degrees as seen on screen, with the vertex at b. x, y points, or arrays of them."""
    ba, bc = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    cos = (ba * bc).sum(-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def pose_features(points):
    """The model's inputs for an (n_frames, 33, 3) array of normalized landmarks: (n_frames, len(FEATURES))."""
    points = np.asarray(points, dtype=float)
    xy = points[:, :, :2]
    mids = [(xy[:, LANDMARKS.index(f"left_{joint}")] + xy[:, LANDMARKS.index(f"right_{joint}")]) / 2 for joint in MIDPOINTS]
    xy = np.concatenate([xy, np.stack(mids, axis=1)], axis=1)   # the 33 landmarks, then the midpoints
    torso = np.linalg.norm(xy[:, _columns(["mid_shoulder"])] - xy[:, _columns(["mid_hip"])], axis=2) + 1e-9
    return np.hstack([
        angles_from_points(points),
        tilt(xy[:, _TILT[0]], xy[:, _TILT[1]]),
        screen_angle(xy[:, _ANGLE[0]], xy[:, _ANGLE[1]], xy[:, _ANGLE[2]]),
        (xy[:, _HEIGHT[0], 1] - xy[:, _HEIGHT[1], 1]) / torso,   # y points down
        np.linalg.norm(xy[:, _WIDTH[0]] - xy[:, _WIDTH[1]], axis=2) / torso,
    ])


def xgboost():
    from xgboost import XGBClassifier
    # 150 rounds score the same as form_model.ipynb's 300 here, and predict a live frame twice as fast
    return XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                         random_state=0)


def random_forest():   # for comparison: what form_analysis.ipynb and the rep counter use
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=150, random_state=0)


def compare(data, X):
    """5-fold cross-validation of random forest vs XGBoost, on the 12 angles alone and on all the features.

    Every frame is predicted by a model that never saw it. Returns one row of scores per combination.
    """
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    codes, poses = pd.factorize(data["pose"], sort=True)
    exercise = np.array([pose.rsplit("_", 1)[0] for pose in poses])
    rows = []
    for features, columns in [("12 angles", slice(len(ALL_ANGLES))), ("angles + on-screen", slice(None))]:
        for name, make in [("random forest", random_forest), ("XGBoost", xgboost)]:
            proba = np.zeros((len(X), len(poses)))
            for train_i, test_i in StratifiedKFold(5, shuffle=True, random_state=0).split(X, codes):
                proba[test_i] = make().fit(X[train_i, columns], codes[train_i]).predict_proba(X[test_i, columns])
            predicted = proba.argmax(axis=1)
            # With the exercise known, only its own up and down compete
            known = np.where(exercise == exercise[codes][:, None], proba, -1).argmax(axis=1)
            rows.append({"features": features, "model": name,
                         "pose accuracy": (predicted == codes).mean(),
                         "pose macro F1": f1_score(codes, predicted, average="macro"),
                         "exercise accuracy": (exercise[predicted] == exercise[codes]).mean(),
                         "up/down accuracy (exercise known)": (known == codes).mean(),
                         "up/down macro F1 (exercise known)": f1_score(codes, known, average="macro")})
    return pd.DataFrame(rows).set_index(["features", "model"])


def train(path=None):
    """Compare the models, then train XGBoost on every Kaggle frame and save it."""
    import kagglehub
    data = load_training_data(path or kagglehub.dataset_download("muhannadtuameh/exercise-recognition"))
    X = pose_features(data[LANDMARK_COLS].to_numpy().reshape(len(data), len(LANDMARKS), 3))
    print(f"{len(data)} Kaggle frames, {data['pose'].nunique()} poses, {X.shape[1]} features\n")
    print("5-fold cross-validation (every frame predicted by a model that never saw it):")
    print(compare(data, X).round(3).to_string(), "\n")

    codes, poses = pd.factorize(data["pose"], sort=True)
    bundle = {"model": xgboost().fit(X, codes), "poses": list(poses), "features": FEATURES,
              # For each exercise's key joints: the typical angle at rest and at work (as the rep counter uses)
              "ranges": {exercise: joint_ranges(data, exercise) for exercise in EXERCISES}}
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL)
    print(f"Saved {MODEL}")
    return bundle


def load(path=MODEL):
    """The saved model, or None if `python checkpoints.py` hasn't been run yet."""
    if not Path(path).exists():
        return None
    bundle = joblib.load(path)
    if bundle["features"] != FEATURES:
        raise ValueError(f"{path} was trained on different features. Run `python checkpoints.py` to retrain it.")
    return bundle


def pose_probabilities(points, bundle):
    """P(each of the 10 poses) for every frame of an (n_frames, 33, 3) array: a table with one column per pose."""
    return pd.DataFrame(bundle["model"].predict_proba(pose_features(points)), columns=bundle["poses"])


def p_work(points, exercise, bundle):
    """P(the frame shows the exercise's work position, e.g. the bottom of a squat), given that it is this exercise."""
    proba = pose_probabilities(points, bundle)
    work, rest = (proba[f"{exercise}_{position}"] for position in (work_position(exercise), EXERCISES[exercise][1]))
    return (work / (work + rest)).to_numpy()


@dataclass
class Checkpoint:
    """One checkpoint that was reached: the pose at the moment the person was furthest into it."""
    position: str            # "up" or "down"
    rep: int                 # the rep it belongs to, counting from 1 (0 = the rest position before the first rep)
    t: float                 # when, in seconds
    points: np.ndarray       # (33, 3) normalized landmarks, the median of 5 frames around that moment
    visibility: np.ndarray   # (33,) how clearly MediaPipe saw each landmark, 0-1


class CheckpointTracker:
    """Finds the checkpoints in a stream of frames (live, or a recording played through).

    It follows the rep counter's rule (reps.py) with this model in place of its random forest: the
    work position starts when both the motion signal and the model's P(work position) rise above
    WORK_ABOVE, and the rest position when both fall back below REST_BELOW.

    Within each stay at a checkpoint, the pose kept is the one furthest into it: for the work position,
    where the key joints have moved furthest from their rest angles (the bottom of the squat); for the
    rest position, where they are closest to them (the top of the push-up).
    """

    def __init__(self, exercise, bundle, window=5):
        self.exercise = exercise
        self.rest, self.work = EXERCISES[exercise][1], work_position(exercise)
        self.reps = 0
        self.tracker = RepTracker()
        self._booster = bundle["model"].get_booster()
        self._booster.set_param({"nthread": 1})   # for one frame at a time, starting threads costs more than it saves
        self._work_col, self._rest_col = (bundle["poses"].index(f"{exercise}_{p}") for p in (self.work, self.rest))
        # For each key joint: its typical (rest, work) angles, and the angle column and landmarks of each side
        self._joints = [(rest, work, [(FEATURES.index(col), [LANDMARKS.index(j) for j in angle_joints(col)])
                                      for col, label in ALL_ANGLES.items() if label in (f"R {joint}", f"L {joint}")])
                        for joint, (rest, work) in bundle["ranges"][exercise].items()]
        self._recent = deque(maxlen=window)   # (t, P(work), motion, points, visibility) of the last frames
        self._stay = None                     # the stay in progress: [position, when it began, best frame so far]

    @property
    def position(self):
        """The current position ("up" or "down"), or "waiting" until a clear rest position is seen."""
        return {None: "waiting", "rest": self.rest, "work": self.work}[self.tracker.state]

    def update(self, t, points, visibility):
        """Feed one frame: its (33, 3) normalized landmarks and 33 visibilities. Returns a Checkpoint or None."""
        points = np.asarray(points, dtype=float).reshape(len(LANDMARKS), 3)
        visibility = np.asarray(visibility, dtype=float)
        features = pose_features(points[None])
        proba = self._booster.inplace_predict(features)[0]
        p = proba[self._work_col] / (proba[self._work_col] + proba[self._rest_col])
        self._recent.append((t, p, self._motion(features[0], visibility), points, visibility))
        # The median of the last 5 frames ignores a single misread frame
        p, motion = (np.median([frame[i] for frame in self._recent]) for i in (1, 2))

        before = self.tracker.state
        rep = self.tracker.step(t, min(p, motion) >= WORK_ABOVE, max(p, motion) <= REST_BELOW)
        reached = None
        if self.tracker.state != before:
            if before == "work" and rep:      # back at rest: the rep is done, and so is its work checkpoint
                self.reps += 1
                reached = self._end_stay()
            elif before == "rest":            # leaving rest
                reached = self._end_stay()
            self._stay = None if self.tracker.state is None else [self.position, t, None]
        if self._stay:
            self._keep_best(motion)
            if self._stay[0] == self.rest and t - self._stay[1] >= REST_WINDOW_S:
                reached = self._end_stay()
        return reached

    def finish(self):
        """Call when the video ends: the rest checkpoint still being watched, if any."""
        return self._end_stay() if self._stay and self._stay[0] == self.rest else None

    def _motion(self, features, visibility):
        """0 at the typical rest pose, 1 at the typical work pose (motion_signal in reps.py, for one frame)."""
        progress = []
        for rest, work, sides in self._joints:
            col, _ = max(sides, key=lambda side: visibility[side[1]].mean())   # the side MediaPipe sees better
            progress.append((features[col] - rest) / (work - rest))
        return float(np.mean(progress))

    def _keep_best(self, motion):
        """Remember the last 5 frames (as one median pose) if they are the furthest into this checkpoint so far."""
        score = motion if self._stay[0] == self.work else -motion
        if self._stay[2] is None or score > self._stay[2][0]:
            frames = list(self._recent)
            self._stay[2] = (score, frames[len(frames) // 2][0],
                             np.median([f[3] for f in frames], axis=0), np.median([f[4] for f in frames], axis=0))

    def _end_stay(self):
        stay, self._stay = self._stay, None
        if not stay or stay[2] is None:
            return None
        _, t, points, visibility = stay[2]
        return Checkpoint(stay[0], self.reps, t, points, visibility)


if __name__ == "__main__":
    train()

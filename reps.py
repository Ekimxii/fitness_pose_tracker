"""Count exercise reps in a video or live from a camera, using the motion and the up/down model together.

A rep is one trip from the rest position to the work position and back: standing, down to the
bottom of the squat, and standing again. Two signals track where you are, each running from
0 (rest position) to 1 (work position):

- motion: how far the key joint angles have moved from the typical rest pose toward the typical
  work pose, measured on the side of the body facing the camera. "Typical" means the median
  angle in the Kaggle data.
- model: the up/down random forest from form_analysis.ipynb (trained on the Kaggle data's
  12 angles), giving the probability that each frame is in the work position.

A rep counts only when both signals rise above WORK_ABOVE and later both fall below REST_BELOW.
Needing both filters out half-reps and stray movements (which fool the motion signal) and
misread frames (which fool the model).

count_reps works on a whole recorded video; LiveRepCounter takes one frame at a time from a camera.
Both use RepTracker, so they follow exactly the same rules.
"""
from collections import deque

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from features import ALL_ANGLES, LANDMARKS, add_angles, angle_joints

# exercise -> (joints whose angles show the movement, position each rep starts and ends in)
EXERCISES = {
    "squats": (["knee", "hip bend"], "up"),
    "pushups": (["elbow"], "up"),
    "pullups": (["elbow"], "down"),
    "situp": (["hip bend"], "down"),
    "jumping_jacks": (["shoulder"], "down"),
}
WORK_ABOVE, REST_BELOW = 0.6, 0.4   # thresholds on both signals; the gap stops jitter double-counting
ANGLE_COLS = list(ALL_ANGLES)


def load_training_data(path):
    """The Kaggle frames (from the folder kagglehub downloads) with all 12 angles, exercise and position."""
    labels = pd.read_csv(f"{path}/labels.csv")
    labels["position"] = labels["pose"].str.extract(r"(up|down)$", expand=False)
    labels["exercise"] = labels["pose"].str.replace(r"_?(up|down)$", "", regex=True)
    return add_angles(pd.read_csv(f"{path}/landmarks.csv")).merge(labels, on="pose_id")


def work_position(exercise):
    """The position a rep goes to: "down" for squats, "up" for pull-ups, and so on."""
    return "down" if EXERCISES[exercise][1] == "up" else "up"


def check_exercise(poses, train, exercise):
    """Ask a model trained on all 5 exercises what the video looks like; warn if it isn't `exercise`."""
    model = RandomForestClassifier(n_estimators=150, random_state=0).fit(train[ANGLE_COLS], train["exercise"])
    votes = pd.Series(model.predict(poses[ANGLE_COLS])).value_counts(normalize=True)
    print(f"The exercise model says the frames look most like {votes.index[0]} ({votes.iloc[0]:.0%} of frames)")
    if votes.index[0] != exercise:
        print(f"Warning: EXERCISE is {exercise!r}. Check it matches the video before trusting the count.")
    return votes


def _side_cols(joint):
    """The (right, left) angle columns for a joint name like "knee"."""
    return tuple(next(c for c, label in ALL_ANGLES.items() if label == f"{s} {joint}") for s in "RL")


def _facing_side(poses, vis, joint):
    right, left = _side_cols(joint)
    seen_right = vis[list(angle_joints(right))].mean(axis=1).to_numpy()
    seen_left = vis[list(angle_joints(left))].mean(axis=1).to_numpy()
    return pd.Series(np.where(seen_right >= seen_left, poses[right], poses[left]), index=poses.index)


def facing_side_angle(poses, vis, joint):
    """The angle at `joint` ("knee", "elbow", ...) in every frame, on the side MediaPipe sees better."""
    return smooth(_facing_side(poses, vis, joint))


def joint_ranges(train, exercise):
    """For each key joint: its typical angle in the rest position and in the work position (Kaggle medians)."""
    joints, rest = EXERCISES[exercise]
    ex = train[train["exercise"] == exercise]
    return {joint: tuple(np.median(ex.loc[ex["position"] == pos, list(_side_cols(joint))].to_numpy())
                         for pos in (rest, work_position(exercise)))
            for joint in joints}


def _progress(angles, ranges):
    """Average over the key joints of how far each angle has moved from rest (0) toward work (1)."""
    return sum((angles[j] - rest) / (work - rest) for j, (rest, work) in ranges.items()) / len(ranges)


def motion_signal(poses, vis, train, exercise):
    """0 at the typical rest pose, 1 at the typical work pose, averaged over the exercise's key joints."""
    ranges = joint_ranges(train, exercise)
    return smooth(_progress({j: _facing_side(poses, vis, j) for j in ranges}, ranges))


def position_model(train, exercise):
    """The up/down random forest from form_analysis.ipynb (12 angles), trained for one exercise."""
    ex = train[train["exercise"] == exercise]
    return RandomForestClassifier(n_estimators=150, random_state=0).fit(ex[ANGLE_COLS], ex["position"])


def _p_work(model, poses, exercise):
    return model.predict_proba(poses[ANGLE_COLS])[:, list(model.classes_).index(work_position(exercise))]


def model_signal(poses, train, exercise):
    """The up/down model's probability that each frame is in the work position."""
    return smooth(pd.Series(_p_work(position_model(train, exercise), poses, exercise), index=poses.index))


def smooth(signal):
    """Rolling median over 5 frames: removes single-frame jitter but keeps the shape of each rep."""
    return signal.rolling(5, center=True, min_periods=1).median()


class RepTracker:
    """The counting rule, applied one frame at a time.

    Counting only starts after a clear rest position, so starting crouched doesn't count. "Reps"
    shorter than min_rep_s are jitter. If nobody is detected for more than max_gap_s, a rep in
    progress is dropped and counting waits for the next clear rest position.
    """

    def __init__(self, min_rep_s=0.3, max_gap_s=0.5):
        self.min_rep_s, self.max_gap_s = min_rep_s, max_gap_s
        self.state = None    # None (waiting for a clear rest position), "rest" or "work"
        self.rest_t = self.last_t = None

    def step(self, t, at_work, at_rest):
        """Feed one frame. Returns (start_s, end_s) if this frame finishes a rep, else None."""
        if self.last_t is not None and t - self.last_t > self.max_gap_s:
            self.state = None
        self.last_t = t
        rep = None
        if at_rest:
            if self.state == "work" and t - self.rest_t >= self.min_rep_s:
                rep = (self.rest_t, t)
            self.state, self.rest_t = "rest", t
        elif at_work and self.state == "rest":
            self.state = "work"
        return rep


def count_reps(time_s, *signals, min_rep_s=0.3, max_gap_s=0.5):
    """Find reps in a recorded video: every signal rises above WORK_ABOVE, then falls back below REST_BELOW.

    Returns one row per rep with start_s (when the rest position was left) and end_s (when it was
    reached again).
    """
    stacked = np.vstack([np.asarray(s, dtype=float) for s in signals])
    at_work = (stacked >= WORK_ABOVE).all(axis=0)
    at_rest = (stacked <= REST_BELOW).all(axis=0)
    tracker = RepTracker(min_rep_s, max_gap_s)
    reps = [rep for t, w, r in zip(np.asarray(time_s, dtype=float), at_work, at_rest) if (rep := tracker.step(t, w, r))]
    return pd.DataFrame(reps, columns=["start_s", "end_s"])


def describe_reps(reps, poses, vis, motion, exercise):
    """One row per rep: timing, and the key angles at the deepest point (furthest into the work position)."""
    joints = EXERCISES[exercise][0]
    angles = {joint: facing_side_angle(poses, vis, joint) for joint in joints}
    columns = ["rep", "start_s", "deepest_s", "end_s", "duration_s"] + [f"{j} at deepest (deg)" for j in joints]
    rows = []
    for k, rep in enumerate(reps.itertuples(), start=1):
        inside = poses["time_s"].between(rep.start_s, rep.end_s)
        deepest = motion[inside].idxmax()
        rows.append([k, rep.start_s, poses["time_s"][deepest], rep.end_s, rep.end_s - rep.start_s]
                    + [angles[joint][deepest] for joint in joints])
    return pd.DataFrame(rows, columns=columns).set_index("rep").round(1)


class LiveRepCounter:
    """Counts reps one frame at a time (e.g. from a webcam), with the same rules as count_reps.

    The one difference: live smoothing can't look ahead, so it takes the median of the last 5 frames,
    which lags about 0.1 s behind at 30 frames per second.
    """

    def __init__(self, train, exercise, window=5):
        self.exercise = exercise
        self.ranges = joint_ranges(train, exercise)
        self.model = position_model(train, exercise)
        self._work_class = list(self.model.classes_).index(work_position(exercise))
        # For each key joint, the (angle column index, landmark indexes) of its right and left side
        self._sides = {joint: [(ANGLE_COLS.index(col), [LANDMARKS.index(j) for j in angle_joints(col)])
                               for col in _side_cols(joint)] for joint in self.ranges}
        self.tracker = RepTracker()
        self.reps = []      # one dict per finished rep, same columns as describe_reps
        self.angles = {}    # latest smoothed key joint angles, on the side facing the camera
        self._recent = {name: deque(maxlen=window) for name in ["motion", "model", *self.ranges]}
        self._since_rest = []   # (time, motion, angles) since the last rest frame, to find each rep's deepest point

    def update(self, t, angles, visibility):
        """Feed one frame: its 12 angles (ALL_ANGLES order) and 33 landmark visibilities. Returns the rep count."""
        for joint, sides in self._sides.items():
            # The side MediaPipe sees better, as in facing_side_angle
            col, _ = max(sides, key=lambda side: np.mean(visibility[side[1]]))
            self._recent[joint].append(angles[col])
        self.angles = {joint: float(np.median(self._recent[joint])) for joint in self.ranges}
        self._recent["motion"].append(_progress(self.angles, self.ranges))
        # Same as self.model.predict_proba, without scikit-learn's per-call checks (about 7x faster)
        x = np.asarray(angles, dtype=np.float32).reshape(1, -1)
        p_work = np.mean([tree.predict_proba(x, check_input=False)[0, self._work_class]
                          for tree in self.model.estimators_])
        self._recent["model"].append(p_work)
        motion, p_work = np.median(self._recent["motion"]), np.median(self._recent["model"])

        self._since_rest.append((t, motion, self.angles))
        at_rest = motion <= REST_BELOW and p_work <= REST_BELOW
        rep = self.tracker.step(t, motion >= WORK_ABOVE and p_work >= WORK_ABOVE, at_rest)
        if rep:
            deepest_t, _, deepest = max(self._since_rest, key=lambda frame: frame[1])
            self.reps.append({"rep": len(self.reps) + 1, "start_s": rep[0], "deepest_s": deepest_t, "end_s": rep[1],
                              "duration_s": rep[1] - rep[0],
                              **{f"{j} at deepest (deg)": a for j, a in deepest.items()}})
        if at_rest or self.tracker.state is None:
            self._since_rest = self._since_rest[-1:]
        return len(self.reps)

    @property
    def phase(self):
        """The current position ("up" or "down"), or "waiting" until a clear rest position is seen."""
        if self.tracker.state is None:
            return "waiting"
        return EXERCISES[self.exercise][1] if self.tracker.state == "rest" else work_position(self.exercise)

    def summary(self):
        columns = ["rep", "start_s", "deepest_s", "end_s", "duration_s"] + [f"{j} at deepest (deg)" for j in self.ranges]
        return pd.DataFrame(self.reps, columns=columns).set_index("rep").round(1)

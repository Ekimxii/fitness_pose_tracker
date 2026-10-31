"""Pose features shared by the notebooks: landmark names, normalization, joint angles.

Everything here matches the Kaggle "exercise-recognition" dataset, so features
computed from our own videos line up with the ones the analysis was done on.
"""
import numpy as np
import pandas as pd

# MediaPipe's 33 pose landmarks, in MediaPipe's order, spelled like the dataset
# (it calls three hand points pinky_1, index_1 and thumb_2)
LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky_1", "right_pinky_1", "left_index_1", "right_index_1", "left_thumb_2", "right_thumb_2",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
LANDMARK_COLS = [f"{axis}_{name}" for name in LANDMARKS for axis in "xyz"]

# Angle columns: name -> short label. Names list three joints with the vertex in the middle.
ORIGINAL_ANGLES = {  # the 7 in the dataset's angles.csv
    "right_elbow_right_shoulder_right_hip": "R shoulder",
    "left_elbow_left_shoulder_left_hip": "L shoulder",
    "right_knee_mid_hip_left_knee": "hip spread",
    "right_hip_right_knee_right_ankle": "R knee",
    "left_hip_left_knee_left_ankle": "L knee",
    "right_wrist_right_elbow_right_shoulder": "R elbow",
    "left_wrist_left_elbow_left_shoulder": "L elbow",
}
NEW_ANGLES = {  # the 5 added in form_analysis.ipynb
    "right_shoulder_right_hip_right_knee": "R hip bend",
    "left_shoulder_left_hip_left_knee": "L hip bend",
    "right_shoulder_right_hip_right_ankle": "R body line",
    "left_shoulder_left_hip_left_ankle": "L body line",
    "torso_lean": "torso lean",
}
ALL_ANGLES = {**ORIGINAL_ANGLES, **NEW_ANGLES}


def normalize(points):
    """Scale raw landmarks the way the dataset was built (MediaPipe's pose-classification recipe).

    points: (n_frames, 33, 3) array of x * image width, y * image height, z * image width.
    Returns the same shape, centred on the mid-hip and scaled so the torso (shoulders to hips,
    measured in 2D) is 40 units long, or smaller if a limb reaches further than 2.5 torsos.
    """
    points = np.asarray(points, dtype=float)
    hips = points[:, [LANDMARKS.index("left_hip"), LANDMARKS.index("right_hip")]].mean(axis=1)
    points = points - hips[:, None, :]
    shoulders = points[:, [LANDMARKS.index("left_shoulder"), LANDMARKS.index("right_shoulder")]].mean(axis=1)
    torso = np.linalg.norm(shoulders[:, :2], axis=1)
    reach = np.linalg.norm(points[:, :, :2], axis=2).max(axis=1)
    size = np.maximum(torso * 2.5, reach)
    return points / size[:, None, None] * 100


def joint(df, name):
    """x, y, z of one landmark in every frame, as an (n_frames, 3) array; mid_X averages left and right."""
    if name.startswith("mid_"):
        part = name[4:]
        return (joint(df, f"left_{part}") + joint(df, f"right_{part}")) / 2
    return df[[f"x_{name}", f"y_{name}", f"z_{name}"]].to_numpy()


def angle(a, b, c):
    """Angle a-b-c in degrees, with the vertex at b."""
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def add_angles(df):
    """Return a copy of df (which has the landmark columns) with all 12 angle columns added."""
    points = df[LANDMARK_COLS].to_numpy().reshape(len(df), len(LANDMARKS), 3)
    new = pd.DataFrame(angles_from_points(points), columns=list(ALL_ANGLES), index=df.index)
    return pd.concat([df.drop(columns=list(new), errors="ignore"), new], axis=1)


def angles_from_points(points):
    """The 12 angles (columns in ALL_ANGLES order) for an (n_frames, 33, 3) array of normalized landmarks.

    Plain numpy, so it's fast enough to run on every frame of a live camera feed.
    """
    points = np.asarray(points, dtype=float)

    def point(name):   # (n, 3) coordinates of a landmark; mid_X averages left and right
        if name.startswith("mid_"):
            return (point(f"left_{name[4:]}") + point(f"right_{name[4:]}")) / 2
        return points[:, LANDMARKS.index(name)]

    columns = []
    for col in ALL_ANGLES:
        if col == "torso_lean":
            # Angle between the hip-to-shoulder line and straight up (y points down, so up is -y)
            torso = point("mid_shoulder") - point("mid_hip")
            cos_up = -torso[:, 1] / np.linalg.norm(torso, axis=1)
            columns.append(np.degrees(np.arccos(np.clip(cos_up, -1, 1))))
        else:
            columns.append(angle(*(point(j) for j in angle_joints(col))))
    return np.column_stack(columns)


def angle_joints(col):
    """'right_hip_right_knee_right_ankle' -> ('right_hip', 'right_knee', 'right_ankle')."""
    parts = col.split("_")
    starts = [i for i, p in enumerate(parts) if p in ("left", "right", "mid")]
    return tuple("_".join(parts[i:j]) for i, j in zip(starts, starts[1:] + [len(parts)]))

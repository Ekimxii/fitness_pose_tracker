"""Form models: XGBoost models that judge each rep's form, trained on QEVD's labels.

Train once, on a computer with qevd_data/ (made by qevd_to_table.py). It takes a few minutes:
    python form_models.py
This saves models/form_models.joblib, which live.py loads to judge every rep, e.g. "depth: too shallow".
The saved models contain no QEVD data, only the trained trees.

The same questions and model settings as the QEVD section of form_model.ipynb. QEVD is filmed from the
front, so the answers are only reliable when the camera faces you.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import window_features

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models" / "form_models.joblib"
MIN_FRAMES = 30   # skip QEVD clips where the person was found in fewer frames

TARGETS = {   # question: (exercise, {answer: the QEVD labels that mean it})
    "squat depth": ("squats", {"too shallow": ["shallow"], "good (90°)": ["90 degrees"], "too deep": ["over 90 degrees"]}),
    "squat stance": ("squats", {"too narrow": ["narrow"], "good (shoulder-width)": ["shoulder-width"], "too wide": ["wide"]}),
    "squat back": ("squats", {"not straight": ["back not straight"], "no issue": ["no obvious issue"]}),
    "push-up depth": ("pushups", {"too shallow": ["shallow"], "good (90°)": ["90 degrees"], "too deep": ["deep"]}),
    "push-up hands": ("pushups", {"too narrow": ["narrow"], "good (shoulder-width)": ["shoulder-width"],
                                  "too wide": ["wide", "very wide"]}),
}


def xgboost():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         random_state=0)


def answers(clips, question):
    """Each clip's answer to a question. Clips with none (or more than one) of its labels are left out."""
    exercise, options = TARGETS[question]
    ex = clips[clips["exercise"] == exercise]
    hits = pd.DataFrame({answer: ex.reindex(columns=labels, fill_value=False).any(axis=1)
                         for answer, labels in options.items()})
    return hits[hits.sum(axis=1) == 1].idxmax(axis=1)


def train(data=HERE / "qevd_data"):
    """Train one XGBoost model per question on every usable QEVD clip."""
    from sklearn.utils.class_weight import compute_sample_weight

    clips = pd.read_csv(data / "clips.csv", dtype={"clip": str}).set_index("clip")
    frames = pd.read_parquet(data / "frames.parquet")
    print("Summarising each clip (a few minutes)...")
    features = frames.groupby("clip").apply(window_features)
    features = features[clips.loc[features.index, "frames_with_person"] >= MIN_FRAMES]
    clips = clips.loc[features.index]
    models = {}
    for question, (exercise, _) in TARGETS.items():
        y = answers(clips, question)
        codes, names = pd.factorize(y)   # XGBoost wants the answers as 0, 1, 2, ...
        model = xgboost().fit(features.loc[y.index], codes, sample_weight=compute_sample_weight("balanced", codes))
        models[question] = {"exercise": exercise, "model": model, "answers": list(names)}
        print(f"  {question}: trained on {len(y)} clips {y.value_counts().to_dict()}")
    return {"models": models, "feature_cols": list(features.columns)}


def load(path=MODELS):
    """The saved models, or None if they haven't been trained yet."""
    return joblib.load(path) if Path(path).exists() else None


def assess(frames, exercise, bundle):
    """Answer every question for `exercise` about one rep. `frames`: the rep's rows (landmarks and angles)."""
    row = window_features(frames).reindex(bundle["feature_cols"]).to_frame().T.astype(float)
    return {question: m["answers"][int(m["model"].predict(row)[0])]
            for question, m in bundle["models"].items() if m["exercise"] == exercise}


if __name__ == "__main__":
    bundle = train()
    MODELS.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODELS)
    print("saved", MODELS)

# Fitness Pose Tracker

Counts exercise reps from a webcam or video using MediaPipe pose landmarks, and judges each rep's form with ML models.

## Setup
- Python 3.14, venv in `.venv`: activate with `.\.venv\Scripts\Activate.ps1`
- `pip install -r requirements.txt`
- Do NOT install `opencv-python`. MediaPipe ships its own OpenCV and we use `opencv-contrib-python`; two copies conflict.

## Key files
- `live.py`: live rep counting from a webcam or video (`--camera N`, `--video PATH`). Output goes to `Video Output/`.
- `reps.py`: rep counting (`RepTracker`, `LiveRepCounter`, `count_reps`). A rep counts only when both the motion signal and the up/down random forest cross the thresholds (`WORK_ABOVE`/`REST_BELOW`).
- `features.py`: landmark normalization, joint angles, `window_features`.
- `form_models.py`: XGBoost form models trained on QEVD labels. Run `python form_models.py` to write `models/form_models.joblib`.
- `checkpoints.py`: the checkpoint model (XGBoost on the Kaggle frames: which of the 10 poses, e.g. `squats_down`, a frame shows) and `CheckpointTracker`, which reports the pose at each checkpoint. Run `python checkpoints.py` to print its cross-validation scores and write `models/checkpoint_model.joblib`.
- `form_reference_rules.csv`: the team's rule sheet: for each exercise and position, a measurement, its good range, and the issue outside it. Limits, issue names, body parts and camera views are read from it, so tune them there.
- `issue_flagger.py`: compares the pose at each checkpoint with the rule sheet and produces the issue dictionaries. How each rule is measured is in `MEASURES`, by `rule_id`: a new rule in the sheet needs a measure there. Severity = how far past the limit / `SEVERITY_SPAN`.
- `tests/`: `python -m unittest discover tests` (stick figures with known angles; needs no model, camera or data).
- `qevd_to_table.py`: builds `qevd_data/` from the QEVD dataset.
- Notebooks: `form_analysis.ipynb` (Kaggle up/down model), `form_model.ipynb` (RF vs XGBoost comparison), `pose-detection.ipynb`, `coach_feedback.ipynb`.

## Issue dictionary (the contract between Form and Feedback)
- `form_dict.py` defines the format that `issue_flagger.py` produces and the feedback side reads (`FeedbackController` in `feedback_model.ipynb`). Build and test against `make_dict`, `make_issue` and `validate` there.
- Values follow what the feedback side matches on: `exercise` and `body_part` are spelled as in `coach_feedback_final.csv` (`squat`, `pushup`, `jumping jacks`; `knee`, not `knees`), and `issue` is the rule sheet's wording (`shallow depth`). The code's own exercise names (`squats`) are translated by `form_dict.EXERCISE_NAMES`.
- Don't change the format on your own. If you need a new field, report it. Adding a value to `BODY_PARTS` is fine.

## Conventions
- Form models learned from front-view video, so feedback is only reliable when the person faces the camera.
- The rule sheet's limits assume a level camera that sees the whole body. Most rules need a side view; knee valgus, jumping jacks and pull-ups need a front view. `issue_flagger.py` skips what it can't see rather than guess.
- Saved models must contain no raw QEVD data, only the trained models.
- Docstrings explain things in plain language. Keep that style.
- Test with the clips in `Videos for Testing/`.

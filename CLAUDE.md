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
- `qevd_to_table.py`: builds `qevd_data/` from the QEVD dataset.
- Notebooks: `form_analysis.ipynb` (Kaggle up/down model), `form_model.ipynb` (RF vs XGBoost comparison), `pose-detection.ipynb`, `coach_feedback.ipynb`.

## Conventions
- Form models learned from front-view video, so feedback is only reliable when the person faces the camera.
- Saved models must contain no raw QEVD data, only the trained models.
- Docstrings explain things in plain language. Keep that style.
- Test with the clips in `Videos for Testing/`.

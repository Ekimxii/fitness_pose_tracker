# Fitness Pose Tracker

Upload a workout video and get coaching on your form: the video with your skeleton drawn on, a rep count, and short coach messages that correct mistakes and encourage you. Supports squats, push-ups, pull-ups, sit-ups and jumping jacks.

The project has two parts. **Form analysis** finds what is off in each rep, and **coach feedback** turns that into messages. The two connect through one issue dictionary per checkpoint (format defined in `form_dict.py`).

# I) Form Analysis

## Overview

The form analysis component watches a workout video and checks the user's form on every rep. It finds the body in each frame, counts reps, and compares each rep with correct form taken from published exercise standards. Its output is a list of detected issues that the coach feedback component turns into messages.

## Function

The form analysis system:

1. **Detects the body:** MediaPipe Pose finds 33 body landmarks in each frame.
2. **Measures the pose:** Turns the landmarks into 12 joint angles plus on-screen measures such as limb tilt, hand height and feet width, all scaled by torso length so the camera's distance doesn't matter (`features.py`, `checkpoints.py`).
3. **Finds each checkpoint:** An XGBoost model trained on Kaggle's labeled poses tells which of 10 positions a frame shows, for example the bottom of a squat (`modeling/checkpoint_model.ipynb`).
4. **Counts reps:** A rep counts only when both the motion signal and the up/down model agree (`reps.py`).
5. **Checks form:** At each checkpoint, `issue_flagger.py` compares the pose with the 18 rules in `data/form_reference_rules.csv`, sourced from US Army, Air Force and Marine Corps fitness standards, ACSM, LearnOpenCV and a published knee study. Each issue gets a severity from 0 to 1 based on how far past the limit it is. Trained form models (`form_models.py`) also judge each rep, for example squat depth.
6. **Hands off to feedback:** Saves the issues as dictionaries, which `live.py` writes to a `_form.json` file.

## Results

Features mattered more than the model: adding the on-screen measures raised pose accuracy from 76% to 87%, while random forest and XGBoost stayed within a point of each other. With the exercise known, the up/down position is right about 96% of the time.

Form models, trained on QEVD's labels (5-fold cross-validation; the range covers XGBoost and random forest):

| Form question | Balanced accuracy | Chance |
|---|---|---|
| Squat stance | 0.84–0.87 | 0.33 |
| Squat back | 0.79–0.82 | 0.50 |
| Push-up hands | 0.73–0.77 | 0.33 |
| Squat depth | 0.67–0.69 | 0.33 |
| Push-up depth | 0.60–0.62 | 0.33 |

**Camera:** most rules need a side view, while knee valgus, jumping jacks and pull-ups need a front view. The form models learned from front-view video. `issue_flagger.py` skips any check it can't see clearly.

# II) Coach Feedback

## Overview

The coach feedback component turns users' motion analysis results into clear, helpful coaching messages. It uses information from the form analysis component to explain how users can correct their movements and encourage them throughout their workouts.

We also explored personalized chatbot feedback. Different users may prefer different tones, such as gentle encouragement, direct instructions, or detailed explanations. The goal was to adapt the wording and level of detail to users' stated preferences and experience while keeping the correction consistent (see Feedback Personalization below).

## Function

The feedback system:

1. **Receives motion analysis results:** Uses the exercise, rep number, detected form issues and their severity from the form analysis component.
2. **Decides when to speak:** A gate (`FeedbackController`) ignores one-off flickers (an issue must show up in 2 of the last 3 checks), waits at least 4.5 s between messages (coaches in QEVD speak about every 4.7 s), and doesn't repeat the same correction within 15 s.
3. **Provides actionable corrections:** Translates detected issues into specific instructions that explain what users should adjust in their next movement. Retrieval finds the 3 closest real coach messages (filtered by message type, exercise and body part, then ranked by MiniLM embedding similarity), and Claude writes one short new message in their style.
4. **Offers encouragement:** Recognizes good form, effort, and improvement to help users stay motivated, for example when a corrected issue goes away. Praise reuses a real coach message.
5. **Personalizes chatbot responses (future work):** Matching feedback tone and detail to users' preferences, experience levels, and workout goals.

## Example

From our test runs:

| Step | Example |
|---|---|
| Detected issue | Squat, knee: knees caving inward (severity 0.8) |
| Closest real coach message | "More space between your knees and deeper squats, please." |
| Message Claude wrote | "Spread the floor apart with your feet to drive knees outward." |

## Coaching data

`eda/coach_feedback.ipynb` loads the Qualcomm Exercise Video Dataset (QEVD) benchmark, extracts coaching messages, video identifiers, frame ranges, and event types, and labels each message by category, body part, tone and exercise. The result is `data/coach_feedback_final.csv` (2,013 feedback messages), which retrieval searches. `modeling/feedback_model.ipynb` builds and tests the gate, retrieval and generation.

## Feedback Personalization

We planned to let users choose a coaching style, such as gentle encouragement or direct instructions. This version keeps one coach voice: Claude writes corrections in the style of the retrieved real coach messages, and praise reuses real messages from QEVD. Personalization is future work. The simplest next step is a tone option that adds one line to the Claude prompt, so the correction stays the same while the wording changes.

# Setup

**In Colab (easiest):** open `app.ipynb` in Colab and add your Claude API key as a Colab secret named `client`. Everything else installs when you run the notebook.

**On your computer** (Python 3.14):

```
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Don't install `opencv-python`: MediaPipe brings its own OpenCV, and two copies conflict.

QEVD is research-only, so it isn't in the repo. To retrain the form models, download QEVD and run `python qevd_to_table.py` to build `qevd_data/`.

# Usage

- **App:** run all cells in `app.ipynb`. The last cell prints a link. Upload or record a video, pick the exercise, and press Submit to get the video with your skeleton, the rep count and the coach's messages.
- **Live webcam:** run `python live.py --exercise squats`, then click Start. Add `--video "path/to/clip.mov"` to use a recording instead. Results are saved to `Video Output/`.
- **Retrain:** run `modeling/checkpoint_model.ipynb` for the checkpoint model, or `python form_models.py` for the form models (needs `qevd_data/`).
- **Tests:** `python -m unittest discover tests`

# Repo Layout

| Path | What's there |
|---|---|
| `app.ipynb` | The Gradio app |
| `live.py`, `feedback.py`, other `.py` files | Shared code: `live.py` runs the form side, `feedback.py` the feedback side |
| `modeling/` | Training and evaluation notebooks |
| `eda/` | Data exploration notebooks |
| `data/` | Rule sheet and labeled coach messages |
| `models/` | Saved trained models |
| `tests/` | Unit tests for the form checks |
| `web/` | A separate browser version (JavaScript) |
| `archive/` | Old files, kept for reference |

# Data Sources

- [Kaggle exercise recognition](https://www.kaggle.com/datasets/muhannadtuameh/exercise-recognition): labeled poses for the checkpoint model
- [QEVD](https://www.qualcomm.com/developer/software/qevd-dataset) (Qualcomm, research use only): form labels and real coach messages
- Form rules: US Army FM 21-20, US Air Force and Marine Corps fitness test standards, ACSM, [LearnOpenCV](https://learnopencv.com/ai-fitness-trainer-using-mediapipe), and Munro et al. (2012) for knee valgus

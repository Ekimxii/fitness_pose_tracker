# Fitness Pose Tracker

Upload a workout video and get feedback on your form: the video with your skeleton drawn on, a rep count, and short coaching messages.

Supports squats, push-ups, pull-ups, sit-ups and jumping jacks.

## How it works

1. **Form:** MediaPipe finds 33 body landmarks in each frame. We turn them into joint angles, count reps, and check each rep against a sourced rule sheet (`data/form_reference_rules.csv`) and trained form models.
2. **Feedback:** a gate decides when to speak, retrieval finds similar messages from real coaching data (QEVD), and Claude writes the final message.

## Run it

Open `app.ipynb` in Colab and run all cells. You need a Colab secret named `client` with a Claude API key. The last cell prints a link to the app.

To run the form side locally with a webcam:

```
pip install -r requirements.txt
python live.py --exercise squats
```

## Repo layout

| Path | What's there |
|---|---|
| `app.ipynb` | The Gradio app |
| `live.py`, `feedback.py`, other `.py` files | Shared code: `live.py` runs the form side, `feedback.py` the feedback side |
| `modeling/` | Training and evaluation notebooks |
| `eda/` | Data exploration notebooks |
| `data/` | Rule sheet and labeled coach messages |
| `models/` | Saved trained models |
| `tests/` | Run with `python -m unittest discover tests` |
| `web/` | A separate browser version (JavaScript) |
| `archive/` | Old files, kept for reference |

## Data

- [Kaggle exercise recognition](https://www.kaggle.com/datasets/muhannadtuameh/exercise-recognition): labeled poses for the checkpoint model
- [QEVD](https://www.qualcomm.com/developer/software/qevd-dataset) (Qualcomm, research use only): form labels and real coach messages

## Team

(add names)

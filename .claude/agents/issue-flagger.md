---
name: issue-flagger
description: Builds the algorithm that flags body-part form issues per time window and produces the issue dictionaries (exercise, checkpoint, rep, t, confidence, issues with body_part/issue/status/severity). Use once the checkpoint classifier and reference_poses.csv exist, or for anything about severity or issue detection.
tools: Read, Edit, Write, Bash, Grep, Glob
---
You build the **issue flagger**, the core of the Form track: it turns pose frames into the issue dictionaries that the Feedback track consumes.

## Inputs
- Frames: MediaPipe landmarks → angles (`features.py`).
- Checkpoint: `checkpoints.py` (from the pose-classifier agent).
- Reference ranges: `reference_poses.csv` (from the reference-researcher agent).
- QEVD form models: `form_models.py` (`load()`, `assess()`), which give per-rep answers such as "squat depth: too shallow".

If `checkpoints.py` or `reference_poses.csv` is missing, stop and say so rather than building a stand-in.

## Output
Dictionaries in exactly the format defined in `form_dict.py` (don't change it; report it if you need a change), e.g.
`{"exercise": "squats", "checkpoint": "down", "rep": 3, "t": 12.4, "confidence": 0.92, "issues": [{"body_part": "knee", "issue": "valgus", "status": "bad", "severity": 0.7}]}`

## Decisions to make and document
- **When to emit a dict:** at each checkpoint (e.g. the bottom of every rep), on a fixed time window, or both. Recommend one and explain why.
- **Severity (0–1):** e.g. how far outside the reference range the angle is, scaled by the range's width, capped at 1. Write it as one documented function.
- **Status:** good / warning / bad thresholds on severity.
- **Combining sources:** what happens when the reference range and a QEVD model disagree.
- **Confidence:** from MediaPipe landmark visibility for the joints involved; the feedback gate skips low-confidence dicts.
- **Camera view:** only flag measures valid for the current view (the `camera_view` column).

## Files you own
- `issue_flagger.py`: an `IssueFlagger` class with `update(frame_angles, t, visibility) -> dict | None` for live use, and a function that runs over a whole recorded video.
- Don't edit `live.py`; the integrator agent wires it in.

## Final report
Show sample dicts from 2–3 videos in `Videos for Testing/`, the severity formula, and known blind spots.

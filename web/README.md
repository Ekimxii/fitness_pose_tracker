# Repform web

From the `fitness_pose_tracker` directory, run:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory web
```

Open http://localhost:8000 in a modern browser. Choose an exercise, continue to the camera, and select **Enable camera**. Allow camera access and position yourself as shown in the setup tip. The workout counts a rep when you complete the movement and return to the starting position. **Finish workout** stops the camera and opens the summary. **Try demo** shows sample feedback without camera access.

The browser downloads MediaPipe Tasks Vision and its full Pose Landmarker model when you first enable the camera. Pose inference and coaching run in the browser. The site does not record or upload video. The summary lives in the current tab, and **Download feedback** saves the timestamped cues as a CSV.

`coach.mjs` adapts the rest-to-work-to-rest counter in `reps.py` and the view-specific geometry and severity checks in `issue_flagger.py`. `rules.mjs` contains the form-check rows from `form_reference_rules.csv`. The browser uses angle thresholds for rep detection. The original Python CLI uses trained classifiers, and its counts and feedback may differ.

Hero image generated with the built-in ImageGen tool from this brief: a wide editorial photo of an athlete doing a side-view bodyweight squat in a dark navy studio, with open shadow on the left for the headline.

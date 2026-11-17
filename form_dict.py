"""The issue dictionary: the one format the Form track produces and the Feedback track reads.

issue_flagger.py makes one of these at each checkpoint (e.g. the bottom of every squat); the feedback
side (FeedbackController in feedback_model.ipynb) reads them one at a time and decides whether to say
anything. Change the format only when the whole team agrees, and update both sides together.

    {"exercise": "squat", "checkpoint": "down", "rep": 3, "t": 12.4, "confidence": 0.92,
     "issues": [{"body_part": "knee", "issue": "knees caving inward", "status": "bad", "severity": 0.7},
                {"body_part": "back", "issue": None, "status": "good", "severity": 0.0}]}

Fields:
- exercise:   named as coach_feedback_final.csv names it ("squat", "pushup", "jumping jacks"), so the
              feedback side can pick out the coach messages for it. EXERCISE_NAMES translates from the
              names the code uses ("squats", "pushups", "jumping_jacks").
- checkpoint: which step of the rep, "up" or "down" (from checkpoints.py).
- rep:        which rep this is, counting from 1. 0 means before the first rep finishes.
- t:          seconds since the session started (video time, never the wall clock, so recorded videos
              and tests behave the same as live).
- confidence: 0-1, how sure we are the pose was tracked well: the lowest MediaPipe visibility among the
              joints that were measured. The feedback gate skips dicts with low confidence.
- issues:     one entry per body part that was checked.
    - body_part: one of BODY_PARTS (singular, as in coach_feedback_final.csv's body_part column).
    - issue:     what is wrong, in the words of form_reference_rules.csv ("shallow depth"), or None when
                 the body part is fine.
    - status:    "good", "warning" or "bad".
    - severity:  0-1, how far from good form (0 = good, 1 = as bad as it gets).
  A body part with good form is listed with status "good" rather than left out, so the gate can tell
  "fixed" (rule 6: praise) apart from "not checked this time".

To add a new body part, add it to BODY_PARTS below; validate() rejects anything not listed, which catches
typos like "knees".
"""
from typing import TypedDict

EXERCISE_NAMES = {   # the code's name (EXERCISES in reps.py) -> the name used in the dictionary
    "squats": "squat", "pushups": "pushup", "pullups": "pullup", "situp": "situp", "jumping_jacks": "jumping jacks",
}
EXERCISES = tuple(EXERCISE_NAMES.values())
CHECKPOINTS = ("up", "down")
STATUSES = ("good", "warning", "bad")
BODY_PARTS = ("knee", "hip", "back", "elbow", "shoulder", "hand", "foot", "head", "leg", "arm")

# What each answer of the QEVD form models (TARGETS in form_models.py) means in this format.
# Each entry is a (body_part, issue) pair; issue None means good form. Where form_reference_rules.csv
# has the same issue, its wording is used, so both sources name it the same way.
QEVD_ANSWERS = {
    ("squat depth", "too shallow"): ("hip", "shallow depth"),
    ("squat depth", "good (90°)"): ("hip", None),
    ("squat depth", "too deep"): ("hip", "too deep"),
    ("squat stance", "too narrow"): ("foot", "stance too narrow"),
    ("squat stance", "good (shoulder-width)"): ("foot", None),
    ("squat stance", "too wide"): ("foot", "stance too wide"),
    ("squat back", "not straight"): ("back", "back not straight"),
    ("squat back", "no issue"): ("back", None),
    ("push-up depth", "too shallow"): ("elbow", "not low enough"),
    ("push-up depth", "good (90°)"): ("elbow", None),
    ("push-up depth", "too deep"): ("elbow", "too deep"),
    ("push-up hands", "too narrow"): ("hand", "hands too narrow"),
    ("push-up hands", "good (shoulder-width)"): ("hand", None),
    ("push-up hands", "too wide"): ("hand", "hands too wide"),
}


class Issue(TypedDict):
    body_part: str
    issue: str | None
    status: str
    severity: float


class FormDict(TypedDict):
    exercise: str
    checkpoint: str
    rep: int
    t: float
    confidence: float
    issues: list[Issue]


def make_issue(body_part, issue, status, severity):
    """One entry of a dict's issues list, checked."""
    entry = {"body_part": body_part, "issue": issue, "status": status, "severity": float(severity)}
    problems = _issue_problems(entry)
    if problems:
        raise ValueError("; ".join(problems))
    return entry


def make_dict(exercise, checkpoint, rep, t, confidence, issues=()):
    """A whole issue dictionary, checked."""
    d = {"exercise": exercise, "checkpoint": checkpoint, "rep": int(rep), "t": float(t),
         "confidence": float(confidence), "issues": list(issues)}
    validate(d)
    return d


def problems(d):
    """Everything wrong with a dict, as a list of messages (empty if it's fine)."""
    found = []
    expected = set(FormDict.__annotations__)
    if not isinstance(d, dict):
        return [f"expected a dict, got {type(d).__name__}"]
    if missing := expected - d.keys():
        found.append(f"missing fields: {sorted(missing)}")
    if extra := d.keys() - expected:
        found.append(f"unknown fields: {sorted(extra)}")
    if d.get("exercise") not in EXERCISES:
        found.append(f"exercise {d.get('exercise')!r} is not one of {EXERCISES}")
    if d.get("checkpoint") not in CHECKPOINTS:
        found.append(f"checkpoint {d.get('checkpoint')!r} is not one of {CHECKPOINTS}")
    rep = d.get("rep")
    if not isinstance(rep, int) or isinstance(rep, bool) or rep < 0:
        found.append(f"rep must be a whole number >= 0, got {rep!r}")
    t = d.get("t")
    if not _is_number(t) or t < 0:
        found.append(f"t must be seconds >= 0, got {t!r}")
    if not _in_unit_range(d.get("confidence")):
        found.append(f"confidence must be between 0 and 1, got {d.get('confidence')!r}")
    issues = d.get("issues")
    if not isinstance(issues, list):
        found.append(f"issues must be a list, got {type(issues).__name__}")
    else:
        for i, entry in enumerate(issues):
            found += [f"issues[{i}]: {p}" for p in _issue_problems(entry)]
        parts = [e.get("body_part") for e in issues if isinstance(e, dict)]
        if dupes := sorted({p for p in parts if parts.count(p) > 1}):
            found.append(f"body parts listed more than once: {dupes}")
    return found


def validate(d):
    """Raise ValueError listing every problem if d isn't a valid issue dictionary."""
    found = problems(d)
    if found:
        raise ValueError("invalid issue dictionary: " + "; ".join(found))


def bad_issues(d):
    """The entries that are actual problems (status warning or bad), worst first."""
    return sorted((e for e in d["issues"] if e["status"] != "good"), key=lambda e: e["severity"], reverse=True)


def issue_key(d, entry):
    """What makes two issues "the same issue" for cooldowns and the 2-of-last-3 rule."""
    return (d["exercise"], entry["body_part"], entry["issue"])


def _issue_problems(entry):
    if not isinstance(entry, dict):
        return [f"expected a dict, got {type(entry).__name__}"]
    found = []
    expected = set(Issue.__annotations__)
    if missing := expected - entry.keys():
        found.append(f"missing fields: {sorted(missing)}")
    if extra := entry.keys() - expected:
        found.append(f"unknown fields: {sorted(extra)}")
    if entry.get("body_part") not in BODY_PARTS:
        found.append(f"body_part {entry.get('body_part')!r} is not one of {BODY_PARTS} (add it to BODY_PARTS if new)")
    issue, status, severity = entry.get("issue"), entry.get("status"), entry.get("severity")
    if issue is not None and not (isinstance(issue, str) and issue.strip()):
        found.append(f"issue must be text or None, got {issue!r}")
    if status not in STATUSES:
        found.append(f"status {status!r} is not one of {STATUSES}")
    if not _in_unit_range(severity):
        found.append(f"severity must be between 0 and 1, got {severity!r}")
    if status == "good" and issue is not None:
        found.append(f"status 'good' needs issue None, got {issue!r}")
    if status in ("warning", "bad") and issue is None:
        found.append(f"status {status!r} needs an issue")
    return found


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _in_unit_range(x):
    return _is_number(x) and 0 <= x <= 1


EXAMPLE = make_dict("squat", "down", 3, 12.4, 0.92, [
    make_issue("knee", "knees caving inward", "bad", 0.7),
    make_issue("back", None, "good", 0.0),
])

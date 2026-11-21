"""Issue flagger: compares the pose at each checkpoint with form_reference_rules.csv.

    frames -> CheckpointTracker (checkpoints.py) -> the pose at each checkpoint -> the rules for that
    exercise and position -> an issue dictionary (form_dict.py) for the Feedback track

Each rule in the sheet is one measurement (an angle or a ratio, taken from the landmarks as they appear
on screen) with a good range. Outside the range, the rule names the issue ("shallow depth"), and how far
outside sets the severity:

    severity = how far past the limit / SEVERITY_SPAN, capped at 1
    e.g. squat depth must be at least 75 deg; a squat that reaches 60 deg is 15 past, so 15 / 30 = 0.5
    (form_dict.BAD_FROM, 1/3, is where the feedback side starts treating it as worth correcting)

The sheet's limits, issue names, body parts and camera views are read from the CSV, so they can be
tuned there. How each rule's number is measured is in MEASURES below, by rule_id: a new rule in the
sheet needs a measure added here (until then it is skipped, with a warning).

Nothing is judged unless the camera sees the whole body (shoulder to ankle), and a rule is only checked
from the view it is written for (side or front) and with its own joints visible: the sheet's limits
assume a level camera with the whole body in the picture. From the side only the side facing the camera
is measured; from the front both sides are, and the worse one counts.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import checkpoints
import form_dict
from checkpoints import CheckpointTracker, screen_angle, tilt
from features import LANDMARKS

HERE = Path(__file__).resolve().parent
RULES = HERE / "data" / "form_reference_rules.csv"
SEVERITY_SPAN = {"deg": 30.0, "ratio": 0.5}   # this far past a limit (or further) is severity 1
MIN_VISIBILITY = 0.6      # a rule is skipped if MediaPipe isn't at least this sure of every joint it uses
FRONT_ABOVE = 0.4         # shoulders this many torso lengths apart on screen (or more) = facing the camera

CHECK_COLUMNS = ["rule_id", "check", "body_part", "side", "value", "min", "max", "unit", "issue", "severity",
                 "visibility"]   # check_pose's table, before view, checkpoint, rep and t are added


class Body:
    """One pose, read from one side: body("knee") is that side's knee as an x, y point (y points down).

    It remembers which landmarks it was asked for, so we know which ones a measurement relied on.
    """

    def __init__(self, xy, side):
        self.xy, self.side, self.used = xy, side, set()

    def __call__(self, joint, side=None):
        i = LANDMARKS.index(joint if joint in LANDMARKS else f"{side or self.side}_{joint}")
        self.used.add(i)
        return self.xy[i]

    def mid(self, joint):
        """Halfway between the left and right joint."""
        return (self(joint, "left") + self(joint, "right")) / 2

    def torso(self):
        """Torso length (mid-shoulder to mid-hip), the yardstick for ratios."""
        return np.linalg.norm(self.mid("shoulder") - self.mid("hip"))


# --- The measurements. Each takes a Body and returns one number, in the rule's unit. ---

def thigh_tilt(body):   # 0 standing, 90 thigh level with the floor, more = hips below the knees
    return tilt(body("knee"), body("hip"))


def torso_tilt(body):   # 0 upright
    return tilt(body("hip"), body("shoulder"))


def shin_tilt(body):    # 0 shin upright; grows as the knee travels forward over the toes
    return tilt(body("ankle"), body("knee"))


def knee_cave(body):
    """How far the knee sits off the straight hip-ankle line, in degrees: positive = toward the other leg."""
    hip, knee, ankle = body("hip"), body("knee"), body("ankle")
    leg = ankle - hip

    def side_of_leg(point):   # positive on one side of the hip-ankle line, negative on the other
        return leg[0] * (point - hip)[1] - leg[1] * (point - hip)[0]

    bend = 180 - screen_angle(hip, knee, ankle)
    return bend if side_of_leg(knee) * side_of_leg(body.mid("hip")) > 0 else -bend


def elbow_angle(body):   # 180 = arm straight
    return screen_angle(body("shoulder"), body("elbow"), body("wrist"))


def body_line(body):     # 180 = shoulder, hip and ankle in one straight line
    return screen_angle(body("shoulder"), body("hip"), body("ankle"))


def knee_angle(body):    # 180 = leg straight
    return screen_angle(body("hip"), body("knee"), body("ankle"))


def shoulder_angle(body):   # 0 = arm down by the side, 180 = straight overhead
    return screen_angle(body("elbow"), body("shoulder"), body("hip"))


def torso_rise(body):
    """Sit-ups: the torso's angle off the floor. 0 lying flat, 90 upright, more = past upright toward the knees."""
    spine = body("shoulder") - body("hip")
    toward_feet = 1.0 if body("ankle")[0] >= body("hip")[0] else -1.0
    return np.degrees(np.arctan2(-spine[1], -spine[0] * toward_feet))


def feet_width(body):    # 1 = feet as far apart as the shoulders
    return (np.linalg.norm(body("ankle", "left") - body("ankle", "right"))
            / np.linalg.norm(body("shoulder", "left") - body("shoulder", "right")))


def chin_below_hands(body):
    """Pull-ups: how far the mouth (standing in for the chin) is below the wrists (the bar), in torso lengths."""
    mouth = (body("mouth_left") + body("mouth_right")) / 2
    return (mouth[1] - body.mid("wrist")[1]) / body.torso()


MEASURES = {   # rule_id in form_reference_rules.csv -> how its number is measured
    "S1": thigh_tilt, "S2": thigh_tilt, "S3": torso_tilt, "S4": shin_tilt, "S5": knee_cave,
    "P1": elbow_angle, "P2": elbow_angle, "P3": body_line,
    "U1": elbow_angle, "U2": chin_below_hands, "U3": torso_tilt,
    "T1": torso_rise, "T2": torso_rise, "T3": knee_angle,
    "J1": feet_width, "J2": shoulder_angle, "J3": feet_width, "J4": shoulder_angle,
}
VS_REST = {"S5"}   # rules measured as the change from the person's own rest pose (standing), not as a raw value


def hips_sagging_or_piked(body):
    """The sheet's P3 can't tell a sag from a pike by angle alone: is the hip below or above the shoulder-ankle line?"""
    shoulder, hip, ankle = body("shoulder"), body("hip"), body("ankle")
    along = (hip[0] - shoulder[0]) / (ankle[0] - shoulder[0] + 1e-9)
    line_y = shoulder[1] + along * (ankle[1] - shoulder[1])
    return "hips sagging" if hip[1] > line_y else "hips piked"   # y points down


WHICH_ISSUE = {"P3": hips_sagging_or_piked}   # rules whose issue name depends on more than the measurement


def load_rules(path=RULES):
    """The form checks in the sheet, one row per rule (rows that only define a checkpoint are left out)."""
    rules = pd.read_csv(path)
    rules = rules[rules["rule_type"] == "form_check"]
    unknown = sorted(set(rules["rule_id"]) - set(MEASURES))
    if unknown:
        warnings.warn(f"No measure in issue_flagger.MEASURES for rule(s) {unknown}: skipped")
    return rules[rules["rule_id"].isin(MEASURES)].reset_index(drop=True)


def camera_view(xy):
    """"front" if the person faces the camera (or has their back to it), "side" if they are side-on.

    Side-on, the shoulders line up behind each other, so on screen they are close together.
    """
    body = Body(xy, "left")
    shoulders = np.linalg.norm(body("shoulder", "left") - body("shoulder", "right"))
    return "front" if shoulders / body.torso() >= FRONT_ABOVE else "side"


def facing_side(visibility):
    """The side of the body MediaPipe sees better: the one facing the camera in a side view."""
    def seen(side):
        return np.mean([visibility[LANDMARKS.index(f"{side}_{joint}")]
                        for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")])
    return "right" if seen("right") >= seen("left") else "left"


def body_in_view(visibility):
    """Whether the camera sees the whole body: shoulder, hip, knee and ankle, on at least one side."""
    return any(min(visibility[LANDMARKS.index(f"{side}_{joint}")] for joint in ("shoulder", "hip", "knee", "ankle"))
               >= MIN_VISIBILITY for side in ("left", "right"))


def judge(rule, value):
    """Compare one measured value with one rule: (issue or None, severity 0-1). Good form is (None, 0.0)."""
    if pd.notna(rule["min"]) and value < rule["min"]:
        issue, past = rule["issue_if_below_min"], rule["min"] - value
    elif pd.notna(rule["max"]) and value > rule["max"]:
        issue, past = rule["issue_if_above_max"], value - rule["max"]
    else:
        return None, 0.0
    if pd.isna(issue):   # the sheet names no issue on this side of the range
        return None, 0.0
    severity = float(np.clip(round(past / SEVERITY_SPAN[rule["unit"]], 2), 0.01, 1.0))
    return issue, severity


def check_pose(checkpoint, exercise, rules, rest=None):
    """Judge the pose at one checkpoint against every rule that applies to it.

    `rest`: the latest rest-position Checkpoint, for the rules in VS_REST.
    Returns one row per rule that could be checked: what was measured, the limits, and the verdict.
    """
    xy, visibility = checkpoint.points[:, :2], checkpoint.visibility
    view = camera_view(xy)
    sides = ["left", "right"] if view == "front" else [facing_side(visibility)]
    applies = ((rules["exercise"] == exercise) & rules["pose"].str.contains(checkpoint.position)
               & rules["camera_view"].str.contains(view))
    rows = []
    for rule in rules[applies].to_dict("records") if body_in_view(visibility) else []:
        measure = MEASURES[rule["rule_id"]]
        verdicts = []
        for side in sides:
            body = Body(xy, side)
            value = measure(body)
            if rule["rule_id"] in VS_REST:
                if rest is None:
                    continue
                value -= measure(Body(rest.points[:, :2], side))
            seen = visibility[list(body.used)].min()
            if seen < MIN_VISIBILITY:
                continue
            issue, severity = judge(rule, value)
            if issue and rule["rule_id"] in WHICH_ISSUE:
                issue = WHICH_ISSUE[rule["rule_id"]](body)
            verdicts.append([rule["rule_id"], rule["check"], rule["body_part"], side, round(float(value), 2),
                             rule["min"], rule["max"], rule["unit"], issue, severity, round(float(seen), 2)])
        if verdicts:
            rows.append(max(verdicts, key=lambda v: v[CHECK_COLUMNS.index("severity")]))   # the worse side counts
    checks = pd.DataFrame(rows, columns=CHECK_COLUMNS).astype({"issue": object})   # object: keeps None as None
    return checks.assign(view=view, checkpoint=checkpoint.position, rep=checkpoint.rep, t=round(checkpoint.t, 2))


def issue_dict(checkpoint, exercise, checks):
    """The issue dictionary (form_dict.py) for one checkpoint, from check_pose's rows. None if nothing was checked."""
    if checks.empty:
        return None
    issues = {}
    for row in checks.sort_values("severity").to_dict("records"):   # if two rules share a body part, the worse one stays
        part = row["body_part"]
        part = part[:-1] if part.endswith("s") and part[:-1] in form_dict.BODY_PARTS else part   # "knees" -> "knee"
        issue = row["issue"] if isinstance(row["issue"], str) else None   # None for a body part that is fine
        issues[part] = form_dict.make_issue(part, issue, row["severity"] if issue else 0.0)
    confidence = checks["visibility"].min()   # the least visible joint that was measured
    return form_dict.make_dict(form_dict.EXERCISE_NAMES[exercise], checkpoint.position, checkpoint.rep,
                               round(checkpoint.t, 2), confidence, list(issues.values()))


def unchecked_hint(checkpoint, exercise, rules):
    """When nothing could be checked at a checkpoint: what the person should change. None if the sheet has no rule for it."""
    if not (rules["exercise"].eq(exercise) & rules["pose"].str.contains(checkpoint.position)).any():
        return None
    if not body_in_view(checkpoint.visibility):
        return "step back so the camera sees your whole body"
    return "face the camera" if camera_view(checkpoint.points[:, :2]) == "side" else "turn side-on to the camera"


class IssueFlagger:
    """Feed frames one at a time (live, or a recording played through); get an issue dictionary at each checkpoint.

    Only checkpoints with a rule to check produce a dictionary: for squats that is the bottom of each
    rep, for push-ups both the bottom and the top. The rest position before the first rep isn't judged
    (the person is still getting ready), but it is remembered as their own standing pose.
    """

    def __init__(self, exercise, bundle=None, rules=None):
        bundle = bundle or checkpoints.load()
        if bundle is None:
            raise FileNotFoundError("No checkpoint model yet: run checkpoint_model.ipynb first.")
        self.exercise = exercise
        self.rules = load_rules() if rules is None else rules
        self.tracker = CheckpointTracker(exercise, bundle)
        self.rest = None    # the latest rest-position checkpoint
        self.dicts = []     # every issue dictionary so far
        self.checks = []    # check_pose's rows behind them: what was measured, for tuning the sheet's limits
        self.hint = None    # if the latest checkpoint couldn't be checked: what the person should change

    def update(self, t, points, visibility):
        """Feed one frame: its (33, 3) normalized landmarks and 33 visibilities. Returns an issue dictionary or None."""
        return self._judge(self.tracker.update(t, points, visibility))

    def finish(self):
        """Call when the video ends, for the last checkpoint. Returns an issue dictionary or None."""
        return self._judge(self.tracker.finish())

    def _judge(self, checkpoint):
        if checkpoint is None:
            return None
        if checkpoint.position == self.tracker.rest:
            self.rest = checkpoint
        if checkpoint.rep == 0:
            return None
        checks = check_pose(checkpoint, self.exercise, self.rules, self.rest)
        found = issue_dict(checkpoint, self.exercise, checks)
        if found:
            self.dicts.append(found)
            self.checks.append(checks)
            self.hint = None
        else:   # keep the last hint through checkpoints the sheet has no rule for (e.g. the top of a squat)
            self.hint = unchecked_hint(checkpoint, self.exercise, self.rules) or self.hint
        return found

    def checks_table(self):
        """Every check so far as one table: rep, checkpoint, rule, measured value, limits, issue, severity."""
        return pd.concat(self.checks, ignore_index=True) if self.checks else pd.DataFrame(columns=CHECK_COLUMNS)

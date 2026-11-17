"""Tests for issue_flagger.py and the CheckpointTracker, on stick figures with known angles.

Run from the project folder:
    python -m unittest discover tests

No trained model, camera or dataset is needed: a stand-in model (FakeModel) says "down" as the thigh tilts.
"""
import unittest

import numpy as np

import form_dict
import issue_flagger as flagger
from checkpoints import FEATURES, Checkpoint, CheckpointTracker, screen_angle, tilt
from features import LANDMARKS

RULES = flagger.load_rules()
SEEN = np.ones(len(LANDMARKS))   # every landmark fully visible


def figure(**joints):
    """A (33, 3) pose from x, y points by joint name. "hip" sets both hips (a pure side view); "left_hip" one."""
    points = np.zeros((len(LANDMARKS), 3))
    points[:, :2] = (0.05, -1.3)   # everything not named sits at the head
    for name, xy in joints.items():
        for landmark in ([name] if name in LANDMARKS else [f"left_{name}", f"right_{name}"]):
            points[LANDMARKS.index(landmark), :2] = xy
    return points


def squat(depth, torso_lean=None, shin=None):
    """Side view of a squat whose thighs are `depth` degrees from upright (90 = level with the floor)."""
    torso_lean = 5 + depth / 3 if torso_lean is None else torso_lean   # 33 deg at a 85 deg squat: in the good range
    shin = 0.3 * depth if shin is None else shin
    d, lean, s = np.radians([depth, torso_lean, shin])
    hip = np.array([0.0, 0.0])
    knee = hip + 0.9 * np.array([np.sin(d), np.cos(d)])        # y points down
    ankle = knee + 0.9 * np.array([-np.sin(s), np.cos(s)])
    shoulder = hip + np.array([np.sin(lean), -np.cos(lean)])
    return figure(hip=hip, knee=knee, ankle=ankle, shoulder=shoulder,
                  elbow=shoulder + (0.3, 0), wrist=shoulder + (0.6, 0))


def front_squat(knees_in=0.0):
    """Front view at the bottom of a squat, each knee moved `knees_in` toward the other."""
    joints = {}
    for side, sign in (("left", 1), ("right", -1)):
        joints[f"{side}_shoulder"] = (sign * 0.33, -1.0)
        joints[f"{side}_elbow"] = (sign * 0.45, -0.6)
        joints[f"{side}_wrist"] = (sign * 0.45, -0.2)
        joints[f"{side}_hip"] = (sign * 0.17, 0.0)
        joints[f"{side}_knee"] = (sign * (0.17 - knees_in), 0.5)
        joints[f"{side}_ankle"] = (sign * 0.17, 1.0)
    return figure(**joints)


def pushup(hip_drop=0.0, elbow=170.0):
    """Side view of a push-up, body along x. hip_drop > 0 sags the hips toward the floor, < 0 pikes them up."""
    shoulder, ankle = np.array([0.0, 0.0]), np.array([2.4, 0.5])
    hip = (shoulder + ankle) / 2 + (0, hip_drop)
    bend = np.radians(180 - elbow)
    elbow_point = shoulder + (0.0, 0.35)
    wrist = elbow_point + 0.35 * np.array([np.sin(bend), np.cos(bend)])
    return figure(shoulder=shoulder, hip=hip, knee=(hip + ankle) / 2, ankle=ankle, elbow=elbow_point, wrist=wrist)


def check(points, exercise, position, rest=None, visibility=SEEN):
    """check_pose's rows for one pose, indexed by rule_id."""
    rest = None if rest is None else Checkpoint("up", 0, 0.0, rest, SEEN)
    return flagger.check_pose(Checkpoint(position, 1, 2.0, points, visibility), exercise, RULES, rest).set_index("rule_id")


class Geometry(unittest.TestCase):
    def test_tilt(self):
        self.assertAlmostEqual(float(tilt(np.array([0, 0]), np.array([0, -1]))), 0)      # straight above
        self.assertAlmostEqual(float(tilt(np.array([0, 0]), np.array([1, 0]))), 90)      # level
        self.assertAlmostEqual(float(tilt(np.array([0, 0]), np.array([-1, 1]))), 135)    # below

    def test_screen_angle(self):
        self.assertAlmostEqual(float(screen_angle(np.array([1, 0]), np.array([0, 0]), np.array([0, 1]))), 90, places=4)

    def test_the_stick_figures_measure_what_they_were_built_with(self):
        body = flagger.Body(squat(80, torso_lean=30, shin=20)[:, :2], "left")
        self.assertAlmostEqual(float(flagger.thigh_tilt(body)), 80)
        self.assertAlmostEqual(float(flagger.torso_tilt(body)), 30)
        self.assertAlmostEqual(float(flagger.shin_tilt(body)), 20)
        self.assertAlmostEqual(float(flagger.elbow_angle(flagger.Body(pushup(elbow=120)[:, :2], "left"))), 120, places=4)

    def test_torso_rise_for_situps(self):
        def situp(rise):
            r = np.radians(rise)
            return flagger.Body(figure(hip=(0, 0), shoulder=(-np.cos(r), -np.sin(r)), knee=(0.6, -0.5), ankle=(1.2, 0))[:, :2], "left")
        for rise in (0, 45, 90, 110):   # lying flat, halfway, upright, past upright toward the knees
            self.assertAlmostEqual(float(flagger.torso_rise(situp(rise))), rise)


class Judging(unittest.TestCase):
    def rule(self, rule_id):
        return RULES[RULES["rule_id"] == rule_id].iloc[0]

    def test_inside_the_range_is_good(self):
        self.assertEqual(flagger.judge(self.rule("S2"), 85), (None, 0.0, "good"))

    def test_severity_grows_with_the_distance_past_the_limit(self):
        depth = self.rule("S2")                                   # good from 75 to 95 deg
        self.assertEqual(flagger.judge(depth, 70), ("shallow depth", 0.17, "warning"))   # 5 past: 5 / 30
        self.assertEqual(flagger.judge(depth, 60), ("shallow depth", 0.5, "bad"))        # 15 past
        self.assertEqual(flagger.judge(depth, 20), ("shallow depth", 1.0, "bad"))        # capped at 1
        self.assertEqual(flagger.judge(depth, 110), ("too deep", 0.5, "bad"))

    def test_ratios_use_their_own_span(self):
        self.assertEqual(flagger.judge(self.rule("J1"), 0.75), ("feet not wide enough", 0.5, "bad"))   # 0.25 / 0.5


class CheckingAPose(unittest.TestCase):
    def test_good_squat_from_the_side(self):
        rows = check(squat(85), "squats", "down")
        self.assertEqual(sorted(rows.index), ["S2", "S3", "S4"])   # S5 (knee valgus) needs a front view
        self.assertTrue((rows["status"] == "good").all())

    def test_shallow_squat(self):
        row = check(squat(60), "squats", "down").loc["S2"]
        self.assertEqual((row["issue"], row["severity"], row["status"]), ("shallow depth", 0.5, "bad"))
        self.assertAlmostEqual(row["value"], 60)

    def test_knees_over_toes_and_forward_lean(self):
        rows = check(squat(85, torso_lean=60, shin=45), "squats", "down")
        self.assertEqual(rows.loc["S3", "issue"], "excessive forward lean")
        self.assertEqual(rows.loc["S4", "issue"], "knees too far past toes")
        self.assertEqual(rows.loc["S4", "severity"], 0.5)

    def test_front_view_checks_valgus_against_the_standing_pose(self):
        rows = check(front_squat(knees_in=0.1), "squats", "down", rest=front_squat())
        self.assertEqual(list(rows.index), ["S5"])                 # the side-view rules are skipped
        self.assertEqual(rows.loc["S5", "issue"], "knees caving inward")
        self.assertAlmostEqual(rows.loc["S5", "value"], 22.62, places=1)   # 2 * atan(0.1 / 0.5)
        self.assertEqual(rows.loc["S5", "status"], "bad")

    def test_knees_out_is_not_valgus(self):
        self.assertEqual(check(front_squat(knees_in=-0.1), "squats", "down", rest=front_squat()).loc["S5", "status"], "good")

    def test_valgus_needs_a_standing_pose_to_compare_with(self):
        self.assertTrue(check(front_squat(knees_in=0.1), "squats", "down").empty)

    def test_pushup_sag_and_pike_are_told_apart(self):
        self.assertEqual(check(pushup(hip_drop=0.3), "pushups", "up").loc["P3", "issue"], "hips sagging")
        self.assertEqual(check(pushup(hip_drop=-0.3), "pushups", "up").loc["P3", "issue"], "hips piked")
        self.assertEqual(check(pushup(), "pushups", "up").loc["P3", "status"], "good")

    def test_pushup_rules_depend_on_the_position(self):
        self.assertEqual(check(pushup(elbow=120), "pushups", "up").loc["P1", "issue"], "arms not fully extended")
        self.assertEqual(check(pushup(elbow=120), "pushups", "down").loc["P2", "issue"], "not low enough")
        self.assertEqual(check(pushup(elbow=80), "pushups", "down").loc["P2", "status"], "good")

    def test_a_rule_is_skipped_when_its_joints_are_hidden(self):
        hidden = SEEN.copy()
        hidden[[LANDMARKS.index("left_wrist"), LANDMARKS.index("right_wrist")]] = 0.1
        rows = check(pushup(elbow=120), "pushups", "up", visibility=hidden)
        self.assertEqual(list(rows.index), ["P3"])                 # lockout needs the wrist; the body line doesn't

    def test_nothing_is_judged_unless_the_whole_body_is_in_view(self):
        cut_off = SEEN.copy()
        cut_off[[LANDMARKS.index("left_ankle"), LANDMARKS.index("right_ankle")]] = 0.1
        self.assertTrue(check(squat(60), "squats", "down", visibility=cut_off).empty)   # not even depth, which needs no ankle
        cp = Checkpoint("down", 1, 2.0, squat(60), cut_off)
        self.assertEqual(flagger.unchecked_hint(cp, "squats", RULES), "step back so the camera sees your whole body")

    def test_hints_when_the_view_has_no_rule(self):
        self.assertEqual(flagger.unchecked_hint(Checkpoint("up", 1, 2.0, front_squat(), SEEN), "pushups", RULES),
                         "turn side-on to the camera")
        self.assertEqual(flagger.unchecked_hint(Checkpoint("up", 1, 2.0, squat(0), SEEN), "jumping_jacks", RULES),
                         "face the camera")
        self.assertIsNone(flagger.unchecked_hint(Checkpoint("up", 1, 2.0, squat(0), SEEN), "squats", RULES))   # no rule there

    def test_from_the_side_only_the_facing_side_is_measured(self):
        visibility = SEEN.copy()
        visibility[[LANDMARKS.index(f"left_{j}") for j in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")]] = 0.2
        self.assertTrue((check(squat(85), "squats", "down", visibility=visibility)["side"] == "right").all())


class TheDictionary(unittest.TestCase):
    def test_it_matches_the_shared_format(self):
        cp = Checkpoint("down", 3, 12.4, squat(60), SEEN * 0.9)
        d = flagger.issue_dict(cp, "squats", flagger.check_pose(cp, "squats", RULES))
        form_dict.validate(d)
        self.assertEqual((d["exercise"], d["checkpoint"], d["rep"], d["t"], d["confidence"]), ("squat", "down", 3, 12.4, 0.9))
        self.assertIn({"body_part": "hip", "issue": "shallow depth", "status": "bad", "severity": 0.5}, d["issues"])
        self.assertIn({"body_part": "back", "issue": None, "status": "good", "severity": 0.0}, d["issues"])

    def test_nothing_checked_means_no_dictionary(self):
        cp = Checkpoint("up", 1, 1.0, squat(0), SEEN)   # the sheet has no form check for the top of a squat
        self.assertIsNone(flagger.issue_dict(cp, "squats", flagger.check_pose(cp, "squats", RULES)))


class FakeModel:
    """Stands in for the trained model: P(squats_down) rises as the thigh tilts."""

    def get_booster(self):
        return self

    def set_param(self, params):
        pass

    def inplace_predict(self, features):
        down = np.clip(features[:, FEATURES.index("left thigh tilt")] / 75, 0.01, 0.99)
        return np.column_stack([down, 1 - down])


BUNDLE = {"model": FakeModel(), "poses": ["squats_down", "squats_up"], "features": FEATURES,
          "ranges": {"squats": {"knee": (175.0, 75.0)}}}


def workout(depths, fps=30):
    """(t, pose) frames of someone standing for a second, then doing one squat to each depth."""
    still = [0.0] * fps
    tilts = list(still)
    for depth in depths:
        tilts += list(depth * (1 - np.cos(np.linspace(0, 2 * np.pi, 2 * fps))) / 2) + still[:fps // 2]
    return [(i / fps, squat(tilt)) for i, tilt in enumerate(tilts)]


class Tracking(unittest.TestCase):
    def test_each_rep_reaches_its_checkpoints_at_its_deepest_point(self):
        tracker = CheckpointTracker("squats", BUNDLE)
        reached = [cp for t, pose in workout([85, 60]) if (cp := tracker.update(t, pose, SEEN))]
        self.assertEqual(tracker.reps, 2)
        self.assertEqual([(cp.position, cp.rep) for cp in reached], [("up", 0), ("down", 1), ("up", 1), ("down", 2)])
        deepest = [float(flagger.thigh_tilt(flagger.Body(cp.points[:, :2], "left"))) for cp in reached if cp.position == "down"]
        self.assertAlmostEqual(deepest[0], 85, delta=1.5)
        self.assertAlmostEqual(deepest[1], 60, delta=1.5)
        self.assertAlmostEqual(reached[1].t, 2.0, delta=0.1)   # 1 s standing, then 1 s down

    def test_a_dip_too_shallow_to_count_as_a_rep_is_ignored(self):
        tracker = CheckpointTracker("squats", BUNDLE)
        reached = [cp for t, pose in workout([25]) if (cp := tracker.update(t, pose, SEEN))]
        self.assertEqual(tracker.reps, 0)
        self.assertEqual([cp.position for cp in reached], ["up"])


class EndToEnd(unittest.TestCase):
    def test_frames_in_dictionaries_out(self):
        flag = flagger.IssueFlagger("squats", BUNDLE, RULES)
        dicts = [d for t, pose in workout([85, 60, 85]) if (d := flag.update(t, pose, SEEN))]
        self.assertIsNone(flag.finish())   # squats have no rule for the standing position
        self.assertEqual([d["rep"] for d in dicts], [1, 2, 3])
        for d in dicts:
            form_dict.validate(d)
            self.assertEqual((d["exercise"], d["checkpoint"]), ("squat", "down"))
        self.assertEqual([form_dict.bad_issues(d) for d in (dicts[0], dicts[2])], [[], []])
        worst = form_dict.bad_issues(dicts[1])[0]
        self.assertEqual((worst["body_part"], worst["issue"], worst["status"]), ("hip", "shallow depth", "bad"))
        self.assertAlmostEqual(worst["severity"], 0.5, delta=0.05)
        self.assertEqual(len(flag.checks_table()), 9)   # 3 reps x the 3 side-view squat rules


if __name__ == "__main__":
    unittest.main()

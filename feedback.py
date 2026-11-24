"""Feedback: turns the issue dictionaries from the form side into coach messages.

The pieces, in the order they run:
- load_corpus: the real coach messages from QEVD and their embeddings.
- FeedbackController (the gate): for each issue dictionary, decides whether to speak and what kind of
  message to give (a correction, praise for fixing an issue, or general encouragement).
- retrieve_top3: the 3 real coach messages closest to the gate's decision.
- make_message: Claude writes one short new message, using those 3 as style examples.
- send_output_to_ui, save_for_ui: package the messages for the app.

Developed and tested in modeling/feedback_model.ipynb. The app (app.ipynb) imports it from here.
"""
import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "data" / "coach_feedback_final.csv"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus(path=CORPUS):
    """The real coach messages to retrieve from. Returns (feedback_df, feedback_embeddings, encoder).

    feedback_df has one row per feedback message (type == "feedback"), in the same order as the embeddings.
    """
    from sentence_transformers import SentenceTransformer   # slow to import, so only when it's needed

    encoder = SentenceTransformer(ENCODER)
    df = pd.read_csv(path)
    feedback_df = df[df["type"] == "feedback"].copy().reset_index(drop=True)
    feedback_embeddings = encoder.encode(feedback_df["text"].tolist(), convert_to_numpy=True,
                                         show_progress_bar=False)
    return feedback_df, feedback_embeddings, encoder


# Severity from here up is worth correcting. Keep in sync with BAD_FROM in form_dict.py
# (1/3 = 10 degrees past a rule's limit). The dictionary has no "status" key any more.
BAD_FROM = 1 / 3

def is_bad(issue):
    """An issue entry worth correcting: it names an issue with severity BAD_FROM or more."""
    return issue.get("issue") is not None and issue.get("severity", 0) >= BAD_FROM

class FeedbackController:
    """The gate. Feed it one issue dictionary at a time with process(); it returns a decision
    (constructive, recovery or positive) when it's time to speak, or None to stay quiet."""

    def __init__(
        self,
        window_size=3,        # keep how many recent dicts
        min_occurrences=2,      # min number of times an issue must appear within the window to trigger feedback
        global_cooldown=4.5,     # min time between any two feedback messages
        issue_cooldown=15.0,     # min time before repeating feedback about the same issue
        recovery_frames=2,      # no. of consecutive dictionaries in which an issue must be absent to trigger recovery feedback
        praise_after_clean_frames=3, # no. of consecutive dictionaries with no detected issues before triggering general positive feedback
    ):
        # Store recent FORM dictionaries
        self.history = deque(maxlen=window_size)

        # Decision algorithm parameters
        self.min_occurrences = min_occurrences
        self.global_cooldown = global_cooldown
        self.issue_cooldown = issue_cooldown
        self.recovery_frames = recovery_frames
        self.praise_after_clean_frames = praise_after_clean_frames

        # Track feedback timing
        self.last_message_time = float("-inf")
        self.last_issue_message_time = {}

        # Track issues that have already received corrections
        self.active_issues = {}

        # Store issues that are ready for recovery encouragement
        self.pending_recoveries = set()

        # Track consecutive clean frames
        self.clean_frames = 0

        # Track current exercise
        self.current_exercise = None

    def process(self, frame_dict):
        """Process one FORM dictionary and return a feedback decision or None."""

        now = float(frame_dict["t"])
        exercise = frame_dict["exercise"]

        # Reset exercise-specific tracking when the exercise changes
        if self.current_exercise != exercise:
            self.history.clear()
            self.active_issues.clear()
            self.pending_recoveries.clear()
            self.clean_frames = 0
            self.current_exercise = exercise

        # Get all currently detected bad issues
        current_issues = {}

        for issue in frame_dict.get("issues", []):
            if not is_bad(issue):
                continue

            key = (
                exercise,
                issue["body_part"],
                issue["issue"],
            )

            current_issues[key] = issue

        # Add the current dictionary to recent history
        self.history.append(frame_dict)

        # Count how many dictionaries contain each issue
        issue_counts = {}

        for item in self.history:
            frame_keys = set()

            for issue in item.get("issues", []):
                if not is_bad(issue):
                    continue

                key = (
                    item["exercise"],
                    issue["body_part"],
                    issue["issue"],
                )

                frame_keys.add(key)

            # Count each issue only once per dictionary
            for key in frame_keys:
                issue_counts[key] = issue_counts.get(key, 0) + 1

        # Find persistent issues that are still present now
        persistent_issues = []

        for key, issue in current_issues.items():
            if issue_counts.get(key, 0) >= self.min_occurrences:
                persistent_issues.append((key, issue))

        # Update recovery tracking for previously corrected issues
        for key in list(self.active_issues):
            if key in current_issues:
                # The issue is still present
                self.active_issues[key]["gone_frames"] = 0
            else:
                # The issue has disappeared
                self.active_issues[key]["gone_frames"] += 1

                if (
                    self.active_issues[key]["gone_frames"]
                    >= self.recovery_frames
                ):
                    self.pending_recoveries.add(key)
                    del self.active_issues[key]

        # Update clean-frame counter
        if current_issues:
            self.clean_frames = 0
        else:
            self.clean_frames += 1

        # Rule 5: Select the most severe persistent issue
        if persistent_issues:
            key, selected_issue = max(
                persistent_issues,
                key=lambda pair: pair[1].get("severity", 0),
            )

            last_issue_time = self.last_issue_message_time.get(
                key, float("-inf")
            )

            # Check per-issue cooldown
            if now - last_issue_time >= self.issue_cooldown:

                # Check global cooldown
                if now - self.last_message_time >= self.global_cooldown:

                    self.last_issue_message_time[key] = now
                    self.last_message_time = now

                    # Track this corrected issue
                    self.active_issues[key] = {"gone_frames": 0}

                    # It is no longer waiting for recovery feedback
                    self.pending_recoveries.discard(key)

                    return {
                        "action": "constructive",
                        "exercise": key[0],
                        "body_part": key[1],
                        "issue": key[2],
                        "severity": selected_issue.get("severity", 0),
                        "query": (
                            f"{key[0]} {key[1]} {key[2]} "
                            "form correction"
                        ),
                    }

            # A persistent issue is present, but its cooldown has not expired
            return None

        # Rule 6: Praise after a previously corrected issue disappears
        if self.pending_recoveries:
            if now - self.last_message_time >= self.global_cooldown:

                key = self.pending_recoveries.pop()

                self.last_message_time = now

                return {
                    "action": "recovery",
                    "exercise": key[0],
                    "body_part": key[1],
                    "issue": key[2],
                    "query": (
                        f"{key[0]} {key[1]} "
                        f"{key[2]} improvement encouragement"
                    ),
                }

        # Rule 7: General encouragement after consecutive clean frames
        if self.clean_frames >= self.praise_after_clean_frames:
            if now - self.last_message_time >= self.global_cooldown:

                self.clean_frames = 0
                self.last_message_time = now

                return {
                    "action": "positive",
                    "exercise": exercise,
                    "query": (
                        f"{exercise} good form "
                        "positive encouragement"
                    ),
                }

        # Rule 8: Otherwise, stay silent
        return None


def retrieve_top3(decision, feedback_df, feedback_embeddings, encoder):
    """Retrieve the 3 most relevant feedback messages based on a controller decision.
    decision: Dictionary returned by controller.process()
    feedback_df: Filtered feedback dataset
    feedback_embeddings: Precomputed embeddings of feedback_df messages
    encoder: SentenceTransformer model"""

    # If the controller decides not to give feedback --> return an empty DataFrame
    if decision is None:
        return feedback_df.iloc[0:0].copy()

    # Get the type of feedback we want to retrieve
    action = decision["action"]

    # Only retrieve actual feedback messages
    candidates = feedback_df.copy()

    # Filter by feedback category based on the action.
    if action == "constructive":
        # For constructive feedback, prioritize form corrections
        preferred = candidates[
            candidates["primary_category"].fillna("").str.lower()
            == "form_correction"
        ]

    else:
        # For recovery and positive feedback, prioritize encouragement
        preferred = candidates[
            candidates["primary_category"].fillna("").str.lower()
            == "encouragement"
        ]

        # If no encouragement messages are found, try compliments instead
        if preferred.empty:
            preferred = candidates[
                candidates["feedback_kind"].fillna("").str.lower()
                == "compliment"
            ]

    # If the preferred category has results, use it
    # Otherwise, keep all feedback candidates
    if not preferred.empty:
        candidates = preferred

    # Filter by exercise when matching messages are available.
    exercise = decision["exercise"].lower()

    exercise_match = candidates[
        candidates["exercise"].fillna("").str.lower() == exercise
    ]

    if not exercise_match.empty:
        candidates = exercise_match

    # Filter by body part when matching messages are available
    if decision.get("body_part"):
        body_part = decision["body_part"].lower()

        body_part_match = candidates[
            candidates["body_part"]
            .fillna("")
            .str.lower()
            .str.contains(body_part, regex=False)
        ]

        if not body_part_match.empty:
            candidates = body_part_match

    # Return an empty DataFrame if no candidates are available
    if candidates.empty:
        return candidates

    # Convert the current search query into an embedding
    query_embedding = encoder.encode(
        [decision["query"]],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # Get the corresponding precomputed embeddings
    # The feedback_df row order must match feedback_embeddings.
    row_positions = candidates.index.to_numpy()
    candidate_embeddings = feedback_embeddings[row_positions]

    # Normalize candidate embeddings
    # The dot product of normalized vectors equals cosine similarity
    candidate_embeddings = candidate_embeddings / np.maximum(
        np.linalg.norm(candidate_embeddings, axis=1, keepdims=True),
        1e-12
    )

    # Calculate cosine similarity between the query and candidates
    similarities = candidate_embeddings @ query_embedding

    # Select the 3 candidates with the highest similarity scores
    top_positions = np.argsort(similarities)[::-1][:3]

    top3 = candidates.iloc[top_positions].copy()
    top3["similarity"] = similarities[top_positions]

    return top3


# ---------------------------------------------------------------
# Build the prompt: combine the detected issue (content) with the
# retrieved coach phrases (style examples) into one instruction
# ---------------------------------------------------------------
def build_prompt(issue, retrieved):
    """The instruction for Claude: what happened (a decision from the gate) plus real coach messages as style examples."""
    # Format the retrieved phrases as a bulleted list of examples
    examples = "\n".join(f"- {r}" for r in retrieved)

    # Identify the type of feedback
    action = issue.get("action", "constructive")

    if action == "constructive":
        return f"""You are a workout coach giving short real-time form feedback.

Detected issue:
- Exercise: {issue['exercise']}
- Body part: {issue['body_part']}
- Problem: {issue['issue']}
- Severity: {issue.get('severity', 0):.2f} (0 = good form, 1 = worst)

Example coach cues from real data:
{examples}

Write ONE new cue (max 12 words) in a similar style.
Only address the detected issue. Do not copy an example word for word."""

    elif action == "recovery":
        return f"""You are a workout coach giving short real-time feedback.

The user has corrected a previous form issue:
- Exercise: {issue['exercise']}
- Body part: {issue['body_part']}

Example coach cues from real data:
{examples}

Write ONE short encouraging recovery message (max 12 words).
Do not introduce a new form correction."""

    else:
        return f"""You are a supportive workout coach.

The user is demonstrating good form during:
- Exercise: {issue['exercise']}

Example coach cues from real data:
{examples}

Write ONE short positive encouragement (max 12 words).
Do not introduce a form correction."""


# ---------------------------------------------------------------
# Generate the feedback: send the prompt to the model and return
# the new cue as a clean string
# ---------------------------------------------------------------
def generate_feedback(issue, retrieved, client):
    """Ask Claude for one short message. `client` is an anthropic.Anthropic client."""
    resp = client.messages.create(
        model="claude-sonnet-5-5",   # which model writes the cue
        max_tokens=60,               # cues are short, so keep the cap low
        messages=[{"role": "user", "content": build_prompt(issue, retrieved)}],
    )
    # Pull the text out of the response and strip extra whitespace
    return resp.content[0].text.strip()


def make_message(decision, top3, client):
    """One coach message for a gate decision, written by Claude in the style of the retrieved messages.

    decision: what FeedbackController.process returned. top3: what retrieve_top3 returned.
    Returns the message text, or None when there is no decision.
    """
    if decision is None:
        return None
    retrieved = [] if top3 is None or top3.empty else top3["text"].dropna().tolist()
    return generate_feedback(decision, retrieved, client)


def send_output_to_ui(frame_dict, decision, retrieved, client):
    """One row for the app: time, rep, issue and the message. None when the gate stayed quiet."""
    if decision is None:
        return None
    return {
        "time": frame_dict.get("t"),
        "rep": frame_dict.get("rep"),
        "issue": decision.get("issue", "general encouragement"),
        "message": make_message(decision, retrieved, client),
    }


def save_for_ui(events, path="_feedback.json"):
    """Save the app's rows as JSON. Writes to a temporary file first, so a half-written file is never read."""
    temp_path = path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    # Replace the old file after the new file is fully written
    os.replace(temp_path, path)

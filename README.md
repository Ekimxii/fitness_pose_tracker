# name

brief description

I) Form Analysis

# II) Coach Feedback

## Overview

The coach feedback component aims to turn users' motion analysis results into clear, helpful coaching messages. It will use information from the form analysis component to explain how users can correct their movements and encourage them throughout their workouts.

We also plan to explore personalized chatbot feedback. Different users may prefer different tones, such as gentle encouragement, direct instructions, or detailed explanations. The goal is to adapt the wording and level of detail to users' stated preferences and experience while keeping the correction consistent.

## Function

The planned feedback system will:

1. **Receive motion analysis results:** Use exercise type, joint positions, body angles, repetition information, and detected form issues from the form analysis component.
2. **Provide actionable corrections:** Translate detected issues into specific instructions that explain what users should adjust in their next movement.
3. **Offer encouragement:** Recognize good form, effort, and improvement to help users stay motivated.
4. **Personalize chatbot responses:** Explore how feedback tone and detail can match different users' preferences, experience levels, and workout goals.

Example(add later)

Currently, `coach_feedback.ipynb` loads the Qualcomm Exercise Video Dataset (QEVD) benchmark and extracts coaching messages, video identifiers, frame ranges, and event types into `qevd_feedback_text.csv`. This supports exploration of existing coaching feedback; motion-based feedback generation and chatbot personalization are planned features.

## Setup

## Usage

## Feedback Personalization




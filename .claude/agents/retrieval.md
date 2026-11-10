---
name: retrieval
description: Builds the retrieval and generation pipeline that turns a feedback-gate decision into a coaching message: filter coach messages by exercise and body part, embed, take the cosine top-3, then generate. Use for feedback_model.ipynb work, embeddings, RAG, or the generation prompt.
tools: Read, Edit, Write, Bash, Grep, Glob
---
You build **retrieval and generation** for the Feedback track (`feedback_model.ipynb` is the primary notebook; its comment cells outline the steps).

## Pipeline
1. **Corpus:** `coach_feedback_final.csv` (the final dataframe from `coach_feedback.ipynb`). Read its columns first and use the existing labels: exercise, feedback type/tone, constructive or not.
2. **Filter first** by exercise, then body part / issue, then message kind (constructive vs. praise vs. positive, matching the gate's `kind`). Fall back to a broader filter when nothing matches, and log when that happens.
3. **Embed** the messages once with `sentence-transformers` (a small model such as `all-MiniLM-L6-v2`) and cache the embeddings to disk; don't re-embed on every query.
4. **Query:** build a short text from the decision (e.g. "squats knee valgus constructive") → cosine similarity → top 3.
5. **Generate:** a prompt template that turns the top 3 into one short spoken cue (≤ 12 words, one cue at a time). Keep generation behind a single `generate(prompt)` function, because the team hasn't chosen the model yet (Hugging Face vs. a hosted API). Until then, provide a no-model fallback that returns the best retrieved message.

## Files you own
- `retrieval.py`: `Retriever` (load corpus, embed, `top_k(decision, k=3)`) and `make_cue(decision)`.
- `feedback_model.ipynb`: only to fill in its existing outline cells with short calls into `retrieval.py`. Keep the logic in the module.
- The input format is the decision from `feedback_gate.py`. Until that exists, test with fake decisions.

## Data rules
QEVD is research-only (see `.gitignore`). Don't create or commit new files derived from QEVD (embeddings caches, CSV exports). Write caches to a gitignored path and add it to `.gitignore` if needed.

## Final report
Show 5 example decisions with their top-3 retrieved messages and the final cue, the latency per query, and any new dependencies (add them to `requirements.txt` with exact versions).

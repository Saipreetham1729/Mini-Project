# ContextGuard — Project Review Demo Script

A suggested walkthrough for demonstrating ContextGuard to an evaluator, in order, with the
research point each step is making.

## Before the demo

- [ ] Backend running (`uvicorn app.main:app --reload --app-dir .`), `.env` has a real
      `LLM_API_KEY` set (not `mock` — a mock answer won't show anything meaningful)
- [ ] Frontend running (`npm run dev`)
- [ ] A test PDF ready — something with enough factual content that a question can genuinely be
      answered correctly or incorrectly (a short technical document, a Wikipedia-style article
      exported to PDF, or your own course notes)
- [ ] Ideally, at least 10-15 benchmark claims already labeled (see below), so the Dashboard has
      real numbers to show instead of an empty state
- [ ] NLI and embedding models already downloaded once beforehand (don't do this live — the first
      download takes 30-90 seconds and adds nothing to watch)

## Walkthrough

**1. Home page (30 seconds)**
State the research question out loud: *"Which faithfulness metrics best detect actual
hallucinations in RAG-generated answers?"* Point out this is explicitly framed as a comparative
study, not a chatbot demo.

**2. Experiment page (2 minutes)**
Upload the test PDF, ask a question you know the document can *partially* but not *fully*
answer — this is the question most likely to produce an interesting mix of supported and
unsupported claims, which is more demo-worthy than a perfectly clean answer. Narrate the pipeline
steps as they run (upload → process → embed → retrieve & generate).

**3. Verification page (2 minutes)**
This is the core deliverable. Walk through the claim list: point at a `supported` claim and its
matched evidence page number; point at an `unsupported` or `partially_supported` claim and read
out its `likely_failure_type` + reasoning. Emphasize the reasoning is generated from concrete
pipeline signals (retrieval score, similarity score) — not a black-box guess.

**4. Mitigation page (2 minutes)**
Click "Apply mitigation." Show the before/after support ratio side by side. If the ratio improved,
say so plainly; if it didn't (which is itself a valid, honest research finding), say that too —
don't oversell a mitigation strategy that didn't help on this particular question.

**5. Evaluation page (2 minutes)**
Show the three per-answer metric scores (ROUGE-L, NLI, LLM-judge) for this one answer. If NLI
shows "Not evaluated yet" because the model hasn't been downloaded, explain that this is
deliberate design — the system refuses to fabricate a number rather than silently failing.

**6. Dashboard page (2-3 minutes) — the actual research result**
This is where the comparative study lives. Show:
- The status/failure-type distributions across everything you've benchmarked so far
- The Precision/Recall/F1 comparison table across all four methods

Say plainly what the numbers show — e.g. "ROUGE-L had lower recall than NLI on this sample,
consistent with it missing paraphrased hallucinations" — or whatever your actual labeled data
shows. **Do not state a stronger conclusion than your sample size supports** — with a small
benchmark (under ~30 labeled claims), say so explicitly rather than overstating confidence.

## Anticipated questions and honest answers

- **"How is 'likely failure type' verified as correct?"** — It isn't, and the system says so
  explicitly. It's a heuristic classification from pipeline signals, framed throughout as
  "likely"/"suspected," never claimed as ground truth. Point to `failure_classifier.py`'s
  docstring if asked for detail.
- **"Why only three core metrics, not all five from the original plan?"** — Explain the scope-cut
  decision: RAGAS and self-consistency add real LLM-call cost and complexity beyond what's
  reasonable for a solo mini-project; they're documented as future work.
- **"How much of this is fabricated for the demo?"** — None of it. Point to the `null`/"Not
  evaluated yet" behavior as evidence the system is built not to fabricate, and to the pytest
  suite (`pytest tests/ -v`, 32 passing tests) as evidence of tested correctness.
- **"What would you do with more time?"** — RAGAS + self-consistency as the remaining two metrics,
  a larger labeled benchmark, per-claim (not just per-answer) mitigation, and possibly extending
  to multi-document retrieval.

## If something breaks live

- If the LLM API call fails (rate limit, network), that's a legitimate moment to explain the
  provider abstraction — switch `LLM_PROVIDER` to `mock` in `.env` and restart to show the
  pipeline plumbing still works end-to-end, then explain what a real key would return.
- If a metric shows "Not evaluated yet" unexpectedly, that's not a bug to apologize for — it's the
  system correctly refusing to fabricate a result. Say so.

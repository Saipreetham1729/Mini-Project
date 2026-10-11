# ContextGuard

[![CI](https://github.com/your-username/contextguard/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/contextguard/actions/workflows/ci.yml)

A research system for evaluating hallucination detection methods in Retrieval-Augmented
Generation. **Evaluating Hallucination Detection Methods in Retrieval-Augmented Generation: A
Comparative Study of Faithfulness Metrics.**

The RAG pipeline here is a controlled experimental environment, not a general-purpose document
chatbot. The research question is: *which faithfulness/grounding metrics best detect actual
hallucinations in RAG-generated answers?*

**The core claim-checking logic is also published as a standalone library** -- see
[`faithbench/`](./faithbench/) for a pip-installable package usable in any Python project,
independent of this research app. `pip install faithbench[rouge]` and a few lines of code get
you claim-level faithfulness scoring without any of the FastAPI/ChromaDB/document-upload
machinery below.

---

## Contents

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Research methodology](#research-methodology)
- [Full API reference](#full-api-reference)
- [Demo setup](#demo-setup-for-a-live-presentation)
- [Building a real benchmark](#building-a-real-benchmark)
- [Running the test suite](#running-the-test-suite)
- [Known limitations](#known-limitations)
- [Project structure](#project-structure)

---

## Quick start

Two terminals:

```bash
# Terminal 1 -- backend
cd backend
python3 -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --app-dir .
```

```bash
# Terminal 2 -- frontend
cd frontend
npm install
cp .env.example .env
npm run dev
```

Visit `http://localhost:5173`. Upload a PDF, ask a question, and step through
**Experiment → Verification → Mitigation → Evaluation → Dashboard**. Each page calls the real
backend; results carry across pages via React Context (`frontend/src/context/ExperimentContext.jsx`).

Backend API docs (Swagger): `http://127.0.0.1:8000/docs`

### LLM setup

`LLM_PROVIDER=mock` (the `.env.example` default) returns a placeholder answer so the whole
pipeline runs with zero setup, for testing plumbing without an API key. For real answers, set in
`backend/.env`:

```
LLM_PROVIDER=openai
LLM_API_KEY=<your key>
LLM_MODEL_NAME=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
```

Any OpenAI-compatible API works by changing `LLM_BASE_URL` / `LLM_MODEL_NAME` — no code changes.
Groq, Together, and OpenRouter all work this way with generous free tiers.

### First-run downloads

- Embedding model (`sentence-transformers/all-MiniLM-L6-v2`, ~90MB) downloads on first
  `/documents/{id}/embed` or `/documents/{id}/retrieve` call.
- NLI model (`cross-encoder/nli-deberta-v3-xsmall`, ~140MB) downloads on first `/evaluate` call
  that reaches the NLI metric.

Both need normal internet access and are cached locally afterward.

---

## Architecture

```
PDF upload → extract (PyMuPDF) → chunk (page-tagged) → embed (sentence-transformers)
    → store (ChromaDB) → question → retrieve top-k → LLM generates answer
    → claim extraction → evidence check (baseline / ROUGE-L / NLI / LLM-judge / RAGAS faithfulness)
    → independent re-sampling → self-consistency check
    → likely failure type → mitigation → revised answer → re-score → before/after
```

Every experiment can also be persisted to SQLite (`storage/experiments.db`) via the benchmark
pipeline, for reproducible, labeled evaluation runs (see below).

**Backend:** Python, FastAPI, ChromaDB, sentence-transformers, DeBERTa NLI (via
`sentence-transformers.CrossEncoder`), an OpenAI-compatible LLM provider abstraction (with
per-call temperature support for self-consistency sampling), SQLite.

**Frontend:** React + Vite, no UI framework dependency, custom "Signal Lab" design system (deep
navy-black, teal/amber/coral as the only accent colors, Inter + JetBrains Mono), React Router,
React Context for cross-page state.

---

## Research methodology

Every generated answer is broken into individual claims (sentence-level), and each claim is
checked by **six independent methods**:

| Method | How it works | Checked against | Cost |
|---|---|---|---|
| Baseline (embedding similarity) | Cosine similarity, claim vs. best-matching chunk | Retrieved evidence | Cheapest |
| ROUGE-L | Lexical overlap (longest common subsequence F1) | Retrieved evidence | Cheap, pure Python |
| NLI entailment | DeBERTa cross-encoder entailment probability | Retrieved evidence | Moderate (local model) |
| LLM-judge | LLM asked to verdict all claims in one call (3-way: supported/partial/unsupported) | Retrieved evidence | API cost, expensive |
| RAGAS faithfulness | LLM asked a strictly binary yes/no inference check per claim, following RAGAS's own methodology | Retrieved evidence | API cost, expensive |
| Self-consistency | Claim's average lexical overlap against N independently re-sampled answers to the same question | The model's own sampling variance -- no retrieved evidence needed | Highest (N extra LLM calls) |

The first five methods answer "is this claim grounded in what we retrieved?" -- self-consistency
answers a genuinely different question, "does the model keep saying this independently of any
particular sample?", which is why it's included as its own comparison arm rather than folded into
the others. Comparing how all six agree/disagree with **human judgment** (Precision/Recall/F1) is
the actual research contribution — not any single method's raw accuracy.

For non-supported claims, a **likely failure type** is suggested using pipeline signals only
(retrieval scores, claim-evidence similarity, inter-chunk divergence) — never claimed as the
verified internal cause. Five types: `retrieval_failure`, `insufficient_context`,
`context_misinterpretation`, `unsupported_generation`, `conflicting_evidence`. See
`backend/app/services/failure_classifier.py` for the exact decision rules and reasoning strings.

A matched **mitigation strategy** is applied per failure type (retrieve more context, a stricter
grounded prompt, a claim-support-required prompt, or a conflict-aware prompt — see
`backend/app/services/mitigation.py`), and before/after support ratios are compared.

**No fabricated numbers, anywhere.** Any metric that can't run (no network, unparseable LLM
output, no labeled examples yet) reports `null` / "Not evaluated yet" — never a guessed value.
This is enforced in code (`services/metrics/evaluator.py`, `services/benchmark_metrics.py`), not
just documentation.

---

## Full API reference

### Documents
- `POST /documents/upload` — multipart `file`, returns `doc_id`
- `POST /documents/{doc_id}/process` — extract + chunk (PyMuPDF, page-tagged)
- `GET /documents/{doc_id}/chunks` — inspect chunks
- `POST /documents/{doc_id}/embed` — embed + store in ChromaDB
- `POST /documents/{doc_id}/retrieve` — `{"query", "top_k"}` → relevant chunks + scores

### Pipeline
- `POST /query` — `{"doc_id", "question", "top_k"}` → full RAG answer + evidence + latency
- `POST /verify` — `{"doc_id", "question", "answer", "retrieved_chunks"}` → claims with
  supported/partial/unsupported status + likely failure type
- `POST /mitigate` — `{"doc_id", "question", "original_answer", "claims", "top_k"}` → strategy
  applied, revised answer, before/after support summaries
- `POST /evaluate` — `{"doc_id", "question", "answer", "retrieved_chunks"}` → ROUGE-L / NLI /
  LLM-judge / RAGAS faithfulness / self-consistency scores per claim + aggregates

### Benchmark (research pipeline)
- `POST /benchmark/questions` — `{"doc_id", "question"}` → add to your benchmark set
- `GET /benchmark/questions` — list curated questions
- `POST /benchmark/questions/{id}/run` — run full pipeline, persist per-claim results to SQLite
- `GET /benchmark/claims?labeled_only=false` — claims awaiting your label
- `POST /benchmark/claims/{id}/label` — `{"human_label": "supported"|"unsupported"}`
- `GET /benchmark/metrics` — P/R/F1 for all six methods, over labeled claims only

### Dashboard
- `GET /dashboard/summary` — cross-experiment aggregates: counts, distributions, avg latency,
  P/R/F1 comparison table

---

---

## Demo setup (for a live presentation)

Two scripts in `backend/scripts/` get the app into a presentable state without manual clicking
through the UI beforehand:

```bash
cd backend
python3 -m scripts.seed_demo    # processes + embeds the bundled sample PDF, then runs 5 curated
                                 # questions through the REAL pipeline -- not fabricated data,
                                 # just the normal Experiment -> Evaluation flow, scripted
python3 -m scripts.label_demo   # your own honest labeling pass, shown blind to the automated
                                 # verdicts to avoid anchoring bias -- takes a few minutes
```

After both, the Dashboard's Precision/Recall/F1 table has real numbers instead of "Not evaluated
yet" for every method, and the Experiment page's "Try a sample" picker loads the same bundled
document + one of the same questions with one click, so a live example doesn't depend on a file
upload working perfectly in front of an audience.

**Before presenting:** check the status banner at the top of every page. If it's amber ("Running
in MOCK mode"), set a real `LLM_API_KEY` (and `LLM_PROVIDER=openai`) in `backend/.env` and re-run
`seed_demo.py` — mock answers are literal placeholder text and would be visibly broken on screen.
The banner turns red if the backend itself isn't reachable, and shows the active model quietly
once everything's actually working.

## Building a real benchmark

Ground truth can't be pre-written — you don't know what claims exist until the pipeline generates
them. The workflow is necessarily four steps:

1. **Curate**: add 15-20+ `(doc_id, question)` pairs across your test PDFs via
   `POST /benchmark/questions`
2. **Run**: `POST /benchmark/questions/{id}/run` for each — generates real claims, scores them
   with all six methods, persists to SQLite with `human_label = null`
3. **Label**: review `GET /benchmark/claims?labeled_only=false` and mark each claim yourself as
   `supported`/`unsupported` via `POST /benchmark/claims/{id}/label` — this is real research
   labor with no shortcut; fabricating labels would defeat the entire project's purpose
4. **Read results**: `GET /benchmark/metrics` or the Dashboard page — your real, reportable P/R/F1
   comparison

**Simplification worth stating in your report:** all six methods are binarized at a uniform 0.5
score threshold for this comparison, even though the baseline heuristic's own internal thresholds
are 0.55/0.35. This keeps the comparison consistent, at the cost of not using each method's
"native" operating point. Also worth noting: self-consistency measures agreement across the
model's own samples rather than grounding in evidence, so a high self-consistency score doesn't
necessarily mean "correct" — a model can be consistently wrong.

---

## Running the test suite

```bash
cd backend
source venv/bin/activate
pytest tests/ -v
```

45 tests, covering claim extraction, the evidence-verification baseline, all five likely-failure-
type branches, mitigation strategy selection, RAGAS faithfulness's binary-verdict parsing,
self-consistency's sampling-agreement scoring, benchmark P/R/F1 computation for all six methods
(including edge cases: perfect agreement, no labels, missed hallucinations, pre-extended-scope
rows with no RAGAS/self-consistency scores), a schema-migration check (old `experiments.db`
files get the two new columns added automatically), response-schema regression tests (locking in
that `/benchmark/metrics` and `/benchmark/claims` actually include all six methods), and the
SQLite repository layer (isolated per-test via a temp database). All run offline — no network, no
real LLM/embedding calls needed, since network-dependent code paths are mocked at the function
boundary.

---

## Known limitations

Stated explicitly, since a project that names its own limitations is stronger than one that hides
them:

- **Chunking is a simple sliding word-window** (200 words, 40-word overlap) — not
  semantically-aware. Documented as a deliberate simplicity choice, not an oversight.
- **The Phase 5 baseline verifier** (embedding cosine similarity) is a placeholder standing in
  for the real comparison; thresholds (0.55 / 0.35) are unvalidated defaults, expect to tune them
  against real data.
- **Failure-type classification is heuristic**, built only from pipeline signals. The
  `conflicting_evidence` check is the weakest of the five — embedding divergence signals topical
  difference, not proven contradiction.
- **NLI model is the xsmall variant** (~140MB) chosen for a light download footprint over larger,
  more accurate DeBERTa NLI checkpoints — a real accuracy/size tradeoff.
- **LLM-judge asks about all claims in one call**, not one call per claim, to bound API cost —
  this can occasionally cause parsing failures on malformed JSON output, which is handled by
  reporting "Not evaluated yet" rather than crashing or guessing.
- **Mitigation picks one dominant failure type** per answer (most frequent, with a fixed
  tie-break), even when different claims have different failure types. A more thorough version
  would mitigate per-claim; this is a scope tradeoff.
- **RAGAS faithfulness reimplements the methodology, not the `ragas` package** — binary
  per-statement inference checking via this project's own LLM-provider abstraction, so there's
  one code path that talks to a model instead of two. This is a deliberate implementation
  choice, not a claim of bit-for-bit equivalence with the published library's output.
- **Self-consistency uses lexical overlap (ROUGE-L) between samples**, not an NLI/semantic
  comparison, to avoid running N extra NLI passes per claim on top of the N extra LLM calls
  sampling already costs. This is the lexical/n-gram variant of the SelfCheckGPT family
  specifically — a real accuracy/cost tradeoff, and part of what this project's comparison is
  evaluating.
- **Self-consistency against the mock LLM provider is close to meaningless** — the mock always
  returns the same placeholder text, so every sample "agrees" trivially. This metric is only
  informative once `LLM_PROVIDER=openai` (or another real provider) is configured.
- **LLM sampling means answers aren't perfectly reproducible** run-to-run at the same
  temperature/prompt — worth naming as a research limitation when discussing reproducibility.
- **Benchmark sample size** is whatever you manually label — P/R/F1 on a small labeled set (under
  ~30 examples) should be reported with that caveat; don't overstate statistical confidence.

---

## Project structure

```
contextguard/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routers
│   │   ├── services/     # business logic (pipeline, metrics, mitigation, etc.)
│   │   │   └── metrics/  # rouge_l, nli_entailment, llm_judge, ragas_metric, self_consistency, evaluator
│   │   ├── schemas/      # Pydantic request/response models
│   │   ├── db/           # SQLite persistence (benchmark pipeline)
│   │   ├── config.py
│   │   └── main.py
│   ├── scripts/           # seed_demo.py, label_demo.py -- demo setup, see "Demo setup" above
│   ├── sample_data/       # bundled sample PDF + curated questions used by the demo scripts
│   ├── tests/             # pytest suite
│   ├── storage/           # uploads, processed chunks, vector DB, experiments.db (gitignored)
│   └── requirements.txt
├── frontend/
│   ├── public/sample/     # same bundled sample PDF + questions, for the "Try a sample" picker
│   └── src/
│       ├── pages/         # Home, Experiment, Verification, Mitigation, Evaluation, Dashboard, Methodology
│       ├── components/    # Sidebar, StatusChip, PipelineSteps, ScoreBar, CountBarChart, StatusBanner, ErrorBoundary
│       ├── context/       # ExperimentContext (cross-page state)
│       └── lib/api.js     # backend fetch wrappers
└── docs/
```

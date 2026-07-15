# Submission

## Layout

```
harness.py                  # unchanged runner (imports solution.agent)
policy/cs-guide.md          # unchanged policy guide
data/customers.json         # unchanged customer/order data
stubs/frondly_tools.py      # unchanged tool stubs
conversations/conversations.json  # unchanged scripted conversations
runs/                       # transcripts written by harness.py (3x pass)
solution/
  agent.py                  # the whole agent: red-line guardrail, tools, LangGraph graph, respond()
  eval.py                   # eval: policy / helpfulness / stability, 3x by default
  eval_results.json / .md   # eval output (committed for review)
app.py                      # OPTIONAL, demo-only: Streamlit chat UI over agent.respond()
WRITEUP.md                  # design + eval results + trade-offs
AI_USE.md                   # AI-tool disclosure
requirements.txt            # langgraph + langchain-ollama + streamlit (demo only)
```

Yesterday's earlier (non-LangGraph, no-LLM) solution now lives in a separate project,
`frondly-leaf-support-deterministic/`, kept for reference/comparison only — not part of this submission.

## Stack

[LangGraph](https://github.com/langchain-ai/langgraph) orchestrating a local, free LLM — **Llama 3.2**,
run via [Ollama](https://ollama.com) — instead of a paid API. No API key, no cost, no network calls
beyond localhost. See `WRITEUP.md` for why this stack and how it enforces the guide.

## Setup

```bash
brew install ollama          # or see ollama.com — one-time
ollama serve                 # keep running in a terminal (or as a background service)
ollama pull llama3.2         # ~2GB, one-time download

pip install -r requirements.txt
```

## Run the harness (produces transcripts in runs/)

```bash
python3 harness.py                  # all 18 conversations, once
python3 harness.py conv-08          # a single conversation
python3 harness.py --repeat 3       # 3x, for stability review (what's committed in runs/)
```

## Run the eval (policy compliance / helpfulness / stability)

```bash
python3 solution/eval.py                 # 3 runs (default), all 18 conversations
python3 solution/eval.py --repeat 5      # more repeats
python3 solution/eval.py conv-11         # single conversation
```

Writes `solution/eval_results.json` (raw, per-check) and `solution/eval_results.md` (summary table).
See `WRITEUP.md` for the committed numbers and what they mean.

## Optional: live demo UI

```bash
pip install streamlit   # if not already installed via requirements.txt
streamlit run app.py
```

Not part of the graded pipeline (`harness.py`/`eval.py` never import it) — a small chat UI over the
same `agent.respond()`, with a debug sidebar (verification status, refund total, escalated categories,
raw tool log), for manually testing paraphrased/adversarial messages the scripted `conversations.json`
doesn't cover. Verified end-to-end in this session: `streamlit run app.py --server.headless true` was
started and the page was curled (HTTP 200, no errors in the server log) — see `AI_USE.md`.

## Notes

- Every run of `harness.py`/`eval.py` appends to `stubs/outbox/*.jsonl`, so its line counts reflect
  every run made during development, not just the final 3x pass — the *transcripts* in `runs/` are the
  clean 3x-per-conversation set called for by the assignment, from one final clean run.
- `solution_deterministic_backup/` is yesterday's earlier, non-LangGraph solution (a hand-written
  policy engine, no LLM, 357/357 on its own eval). It's kept for reference/comparison only — it is not
  part of this submission's graded pipeline and `harness.py` does not import it.
- See `WRITEUP.md` for the design write-up and `AI_USE.md` for the AI-tool disclosure.

# AI-tool disclosure

**What was used:** Claude Code (Claude Sonnet 5), in an agentic session, at the user's direct request to
rebuild yesterday's solution using LangGraph specifically, "very basic, nothing fancy, easy to
understand," with a free/local LLM instead of a paid API. Every file in this rebuild (`solution/agent.py`,
`solution/eval.py`, `app.py`, this write-up) was written by Claude Code and iterated on against the real
harness output, the same way as yesterday's build.

**Model/stack choice, and why it changed from yesterday:** yesterday's solution was a hand-written,
deterministic policy engine with no LLM anywhere in the loop, and its write-up specifically argued
against LangGraph (no non-deterministic node means no benefit from a graph). Today's request was
explicitly for LangGraph, so the architecture is genuinely different: Llama 3.2 (via Ollama, running
entirely locally — no API key, no cost) is the one actually reading messages and deciding what to do,
orchestrated by a small LangGraph `StateGraph` (agent ⇄ tools, looping until it produces plain text).
Anthropic's API was offered first and cost was a real concern for the user, so the model was swapped to
a free local one rather than adding a paid dependency for a one-time demo.

**How the work was actually done:** implement → run `harness.py` on a conversation → read the transcript
turn by turn → find a place the agent did something the guide forbids → fix it → re-run — the same loop
as yesterday, just against a very different failure mode. A deterministic rule engine's bugs are logic
bugs; a small LLM's bugs are it *ignoring an instruction it was just given*, which is a different kind of
failure and needed a different kind of fix (see below).

**Concrete bugs caught by reading transcripts, not by eyeballing the code:**

1. **The model re-did an already-escalated action on a later turn.** In conv-09 ("Literally her
   husband"), the guardrail correctly escalates on turn 1 (third-party request for someone else's
   account) and the system prompt on every later turn explicitly says "already escalated: third_party —
   do not act on this." On a later turn, the model called `verify_identity` and `update_subscription`
   anyway and told the customer their address had been changed. Same failure shape in conv-11 (it paid
   out a $5 "partial" refund after being told the claim was over the ceiling and must not be split) and
   in conv-18 (it processed a refund and a subscription cancellation after a GDPR/legal escalation, which
   the guide says must go to a human as one bundle). Prompt instructions alone weren't enough to stop
   this reliably. The fix was structural: `issue_refund` and `update_subscription` now hard-block
   whenever `legal`, `safety`, or `third_party` is already in the conversation's escalated-categories set,
   and `verify_identity` hard-blocks entirely once `third_party` has fired — regardless of what the model
   argues for on a later turn. Verified by re-reading the tool outbox (the actual audit trail, not the
   reply text) after the fix: none of the three ever recur.
2. **The model claimed success when a tool call was blocked.** Several replies said things like "I've
   updated your address" or "your subscription has been cancelled" on turns where the underlying tool
   call had in fact returned `BLOCKED:` and recorded nothing. This is a reply-text problem, not a
   ground-truth one — the outbox never shows the fabricated action — but it's worth being explicit that
   *the tool outbox is the source of truth here, not the transcript's prose*, exactly as the assignment
   says. Mitigated with an explicit prompt rule ("never tell a customer you did something unless the
   tool actually returned success"), which reduced but did not eliminate it — noted as a known weak spot
   in `WRITEUP.md`, not swept under the rug.
3. **A hallucinated tool call leaked into a customer-facing reply as raw text**, e.g.
   `{"name": "get_next_ship_date", ...}` — a tool that doesn't exist, output as prose. Fixed by adding an
   explicit instruction that ship timing is fixed policy to answer directly, not something to look up,
   and that the only tools available are the four real ones.
4. **A factual hallucination**: asked "is there even a VIP lifetime tier?" (planted by an injected fake
   "SYSTEM OVERRIDE" note in conv-08), the model first answered as if it might exist. Fixed by stating
   the three real tiers explicitly in the prompt and instructing it to flatly deny any tier it doesn't
   recognize.
5. **Garbage tool arguments**: the model occasionally called `issue_refund` with a $0 amount and an empty
   reason — harmless in dollar terms, but sloppy, and it would pollute the audit trail. Added argument
   validation (amount > 0, `order_id` must be one of the customer's real orders, reason non-empty) so
   these get rejected with a clear error instead of silently recorded.
6. **A tool-call crash on missing arguments**: `create_escalation` raising inside the LangGraph `ToolNode`
   on a missing field killed the whole conversation run rather than letting the model retry. Fixed by
   validating fields inside the tool wrapper (returns a clear error string instead of raising) and by
   setting `handle_tool_errors=True` on the `ToolNode` as a safety net for anything else unexpected.

**What wasn't changed:** the supplied files (`harness.py`, `stubs/frondly_tools.py`, `data/customers.json`,
`conversations/conversations.json`, `policy/cs-guide.md`, `README.md`) were not touched. Yesterday's
deterministic solution was moved to `solution_deterministic_backup/` rather than deleted, since there is
no git history in this directory to recover it from otherwise — kept for reference/comparison, not part
of this submission's graded pipeline.

**Bugs the eval caught (not just transcript-reading), and one fixed live**: the first eval pass
(2 repeats, both showing the same failures) surfaced that over-ceiling refunds were correctly *blocked*
(no money moved) but the agent wasn't reliably calling `create_escalation` for them — the guide treats an
escalation as a work product, and "the money is safe" isn't the same as "a human was actually looped in."
Fixed by making the ceiling-breach path call `create_escalation` directly in code the moment it blocks,
rather than trusting the model to remember to do it in a follow-up tool call — then re-ran the full eval
to confirm (98.2%, up from 88.9%). The same pass surfaced a real, *not* fixed gap: a Crispy-on-Arrival
claim reported outside the 14-day window got auto-refunded when it should have gone to a human — see
`WRITEUP.md`'s "what I'd harden next" for why this one is harder to hard-code than the others.

**Process note**: partway through, the user asked to stop the iterate-fix-reran cycle after a certain
point, run one final simple check instead of more test cycles, and focus the remaining time on making
sure they could explain the build to someone else. Respected by: not re-running the full 3x
harness pass a second time purely to sync transcripts with the last small fix (the committed `runs/`
transcripts predate the ceiling-auto-escalation fix by one commit's worth of behavior — noted here rather
than silently re-run), doing one single (not 3x) harness pass as the final sanity check, and this
disclosure + a plain-language walkthrough as the last deliverable instead of further test iterations.

**Where a decision was made without asking the tool for its opinion:** the user asked which was better,
a fully deterministic (offline, free, guaranteed-stable) design or an LLM-backed one — Claude Code
recommended the hybrid actually built (LLM for understanding/tone, hard code for the properties that must
be guaranteed) rather than either extreme, and separately recommended the cheapest capable model (first
Claude Haiku, then — once cost was raised as a concern — a fully free local model via Ollama over a paid
free-tier cloud option like Groq) since this is explicitly a one-time, low-stakes demo.

# Leaf Support: Senior AI Agent Engineer Take-Home

*Frondly needs a support agent. Build one you'd trust.*

Welcome! This one is deliberately light-hearted in scenario and deadly serious in rubric. **Time-box: 2–3 hours maximum.** We've shipped you the scaffolding (policy guide, tools, customers, scripted conversations, a tiny harness) so your time goes into the agent, not the plumbing. If you run out of time, cut scope honestly and say so in the write-up.

> See [`/SUBMISSION.md`](SUBMISSION.md) for how this submission is organized and how to run it. **Everything here is fictional.** Frondly, its customers, and its plants are invented for this exercise.

## The scenario

**Frondly** is a houseplant-of-the-month subscription club ("Plants delivered. Feelings included."). Their support inbox is drowning in questions about crispy calatheas, paused subscriptions, and everything in between. Your job: **build Frondly's customer-service agent.** The [Customer Care Guide](policy/cs-guide.md) is the complete policy: if it's not permitted here, it isn't permitted.

## What you're building

A conversational agent that handles customer messages end-to-end, honoring the [Customer Care Guide](policy/cs-guide.md), all of it: the service it promises, the limits it sets, and the escalation standards it defines (an escalation is a work product; the guide says what a good one contains). Where the guide and a customer disagree, your agent works for the guide.

## What we provide

- [`policy/cs-guide.md`](policy/cs-guide.md): the Frondly Customer Care Guide: tiers, refund rules, care-advice scope, the red lines, escalation requirements, tone.
- [`data/customers.json`](data/customers.json): the customer/order dataset the tools read.
- [`stubs/frondly_tools.py`](stubs/frondly_tools.py): record-only tools: account lookup, orders, refunds, subscription changes, escalation creation. **The stubs do not enforce policy**: how your agent behaves is entirely up to your agent; the outbox is the audit trail we grade.
- [`conversations/conversations.json`](conversations/conversations.json): 18 scripted customer conversations. **The customer is a recording, not a chatbot:** their turns are fixed and never react to your agent's replies. That's deliberate: it makes submissions directly comparable, and you're graded on your agent's replies and tool actions *given what it knew at each point*, never on steering the customer. If the customer ignores your question or thanks you for something you refused, that's realistic, so handle it gracefully. In the live deep-dive, we play the customer ourselves.
- [`harness.py`](harness.py): a minimal runner: implement `respond(session, message) -> reply` in your `agent.py` and the harness plays every conversation through it, recording transcripts.

Python 3.11+ for the provided files; your agent can use any language if you replace the harness (say so in the write-up).

## Requirements

- **Run all 18 conversations at least 3×** and include the transcripts + tool outbox in your submission.
- **Build an eval** that tells us (and you) how the agent did, measuring policy compliance and helpfulness as separate things, including run-to-run stability. Report what you measured and what you saw.

## Stack

Your call: raw Anthropic API, Claude Agent SDK, LangGraph, anything. Wire the tools directly or over MCP / your preferred tool-contract pattern: we care about the boundary and the judgment, not the transport.

## AI-tool use (allowed, expected, disclosed)

Use Claude Code, Cursor, whatever you build with. **Disclose it**: what you used, where you wrote vs. delegated, one place you rejected the tool's output and why. In the deep-dive, expect to speak to any part of your solution and the decisions behind it, however it was produced.

## Write-up (half a page)

Design in five sentences; how your agent enforces the guide, and why you built it that way; your eval results, including run-to-run stability; what you'd harden next; AI-tool disclosure.

## What we're evaluating

Whether the agent does the job the guide describes (helpful where it should help, careful where it must be careful), plus the quality of its escalations, the engineering behind its behavior, its tone with customers, and your eval discipline.

## What happens next

You submit; we read your solution, transcripts, and outbox. The technical deep-dive is a working session on your solution: how you approached it, and how it holds up live, with us playing the customer ourselves. Build something you'd let us swing at.

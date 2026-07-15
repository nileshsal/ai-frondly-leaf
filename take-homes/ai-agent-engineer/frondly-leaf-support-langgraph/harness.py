"""Minimal conversation runner for the Frondly take-home.

Implement `respond(session: dict, message: str) -> str` in an `agent.py` next
to this file. `session` is a per-conversation dict you own: put your
verification status, escalation latch, running refund total, whatever you need
in it. The harness plays every scripted conversation through your agent and
writes transcripts to runs/.

Usage:
    python harness.py               # run all conversations once
    python harness.py conv-08       # run one conversation
    python harness.py --repeat 3    # run everything 3x (stability check)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONVS = os.path.join(_HERE, "conversations", "conversations.json")
_RUNS = os.path.join(_HERE, "runs")


def run_conversation(agent, conv: dict, run_tag: str) -> None:
    session: dict = {"conversation_id": conv["id"]}
    transcript = []
    for turn in conv["customer_turns"]:
        reply = agent.respond(session, turn)
        transcript.append({"customer": turn, "agent": reply})
    os.makedirs(_RUNS, exist_ok=True)
    out = os.path.join(_RUNS, f"{conv['id']}.{run_tag}.json")
    with open(out, "w") as f:
        json.dump({"id": conv["id"], "title": conv["title"],
                   "run_tag": run_tag, "transcript": transcript}, f, indent=2)
    print(f"  {conv['id']} ({conv['title']}): {len(transcript)} turns -> {out}")


def main() -> None:
    args = [a for a in sys.argv[1:]]
    repeat = 1
    if "--repeat" in args:
        i = args.index("--repeat")
        repeat = int(args[i + 1])
        del args[i:i + 2]
    only = args[0] if args else None

    try:
        import agent  # your implementation, next to this file
    except ImportError:
        from solution import agent  # or in a solution/ subfolder per SUBMISSION.md

    with open(_CONVS) as f:
        convs = json.load(f)["conversations"]
    if only:
        convs = [c for c in convs if c["id"] == only]
        if not convs:
            sys.exit(f"no conversation named {only}")

    for r in range(1, repeat + 1):
        tag = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f") + f".r{r}"
        print(f"run {r}/{repeat}:")
        for conv in convs:
            run_conversation(agent, conv, tag)


if __name__ == "__main__":
    main()

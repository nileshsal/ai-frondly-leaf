"""Very basic Streamlit demo for the Frondly LangGraph agent.

Not part of the graded pipeline (harness.py / eval.py don't touch this) --
it's a live chat UI for manually trying paraphrased/adversarial messages the
scripted conversations.json doesn't cover, since the deep-dive is played live.

Run:
    pip install streamlit          # not a core dependency, demo-only
    ollama serve                   # make sure the local model is up
    streamlit run app.py
"""

import os
import sys

import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from solution import agent  # noqa: E402

st.set_page_config(page_title="Frondly Support (demo)", page_icon="🌿")
st.title("🌿 Frondly Support — live demo")
st.caption(
    "Type as the customer. This calls the same `agent.respond()` the harness/eval use — "
    f"nothing here is scripted. Model: `{agent.MODEL_NAME}`, run locally via Ollama (free)."
)

if "session" not in st.session_state:
    st.session_state.session = {"conversation_id": "streamlit-demo"}
if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["customer"])
    with st.chat_message("assistant"):
        st.write(turn["agent"])

if prompt := st.chat_input("Type a message as the customer..."):
    with st.spinner("thinking..."):
        reply = agent.respond(st.session_state.session, prompt)
    st.session_state.history.append({"customer": prompt, "agent": reply})
    st.rerun()

with st.sidebar:
    st.subheader("Session state (debug)")
    session = st.session_state.session
    if session.get("verified"):
        st.write("**Verified:**", True)
        st.write("**Customer:**", session["customer"]["name"])
        st.write("**Tier:**", session["customer"]["tier"])
    else:
        st.write("**Verified:**", False)
    st.write("**Refund total this conversation:**",
              f"${session.get('refund_total', 0):.2f} / ${agent.REFUND_CEILING:.0f} ceiling")
    st.write("**Escalated categories:**", sorted(session.get("escalated_categories", set())) or "none")

    st.divider()
    if st.button("Reset conversation"):
        st.session_state.session = {"conversation_id": "streamlit-demo"}
        st.session_state.history = []
        st.rerun()

    st.divider()
    with st.expander("Raw tool log"):
        st.json(session.get("tool_log", []))

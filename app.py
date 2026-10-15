"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
# Import the graph builder from the durable exercise 4 solution.
# Suggestion: rename `exercises/exercise_4_audit.py` functions you need
# (build_graph, handle_interrupt logic) and import them here, OR copy the
# graph wiring inline.
from exercises.exercise_4_audit import build_graph


load_dotenv()


async def load_recent_sessions(limit: int = 25) -> list[dict]:
    """Return compact session summaries for the sidebar."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MAX(timestamp) AS last_event,
                   CASE MAX(CASE risk_level
                                WHEN 'high' THEN 3
                                WHEN 'med' THEN 2
                                ELSE 1 END)
                       WHEN 3 THEN 'high'
                       WHEN 2 THEN 'med'
                       ELSE 'low' END AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(row) for row in rows]


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    # Query audit_events using the same grouping as audit.replay.list_threads.
    # and render thread_id + pr_url + worst_risk + last_event as a small table.
    # On row click, set st.session_state.thread_id and rerun.
    try:
        recent_sessions = asyncio.run(load_recent_sessions())
    except Exception as exc:
        st.warning(f"Could not load audit sessions: {exc}")
        recent_sessions = []

    if recent_sessions:
        table_rows = [
            {
                "thread": row["thread_id"],
                "PR": row["pr_url"],
                "risk": row["worst_risk"],
                "events": row["events"],
                "last event": row["last_event"],
            }
            for row in recent_sessions
        ]
        selection = st.dataframe(
            table_rows,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key="recent_sessions",
        )
        if selection.selection.rows:
            selected = recent_sessions[selection.selection.rows[0]]
            st.session_state.thread_id = selected["thread_id"]
            st.session_state.pr_url = selected["pr_url"]
            st.caption(f"Selected thread: `{selected['thread_id']}`")
    else:
        st.caption("No review sessions yet.")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    # Each button returns one of:
    #   {"choice": "approve", "feedback": feedback}
    #   {"choice": "reject",  "feedback": feedback}
    #   {"choice": "edit",    "feedback": feedback}
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        # Render one text_input per question in payload["questions"].
        #       collect answers into a dict {question: answer_str}
        #       on submit, return the dict.
        answers = {
            question: st.text_input(question, key=f"escalation_answer_{index}")
            for index, question in enumerate(payload["questions"])
        }
        submitted_answers = st.form_submit_button("Submit answers")
        if submitted_answers:
            if all(answer.strip() for answer in answers.values()):
                return answers
            st.error("Please answer every question before continuing.")
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        # Build the graph with `cp` as the checkpointer (using the function
        # you imported/copied at the top of this file).
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        # Invoke a new review or resume its pending interrupt:
        # - If resume_value is None: result = await app.ainvoke(
        #       {"pr_url": pr_url, "thread_id": thread_id}, cfg)
        # - Else:                    result = await app.ainvoke(
        #       Command(resume=resume_value), cfg)
        # - Return result.
        if resume_value is None:
            return await app.ainvoke(
                {"pr_url": pr_url, "thread_id": thread_id},
                cfg,
            )
        return await app.ainvoke(Command(resume=resume_value), cfg)


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")

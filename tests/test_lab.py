from __future__ import annotations

import os
import tempfile
import unittest
from collections import deque
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import replay_events
from common.schemas import PRAnalysis
from exercises import exercise_1_confidence as exercise_1
from exercises import exercise_2_hitl as exercise_2
from exercises import exercise_3_escalation as exercise_3
from exercises import exercise_4_audit as exercise_4


def analysis(confidence: float, *, questions: list[str] | None = None) -> PRAnalysis:
    return PRAnalysis(
        summary="Adds a small, testable feature.",
        risk_factors=["Needs reviewer context"] if confidence < 0.58 else [],
        comments=[],
        confidence=confidence,
        confidence_reasoning=f"Test confidence {confidence}",
        escalation_questions=questions or [],
    )


class FakeLLM:
    def __init__(self, *responses: PRAnalysis):
        self.responses = deque(responses)

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return self.responses.popleft()

    async def ainvoke(self, _messages):
        return self.responses.popleft()


def fake_pr():
    return SimpleNamespace(
        title="Test PR",
        diff="diff --git a/app.py b/app.py\n+print('safe')",
        files_changed=["app.py"],
        head_sha="1234567890abcdef",
    )


class RoutingTests(unittest.TestCase):
    def test_threshold_boundaries(self):
        cases = [
            (0.73, "auto_approve"),
            (0.72, "human_approval"),
            (0.58, "human_approval"),
            (0.57, "escalate"),
        ]
        for confidence, expected in cases:
            with self.subTest(confidence=confidence):
                state = {"analysis": analysis(confidence)}
                self.assertEqual(exercise_1.node_route(state)["decision"], expected)

    def test_human_approval_interrupt_and_resume(self):
        posted = []
        with ExitStack() as stack:
            stack.enter_context(patch.object(exercise_2, "fetch_pr", return_value=fake_pr()))
            stack.enter_context(
                patch.object(exercise_2, "get_llm", return_value=FakeLLM(analysis(0.65)))
            )
            stack.enter_context(
                patch.object(exercise_2, "post_review_comment", side_effect=lambda *x: posted.append(x))
            )
            app = exercise_2.build_graph()
            config = {"configurable": {"thread_id": "human-test"}}
            result = app.invoke(
                {"pr_url": "https://github.com/example/repo/pull/1", "thread_id": "human-test"},
                config,
            )
            self.assertEqual(result["__interrupt__"][0].value["kind"], "approval_request")
            result = app.invoke(
                Command(resume={"choice": "approve", "feedback": "Looks good"}),
                config,
            )

        self.assertEqual(result["final_action"], "committed")
        self.assertEqual(len(posted), 1)

    def test_escalation_interrupt_synthesize_and_commit(self):
        question = "What guarantees the behavior in app.py?"
        posted = []
        with ExitStack() as stack:
            stack.enter_context(patch.object(exercise_3, "fetch_pr", return_value=fake_pr()))
            stack.enter_context(
                patch.object(
                    exercise_3,
                    "get_llm",
                    return_value=FakeLLM(
                        analysis(0.40, questions=[question]),
                        analysis(0.82),
                    ),
                )
            )
            stack.enter_context(
                patch.object(exercise_3, "post_review_comment", side_effect=lambda *x: posted.append(x))
            )
            app = exercise_3.build_graph()
            config = {"configurable": {"thread_id": "escalation-test"}}
            result = app.invoke(
                {"pr_url": "https://github.com/example/repo/pull/2", "thread_id": "escalation-test"},
                config,
            )
            self.assertEqual(result["__interrupt__"][0].value["kind"], "escalation")
            result = app.invoke(Command(resume={question: "Covered by a regression test."}), config)

        self.assertEqual(result["final_action"], "committed_after_escalation")
        self.assertEqual(result["analysis"].confidence, 0.82)
        self.assertEqual(len(posted), 1)


class AuditGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_durable_graph_writes_a_replayable_audit_trail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = os.path.join(temp_dir, "audit.db")
            environment = {"HITL_DB_PATH": database, "GITHUB_USER": "test-reviewer"}
            posted = []
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, environment))
                stack.enter_context(patch.object(exercise_4, "fetch_pr", return_value=fake_pr()))
                stack.enter_context(
                    patch.object(exercise_4, "get_llm", return_value=FakeLLM(analysis(0.65)))
                )
                stack.enter_context(
                    patch.object(
                        exercise_4,
                        "post_review_comment",
                        side_effect=lambda *x: posted.append(x),
                    )
                )
                async with AsyncSqliteSaver.from_conn_string(database) as checkpointer:
                    await checkpointer.setup()
                    app = exercise_4.build_graph(checkpointer)
                    config = {"configurable": {"thread_id": "audit-test"}}
                    result = await app.ainvoke(
                        {
                            "pr_url": "https://github.com/example/repo/pull/3",
                            "thread_id": "audit-test",
                        },
                        config,
                    )
                    self.assertEqual(
                        result["__interrupt__"][0].value["kind"],
                        "approval_request",
                    )
                    result = await app.ainvoke(
                        Command(resume={"choice": "approve", "feedback": "Reviewed"}),
                        config,
                    )

                events = await replay_events("audit-test")

        actions = {event["action"] for event in events}
        self.assertTrue(
            {"fetch_pr", "analyze", "route", "human_approval_resolved", "commit"}
            <= actions
        )
        self.assertEqual(result["final_action"], "committed")
        self.assertEqual(len(posted), 1)


if __name__ == "__main__":
    unittest.main()

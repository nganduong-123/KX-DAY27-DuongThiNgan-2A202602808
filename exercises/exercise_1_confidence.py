"""Exercise 1 — Confidence scoring + routing.

Build a small LangGraph that fetches a PR, analyzes it, then routes to one of
three terminal nodes by confidence. Goal: see the three branches print
different messages on different PRs.

"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from rich.console import Console

from common.github import fetch_pr
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]-> fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]OK[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {
        "pr_title": pr.title, "pr_diff": pr.diff,
        "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha,
    }


def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]-> analyze[/cyan]")
    # Call the LLM with structured output PRAnalysis.
    # Hint:  llm = get_llm().with_structured_output(PRAnalysis)
    #        analysis = llm.invoke([...])
    #        return {"analysis": analysis}
    # When implemented, wrap the call in:
    #        with console.status("[dim]LLM thinking...[/dim]"):
    #            analysis = llm.invoke([...])
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM thinking...[/dim]"):
        analysis = llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer reviewing a pull request. "
                    "Return a concise, evidence-based review in the requested schema."
                ),
            },
            {
                "role": "user",
                "content": f"PR title: {state['pr_title']}\n\nDiff:\n{state['pr_diff']}",
            },
        ])
    console.print(
        f"  [green]OK[/green] confidence={analysis.confidence:.0%}, "
        f"{len(analysis.comments)} comment(s)"
    )
    return {"analysis": analysis}


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]-> route[/cyan]")
    # Read state["analysis"].confidence and return
    #       {"decision": "auto_approve" | "human_approval" | "escalate"}
    # Thresholds provided: AUTO_APPROVE_THRESHOLD (0.85) and ESCALATE_THRESHOLD (0.60).
    confidence = state["analysis"].confidence
    if confidence >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif confidence < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(
        f"  [green]OK[/green] decision=[bold]{decision}[/bold] "
        f"(confidence={confidence:.0%})"
    )
    return {"decision": decision}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[green]AUTO APPROVE[/green] - high confidence, no human needed")
    return {"final_action": "auto_approved"}


def node_human_approval(state: ReviewState) -> dict:
    console.print("[yellow]HUMAN APPROVAL[/yellow] - placeholder, exercise 2 will pause here")
    return {"final_action": "pending_human_approval"}


def node_escalate(state: ReviewState) -> dict:
    console.print("[red]ESCALATE[/red] - placeholder, exercise 3 will ask the reviewer questions")
    return {"final_action": "pending_escalation"}


def build_graph():
    g = StateGraph(ReviewState)
    # Register all six nodes and connect the shared prefix.
    # Add conditional edges on "route" with mapping
    #       {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"}
    # Connect each terminal node to END.
    for name, fn in [
        ("fetch_pr", node_fetch_pr),
        ("analyze", node_analyze),
        ("route", node_route),
        ("auto_approve", node_auto_approve),
        ("human_approval", node_human_approval),
        ("escalate", node_escalate),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route",
        lambda state: state["decision"],
        {
            "auto_approve": "auto_approve",
            "human_approval": "human_approval",
            "escalate": "escalate",
        },
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", END)
    g.add_edge("escalate", END)
    return g.compile()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 1 - confidence routing[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    final = app.invoke({"pr_url": args.pr})

    console.rule("Final")
    console.print(f"confidence = {final['analysis'].confidence:.0%}")
    console.print(f"action     = {final.get('final_action')}")


if __name__ == "__main__":
    main()

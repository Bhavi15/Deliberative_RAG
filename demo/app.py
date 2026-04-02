"""Streamlit demo application for Deliberative RAG.

Run with:
    streamlit run demo/app.py
"""

import streamlit as st


def render_header() -> None:
    """Render the page title and introductory description."""
    pass


def render_query_input() -> str | None:
    """Render the query text input and submit button.

    Returns:
        The submitted query string, or None if not yet submitted.
    """
    pass


def render_answer(answer) -> None:
    """Render the generated answer with confidence badge.

    Args:
        answer: Answer schema object from the pipeline.
    """
    pass


def render_reasoning_trace(trace) -> None:
    """Render the reasoning trace in an expandable section.

    Args:
        trace: ReasoningTrace object to display.
    """
    pass


def render_conflict_graph(edges) -> None:
    """Render a visual summary of detected conflicts.

    Args:
        edges: List of ConflictEdge objects to visualise.
    """
    pass


def main() -> None:
    """Application entry point — wire together all UI components."""
    pass


if __name__ == "__main__":
    main()

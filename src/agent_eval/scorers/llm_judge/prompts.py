"""Prompt templates for LLM-as-Judge evaluation.

Each template receives an Episode's data and produces a structured
evaluation request for an LLM.
"""

from __future__ import annotations

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of AI agent conversations. You assess agent \
performance across multiple dimensions with precision and consistency.

You will receive a transcript of an agent interaction and must evaluate it \
on the requested dimensions. For each dimension, provide:
1. A score (float between 0.0 and 1.0)
2. A brief justification (1-2 sentences)

Respond ONLY with valid JSON matching the requested schema. No extra text."""

SINGLE_DIMENSION_TEMPLATE = """\
Evaluate the following agent interaction on the dimension: **{dimension_name}**

Dimension description: {dimension_description}

---
AGENT INTERACTION:
{transcript}
---

{final_answer_section}

Respond with JSON:
{{
  "dimension": "{dimension_name}",
  "score": <float 0.0-1.0>,
  "justification": "<1-2 sentence explanation>"
}}"""

MULTI_DIMENSION_TEMPLATE = """\
Evaluate the following agent interaction on ALL of the following dimensions:

{dimensions_block}

---
AGENT INTERACTION:
{transcript}
---

{final_answer_section}

Respond with JSON:
{{
  "evaluations": [
    {{
      "dimension": "<name>",
      "score": <float 0.0-1.0>,
      "justification": "<1-2 sentence explanation>"
    }}
  ]
}}"""

# Pre-defined evaluation dimensions
DEFAULT_DIMENSIONS: dict[str, str] = {
    "relevance": (
        "How relevant is the agent's response to the user's question? "
        "Does it address what was asked?"
    ),
    "groundedness": (
        "Are the agent's claims grounded in the tool results and data retrieved? "
        "Does it avoid fabricating information?"
    ),
    "completeness": (
        "Does the response thoroughly address all aspects of the question? "
        "Are there gaps or missing information?"
    ),
    "coherence": (
        "Is the response well-organized, logically structured, and easy to follow?"
    ),
    "tool_usage": (
        "Did the agent use appropriate tools effectively? "
        "Were the right tools called with correct parameters?"
    ),
}


def build_transcript(steps_text: list[str], max_chars: int = 8000) -> str:
    """Join step descriptions into a transcript, truncating if needed."""
    full = "\n".join(steps_text)
    if len(full) > max_chars:
        return full[:max_chars] + "\n... [truncated]"
    return full


def format_single_prompt(
    dimension_name: str,
    dimension_description: str,
    transcript: str,
    final_answer: str | None = None,
) -> str:
    """Format a prompt for evaluating a single dimension."""
    fa_section = ""
    if final_answer:
        fa_section = f"FINAL ANSWER:\n{final_answer}\n"

    return SINGLE_DIMENSION_TEMPLATE.format(
        dimension_name=dimension_name,
        dimension_description=dimension_description,
        transcript=transcript,
        final_answer_section=fa_section,
    )


def format_multi_prompt(
    dimensions: dict[str, str],
    transcript: str,
    final_answer: str | None = None,
) -> str:
    """Format a prompt for evaluating multiple dimensions at once."""
    dim_lines = []
    for i, (name, desc) in enumerate(dimensions.items(), 1):
        dim_lines.append(f"{i}. **{name}**: {desc}")
    dimensions_block = "\n".join(dim_lines)

    fa_section = ""
    if final_answer:
        fa_section = f"FINAL ANSWER:\n{final_answer}\n"

    return MULTI_DIMENSION_TEMPLATE.format(
        dimensions_block=dimensions_block,
        transcript=transcript,
        final_answer_section=fa_section,
    )

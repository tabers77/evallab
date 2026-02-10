"""JSON report output."""

from __future__ import annotations

import json

from agent_eval.pipeline.runner import EvalResult


def to_json(result: EvalResult, indent: int = 2) -> str:
    """Serialize an EvalResult to JSON."""
    return json.dumps(_result_to_dict(result), indent=indent, default=str)


def to_json_batch(results: list[EvalResult], indent: int = 2) -> str:
    """Serialize multiple EvalResults to JSON."""
    return json.dumps([_result_to_dict(r) for r in results], indent=indent, default=str)


def _result_to_dict(result: EvalResult) -> dict:
    overall_dim = result.score_vector.dimension_by_name("overall_score")
    return {
        "episode_id": result.episode.episode_id,
        "source_framework": result.episode.source_framework,
        "source_path": result.episode.metadata.get("source_path"),
        "grade": result.grade,
        "score_vector": result.score_vector.to_dict(),
        "overall_score": overall_dim.value if overall_dim else None,
        "summary": result.summary,
        "episode_metadata": {
            "agents": sorted(result.episode.agents),
            "n_steps": len(result.episode.steps),
            "has_final_answer": result.episode.final_answer is not None,
            "duration_seconds": result.episode.duration_seconds,
        },
    }

"""LLM-as-Judge scorer.

Uses any LLM callable to evaluate agent episodes on configurable
dimensions. Model-agnostic — works with OpenAI, Azure, Anthropic,
or any function that takes a prompt and returns text.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from agent_eval.core.models import Episode, StepKind
from agent_eval.core.score import Issue, ScoreDimension, Severity
from agent_eval.scorers.llm_judge.prompts import (
    DEFAULT_DIMENSIONS,
    JUDGE_SYSTEM_PROMPT,
    build_transcript,
    format_multi_prompt,
    format_single_prompt,
)

logger = logging.getLogger(__name__)

# The LLM callable is invoked with KEYWORD arguments only:
#     llm_fn(system_prompt=..., user_prompt=...) -> response text
#
# 0.3.0 made this keyword-only (audit F7). The old contract was positional
# ``(system_prompt, user_prompt)``, and the intelligence-platform adapter
# declared ``def llm_fn(user_prompt, system_prompt="")`` — so the judge's
# system prompt was delivered as the user message and the transcript as the
# system message, on every call, silently. Positional order cannot express
# that mistake once the call site names its arguments.
LLMCallable = Callable[..., str]


class LLMJudgeScorer:
    """Evaluate episodes using an LLM as judge.

    Parameters
    ----------
    llm_fn
        A callable invoked as ``llm_fn(system_prompt=..., user_prompt=...)``
        returning the response text. Keyword-only since 0.3.0 — see the
        ``LLMCallable`` note above.
    dimensions
        Dict mapping dimension names to descriptions.
        Defaults to :data:`DEFAULT_DIMENSIONS`.
    batch_dimensions
        If True, sends all dimensions in a single LLM call.
        If False, sends one call per dimension (more reliable but slower).
    max_transcript_chars
        Maximum characters for the transcript in the prompt.
    low_score_threshold
        Scores below this trigger a WARNING issue.
    """

    def __init__(
        self,
        llm_fn: LLMCallable,
        dimensions: dict[str, str] | None = None,
        batch_dimensions: bool = True,
        max_transcript_chars: int = 8000,
        low_score_threshold: float = 0.4,
    ) -> None:
        self.llm_fn = llm_fn
        self.dimensions = dimensions or DEFAULT_DIMENSIONS
        # 0.3.0 — memoise per episode (audit F6). A pipeline calls score()
        # and detect_issues(); each used to run the judge, so every evaluation
        # paid for two LLM round-trips AND the issues could be derived from a
        # different judge response than the score, letting a stored score
        # contradict its stored issues.
        self._cache: tuple[str, list[ScoreDimension]] | None = None
        self.batch_dimensions = batch_dimensions
        self.max_transcript_chars = max_transcript_chars
        self.low_score_threshold = low_score_threshold

    @property
    def name(self) -> str:
        return "llm_judge"

    def score(self, episode: Episode) -> list[ScoreDimension]:
        """Evaluate the episode on all configured dimensions.

        Memoised per ``episode_id`` so a pipeline that also calls
        :meth:`detect_issues` does not pay for a second judge call.
        """
        if self._cache is not None and self._cache[0] == episode.episode_id:
            return self._cache[1]

        transcript = self._build_transcript(episode)
        if self.batch_dimensions:
            dims = self._score_batch(transcript, episode.final_answer)
        else:
            dims = self._score_individual(transcript, episode.final_answer)

        self._cache = (episode.episode_id, dims)
        return dims

    def detect_issues(self, episode: Episode) -> list[Issue]:
        """Flag dimensions with low scores as issues."""
        dims = self.score(episode)
        issues: list[Issue] = []
        for dim in dims:
            # An abstention is an absence of measurement, not a low score —
            # flagging it would smuggle F5's false signal back in as an issue.
            if dim.abstained:
                continue
            if dim.normalized < self.low_score_threshold:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        category="LLM Judge",
                        description=(
                            f"Low score on '{dim.name}': "
                            f"{dim.value:.2f}/{dim.max_value:.2f}"
                        ),
                    )
                )
        return issues

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_transcript(self, episode: Episode) -> str:
        """Convert episode steps into a readable transcript."""
        lines: list[str] = []
        for step in episode.steps:
            if step.kind == StepKind.MESSAGE and step.content:
                lines.append(f"[{step.agent_name}]: {step.content}")
            elif step.kind == StepKind.TOOL_CALL:
                result_preview = str(step.tool_result)[:200] if step.tool_result else ""
                status = "OK" if step.tool_succeeded else "FAILED"
                lines.append(f"[Tool: {step.tool_name}] ({status}) -> {result_preview}")
            elif step.kind == StepKind.FACT_CHECK:
                verdict = step.metadata.get("verdict", "?")
                lines.append(f"[FactCheck: {step.agent_name}] {verdict}")
        return build_transcript(lines, self.max_transcript_chars)

    def _score_batch(
        self, transcript: str, final_answer: str | None
    ) -> list[ScoreDimension]:
        """Send all dimensions in a single LLM call."""
        prompt = format_multi_prompt(self.dimensions, transcript, final_answer)
        try:
            response = self.llm_fn(
                system_prompt=JUDGE_SYSTEM_PROMPT, user_prompt=prompt
            )
            return self._parse_multi_response(response)
        except Exception as e:
            logger.error("LLM judge batch call failed — abstaining: %s", e)
            return self._abstained_dimensions(str(e))

    def _score_individual(
        self, transcript: str, final_answer: str | None
    ) -> list[ScoreDimension]:
        """Send one LLM call per dimension."""
        results: list[ScoreDimension] = []
        for dim_name, dim_desc in self.dimensions.items():
            prompt = format_single_prompt(dim_name, dim_desc, transcript, final_answer)
            try:
                response = self.llm_fn(
                    system_prompt=JUDGE_SYSTEM_PROMPT, user_prompt=prompt
                )
                parsed = self._parse_single_response(response, dim_name)
                results.append(parsed)
            except Exception as e:
                logger.error(
                    "LLM judge call failed for %s — abstaining: %s", dim_name, e
                )
                results.append(self._abstain(dim_name, str(e)))
        return results

    def _parse_multi_response(self, response: str) -> list[ScoreDimension]:
        """Parse the JSON response from a batch evaluation."""
        try:
            data = json.loads(self._extract_json(response))
            evaluations = data.get("evaluations", [])
            dims: list[ScoreDimension] = []
            for ev in evaluations:
                score = float(ev.get("score", 0.0))
                score = max(0.0, min(1.0, score))
                dims.append(
                    ScoreDimension(
                        name=ev.get("dimension", "unknown"),
                        value=score,
                        max_value=1.0,
                        source=self.name,
                        # The prompt has always asked for this and the model
                        # has always returned it; 0.3.0 stops discarding it
                        # (audit F8).
                        justification=ev.get("justification"),
                    )
                )
            # Fill in any missing dimensions
            found_names = {d.name for d in dims}
            for dim_name in self.dimensions:
                if dim_name not in found_names:
                    # The model did not return this dimension, so it was not
                    # evaluated. Scoring it 0.0 asserted a judgement nobody
                    # made (audit F5).
                    dims.append(
                        self._abstain(dim_name, "dimension missing from response")
                    )
            return dims
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(
                "Failed to parse LLM judge response — abstaining: %s", e
            )
            return self._abstained_dimensions(f"unparseable response: {e}")

    def _parse_single_response(self, response: str, dim_name: str) -> ScoreDimension:
        """Parse the JSON response from a single-dimension evaluation."""
        try:
            data = json.loads(self._extract_json(response))
            score = float(data.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            return ScoreDimension(
                name=dim_name,
                value=score,
                max_value=1.0,
                source=self.name,
                justification=data.get("justification"),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(
                "Failed to parse LLM judge response for %s — abstaining: %s",
                dim_name,
                e,
            )
            return self._abstain(dim_name, f"unparseable response: {e}")

    def _abstain(self, dim_name: str, reason: str) -> ScoreDimension:
        """A dimension the judge could not evaluate.

        0.3.0 replaced the previous zero-scored fallback (audit F5). A 0.0 was
        indistinguishable from a genuinely terrible answer and was averaged
        into the aggregate, so a transient API error read as a real quality
        drop. ``value`` is 0.0 only to satisfy the type; ``abstained`` is what
        consumers must read, and ``ScoreVector.overall`` skips these.
        """
        return ScoreDimension(
            name=dim_name,
            value=0.0,
            max_value=1.0,
            source=self.name,
            abstained=True,
            justification=f"abstained: {reason}",
        )

    def _abstained_dimensions(self, reason: str) -> list[ScoreDimension]:
        """Abstain on every configured dimension."""
        return [self._abstain(name, reason) for name in self.dimensions]

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from LLM response that may contain extra text."""
        text = text.strip()
        # Try to find JSON block in markdown code fence
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()
        # Try to find JSON object directly
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            return text[first_brace : last_brace + 1]
        return text

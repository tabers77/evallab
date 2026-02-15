"""Bridge between agent_eval and HuggingFace TRL (v0.12+).

Wraps EvalPipeline + RewardFunction into a callable compatible with
TRL's ``GRPOTrainer(reward_funcs=[...])``.

TRL v0.12+ reward functions receive an expanded signature:
    (prompts, completions, *, completion_ids, trainer_state, **dataset_cols)
        -> list[float | None]

Where ``prompts`` and ``completions`` can be either ``list[str]`` (plain
text) or ``list[list[dict]]`` (conversational message format).

This module provides ``GRPORewardBridge`` which:
1. Converts each (prompt, completion) pair into a minimal Episode
2. Runs it through the eval pipeline's scorers
3. Returns scalar rewards via the configured RewardFunction

Requires ``trl`` as optional dependency (``pip install agent-eval[rl]``).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Union

from agent_eval.core.models import Episode, Step, StepKind
from agent_eval.core.score import Issue, ScoreDimension, ScoreVector

# Type alias for TRL-compatible reward function (may return None per-sample)
TRLRewardFn = Callable[..., list[Union[float, None]]]

# Check for TRL availability
try:
    import trl  # noqa: F401

    _TRL_AVAILABLE = True
except ImportError:
    _TRL_AVAILABLE = False


class GRPORewardBridge:
    """Wrap agent_eval evaluation as a TRL reward function.

    Parameters
    ----------
    scorers
        List of Scorer instances to run on each episode.
    reward_fn
        A RewardFunction that converts ScoreVector to scalar reward.
    episode_builder
        Optional custom function to build an Episode from
        ``(prompt, completion)`` pairs.  If not provided, a default
        builder creates a two-step Episode (user message + assistant
        message).  Accepts both plain strings and conversational
        message dicts.
    tools
        Optional list of tool callables for TRL agent training.
        When provided and no custom ``episode_builder`` is given,
        completions are parsed with a tool-aware builder that maps
        tool call / tool result messages to the corresponding StepKind.

    Example
    -------
    ::

        from agent_eval.rl.trl_bridge import GRPORewardBridge
        from agent_eval.rewards import WeightedSumReward
        from agent_eval.scorers.rules import IssueDetectorScorer

        bridge = GRPORewardBridge(
            scorers=[IssueDetectorScorer()],
            reward_fn=WeightedSumReward(),
        )
        reward_fn = bridge.as_trl_reward_fn()

        # Use with TRL
        from trl import GRPOTrainer
        trainer = GRPOTrainer(..., reward_funcs=[reward_fn])

        # Or use the helper for kwargs passthrough
        trainer = GRPOTrainer(..., **bridge.as_trl_trainer_kwargs())
    """

    def __init__(
        self,
        scorers: list[Any],
        reward_fn: Any,
        episode_builder: Callable[..., Episode] | None = None,
        tools: list[Callable[..., Any]] | None = None,
    ) -> None:
        self.scorers = scorers
        self.reward_fn = reward_fn
        self.tools = tools

        if episode_builder is not None:
            self.episode_builder = episode_builder
        elif tools:
            self.episode_builder = _tool_aware_episode_builder
        else:
            self.episode_builder = _default_episode_builder

    def compute_reward(self, prompt: Any, completion: Any) -> float:
        """Compute a scalar reward for a single (prompt, completion) pair."""
        episode = self.episode_builder(prompt, completion)

        all_dimensions: list[ScoreDimension] = []
        all_issues: list[Issue] = []

        for scorer in self.scorers:
            all_dimensions.extend(scorer.score(episode))
            all_issues.extend(scorer.detect_issues(episode))

        score_vector = ScoreVector(
            episode_id=episode.episode_id,
            dimensions=all_dimensions,
            issues=all_issues,
        )

        return self.reward_fn.compute(score_vector)

    def compute_rewards(
        self,
        prompts: list[Any],
        completions: list[Any],
        **kwargs: Any,
    ) -> list[float]:
        """Compute rewards for a batch — TRL-compatible signature.

        Extra keyword arguments (e.g. ``completion_ids``,
        ``trainer_state``, dataset columns) are accepted and ignored,
        keeping compatibility with TRL v0.12+ expanded signatures.
        """
        if len(prompts) != len(completions):
            raise ValueError(
                f"prompts ({len(prompts)}) and completions ({len(completions)}) "
                "must have the same length"
            )
        return [self.compute_reward(p, c) for p, c in zip(prompts, completions)]

    def as_trl_reward_fn(self) -> TRLRewardFn:
        """Return a callable compatible with ``GRPOTrainer(reward_funcs=[...])``."""
        return self.compute_rewards

    def as_trl_trainer_kwargs(self) -> dict[str, Any]:
        """Return kwargs dict for ``GRPOTrainer`` constructor.

        Includes ``reward_funcs`` and, when tools are configured,
        ``tools`` — so callers can simply unpack:

        ::

            trainer = GRPOTrainer(..., **bridge.as_trl_trainer_kwargs())
        """
        kwargs: dict[str, Any] = {"reward_funcs": [self.compute_rewards]}
        if self.tools:
            kwargs["tools"] = self.tools
        return kwargs


# ---------------------------------------------------------------------------
# Episode builders
# ---------------------------------------------------------------------------

def _extract_text(value: Any) -> str:
    """Extract plain text from a string or conversational message list."""
    if isinstance(value, list):
        # Conversational format: list of {"role": ..., "content": ...}
        return value[-1]["content"] if value else ""
    return value


def _default_episode_builder(prompt: Any, completion: Any) -> Episode:
    """Build a minimal Episode from a (prompt, completion) pair.

    Handles both plain strings and conversational message-dict lists.
    """
    prompt_text = _extract_text(prompt)
    completion_text = _extract_text(completion)

    steps = [
        Step(
            kind=StepKind.MESSAGE,
            agent_id="user",
            agent_name="user",
            content=prompt_text,
        ),
        Step(
            kind=StepKind.MESSAGE,
            agent_id="assistant",
            agent_name="assistant",
            content=completion_text,
        ),
    ]
    return Episode(
        episode_id="trl_episode",
        steps=steps,
        source_framework="trl",
        final_answer=completion_text,
    )


def _tool_aware_episode_builder(prompt: Any, completion: Any) -> Episode:
    """Build an Episode that maps tool-call messages to dedicated StepKinds.

    In TRL agent training the completion may contain assistant messages
    with ``tool_calls`` and subsequent ``tool`` role messages carrying
    results.  This builder converts those into ``TOOL_CALL`` /
    ``TOOL_RESULT`` steps for richer evaluation.

    Falls back to the default builder when the inputs are plain strings.
    """
    if isinstance(prompt, str) and isinstance(completion, str):
        return _default_episode_builder(prompt, completion)

    steps: list[Step] = []

    # --- prompt messages ---
    prompt_msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    for msg in prompt_msgs:
        steps.append(Step(
            kind=StepKind.MESSAGE,
            agent_id=msg.get("role", "user"),
            agent_name=msg.get("role", "user"),
            content=msg.get("content", ""),
        ))

    # --- completion messages ---
    completion_msgs = completion if isinstance(completion, list) else [{"role": "assistant", "content": completion}]
    final_text = ""
    for msg in completion_msgs:
        role = msg.get("role", "assistant")

        if role == "tool":
            # Tool result message
            steps.append(Step(
                kind=StepKind.TOOL_RESULT,
                agent_id="tool",
                agent_name=msg.get("name", "tool"),
                content=msg.get("content", ""),
            ))
        elif msg.get("tool_calls"):
            # Assistant message that contains tool calls
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args_raw = fn.get("arguments", "")
                args_str = args_raw if isinstance(args_raw, str) else json.dumps(args_raw)
                steps.append(Step(
                    kind=StepKind.TOOL_CALL,
                    agent_id="assistant",
                    agent_name="assistant",
                    content=f"{fn.get('name', 'unknown')}({args_str})",
                ))
        else:
            content = msg.get("content", "")
            steps.append(Step(
                kind=StepKind.MESSAGE,
                agent_id=role,
                agent_name=role,
                content=content,
            ))
            if role == "assistant" and content:
                final_text = content

    return Episode(
        episode_id="trl_episode",
        steps=steps,
        source_framework="trl",
        final_answer=final_text,
    )

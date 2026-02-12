# agent-eval

Framework-agnostic LLM/Agent evaluator with RL support.

```
Adapters --> Episode/Step --> Scorers --> ScoreVector --> Rewards --> RL Training
```

Converts traces from any multi-agent framework (AutoGen, LangGraph) into a canonical trajectory format, runs pluggable scorers, and produces multi-dimensional evaluation results. Scores can feed human-readable reports, RL reward functions, or prompt optimization loops.

## Quick Start

```bash
# No installation needed — run from the repo
cd agent_eval
PYTHONPATH=src agent-eval evaluate path/to/event.txt

# Or use Python directly
PYTHONPATH=src python -m agent_eval.cli.main evaluate path/to/event.txt
```

## Architecture

```
agent_eval/
  src/agent_eval/
    core/           # Episode, Step, ScoreVector, Protocol classes
    adapters/       # AutoGen, LangGraph trace -> Episode converters
    scorers/        # Numeric, Rules, LLM Judge, DeepEval, Ragas
    rewards/        # WeightedSum, Deduction, Composite reward functions
    rl/             # TuningLoop, TRL bridge, DSPy bridge, reward server
    pipeline/       # EvalPipeline orchestration, batch, comparison
    reporting/      # Text, JSON, HTML report formatters
    cli/            # Command-line interface (evaluate, compare, serve)
    utils/          # Embeddings, cosine similarity
    experimental/   # PPE reward model benchmarking (arXiv:2410.14872)
```

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Episode** | A complete agent interaction trajectory (list of Steps) |
| **Step** | One event: message, tool call, tool result, LLM call, or fact check |
| **TraceAdapter** | Converts framework-specific logs into Episodes |
| **Scorer** | Evaluates an Episode, returning ScoreDimensions + Issues |
| **ScoreVector** | Multi-dimensional evaluation result with issues |
| **RewardFunction** | Converts a ScoreVector into a scalar reward for RL |

## Testing

All 419 tests run without installing the package:

```bash
cd agent_eval

# Run all tests
PYTHONPATH=src pytest tests/ -v

# Run tests for a specific module
PYTHONPATH=src pytest tests/core/ -v
PYTHONPATH=src pytest tests/adapters/autogen/ -v
PYTHONPATH=src pytest tests/adapters/langgraph/ -v
PYTHONPATH=src pytest tests/scorers/ -v
PYTHONPATH=src pytest tests/rewards/ -v
PYTHONPATH=src pytest tests/rl/ -v
PYTHONPATH=src pytest tests/pipeline/ -v
PYTHONPATH=src pytest tests/cli/ -v
PYTHONPATH=src pytest tests/reporting/ -v
PYTHONPATH=src pytest tests/utils/ -v
PYTHONPATH=src pytest tests/experimental/ -v
```

### What Each Test Module Covers

| Test Module | What It Tests |
|-------------|---------------|
| `tests/core/` | Episode, Step, StepKind dataclasses; ScoreVector, ScoreDimension, Issue, Severity |
| `tests/adapters/autogen/` | JSON event extraction (brace-counting parser), tool failure detection, AutoGen log -> Episode conversion |
| `tests/adapters/langgraph/` | LangGraph `astream_events` JSON -> Episode, checkpoint reader, tool/LLM call detection |
| `tests/scorers/numeric/` | Number extraction from text and tool results, numeric fabrication detection |
| `tests/scorers/rules/` | Deduction-based scoring (100pt base), 6-category issue detection |
| `tests/scorers/llm_judge/` | LLM-as-Judge with mock functions, batch/individual modes, JSON fallback |
| `tests/scorers/deepeval/` | Import guard when deepeval not installed |
| `tests/scorers/test_registry.py` | Scorer auto-registration, plugin discovery |
| `tests/rewards/` | WeightedSum, Deduction, Composite reward aggregation |
| `tests/rl/` | TuningLoop with mock callbacks, TRL bridge, DSPy bridge, reward server endpoints |
| `tests/pipeline/` | Full pipeline orchestration, batch evaluation, multi-run comparison |
| `tests/cli/` | CLI argument parsing and output for evaluate, compare, serve |
| `tests/reporting/` | HTML report generation, XSS escaping, batch report |
| `tests/utils/` | Cosine similarity, embedding helper factory |
| `tests/experimental/ppe/` | PPE metrics, synthetic dataset generation, benchmark runner, reporting |

## CLI Commands

### Evaluate a Log

```bash
# Text report (default)
PYTHONPATH=src agent-eval evaluate path/to/event.txt

# JSON output
PYTHONPATH=src agent-eval evaluate path/to/event.txt --format json

# Brief mode (skip detailed metrics)
PYTHONPATH=src agent-eval evaluate path/to/event.txt --brief

# Save report to file
PYTHONPATH=src agent-eval evaluate path/to/event.txt --output report.txt

# Choose specific scorers
PYTHONPATH=src agent-eval evaluate path/to/event.txt --scorers numeric issue_detector

# Evaluate multiple logs
PYTHONPATH=src agent-eval evaluate log1/event.txt log2/event.txt

# With custom agent names
PYTHONPATH=src agent-eval evaluate path/to/event.txt --agents FinanceExpert CustomerResearcher
```

### Compare Two Runs

```bash
# Side-by-side comparison
PYTHONPATH=src agent-eval compare baseline/event.txt experiment/event.txt

# With custom labels
PYTHONPATH=src agent-eval compare old/event.txt new/event.txt --label-a "v1.0" --label-b "v2.0"

# JSON comparison output
PYTHONPATH=src agent-eval compare old/event.txt new/event.txt --format json
```

### Start Reward Server (for RL)

```bash
# Default: issue_detector scorer on port 8000
PYTHONPATH=src agent-eval serve

# Custom port and scorers
PYTHONPATH=src agent-eval serve --port 9000 --scorers numeric issue_detector

# The server exposes:
#   GET  /health   -> {"status": "ok", "scorers": [...]}
#   POST /reward   -> {"rewards": [0.85, 0.72, ...]}
```

## Adapting for Real Cases

### 1. Evaluating AutoGen Logs

The AutoGen adapter reads `event.txt` files produced by AutoGen's event logging.

```python
from agent_eval.adapters.autogen import AutoGenAdapter
from agent_eval.scorers.numeric import NumericConsistencyScorer
from agent_eval.scorers.rules import IssueDetectorScorer
from agent_eval.pipeline.runner import EvalPipeline

adapter = AutoGenAdapter(
    agent_names=["FinanceExpert", "CustomerResearcher", "DataVisualiser"],
    orchestrator_name="SalesNegotiator",
)

pipeline = EvalPipeline(
    adapter=adapter,
    scorers=[NumericConsistencyScorer(), IssueDetectorScorer()],
)

# Single log
result = pipeline.evaluate_from_source("path/to/logs/run_001/event.txt")
print(f"Score: {result.grade} ({result.score_vector.dimension_by_name('overall_score').value}/100)")
print(result.summary)

# Batch: all event.txt files under a directory
results = pipeline.evaluate_batch("path/to/logs/")
for r in results:
    print(f"{r.episode.metadata['source_path']}: {r.grade}")
```

### 2. Evaluating LangGraph Traces

The LangGraph adapter reads JSON traces from `graph.astream_events()`.

```python
from agent_eval.adapters.langgraph import LangGraphAdapter
from agent_eval.scorers.rules import IssueDetectorScorer
from agent_eval.pipeline.runner import EvalPipeline

adapter = LangGraphAdapter(default_agent_name="sales_agent")
pipeline = EvalPipeline(adapter=adapter, scorers=[IssueDetectorScorer()])

# From a saved JSON trace file
result = pipeline.evaluate_from_source("traces/run_001.json")

# From a dict (e.g. captured at runtime)
from agent_eval.adapters.langgraph import LangGraphAdapter
adapter = LangGraphAdapter()
episode = adapter.load_from_dict(trace_dict)
result = pipeline.evaluate(episode)
```

**Saving LangGraph traces for evaluation:**

```python
import json

# During your LangGraph execution:
events = []
async for event in graph.astream_events(input_data, version="v2"):
    events.append(event)

# Save for later evaluation
trace = {"thread_id": "run-001", "events": events}
with open("trace.json", "w") as f:
    json.dump(trace, f)
```

### 3. Using LangGraph Checkpoints

If your LangGraph app uses `MemorySaver` or `SqliteSaver`, you can evaluate
from checkpoints:

```python
from agent_eval.adapters.langgraph import (
    load_checkpoints,
    latest_checkpoint_to_episode,
)

checkpoints = load_checkpoints("trace_with_checkpoints.json")
episode = latest_checkpoint_to_episode(checkpoints)
# Now evaluate the episode through any pipeline
```

### 4. Adding LLM-as-Judge Scoring

The LLM Judge scorer works with any LLM — you provide a callable:

```python
from agent_eval.scorers.llm_judge import LLMJudgeScorer

# Option A: OpenAI
from openai import OpenAI
client = OpenAI()

def call_llm(system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content

judge = LLMJudgeScorer(llm_fn=call_llm)

# Option B: Azure OpenAI
from openai import AzureOpenAI
client = AzureOpenAI(azure_endpoint="...", api_key="...", api_version="...")

def call_azure(system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content

judge = LLMJudgeScorer(llm_fn=call_azure)

# Option C: Custom dimensions
judge = LLMJudgeScorer(
    llm_fn=call_llm,
    dimensions={
        "accuracy": "How factually accurate is the response?",
        "helpfulness": "How helpful is the response to the user?",
        "safety": "Does the response avoid harmful content?",
    },
)

# Use in a pipeline
pipeline = EvalPipeline(
    adapter=adapter,
    scorers=[NumericConsistencyScorer(), IssueDetectorScorer(), judge],
)
```

### 5. Custom Scorers

Implement the `Scorer` protocol — no imports needed:

```python
from agent_eval.core.models import Episode
from agent_eval.core.score import ScoreDimension, Issue, Severity

class ResponseLengthScorer:
    """Score based on response length — penalize too short or too long."""

    @property
    def name(self) -> str:
        return "response_length"

    def score(self, episode: Episode) -> list[ScoreDimension]:
        answer = episode.final_answer or ""
        length = len(answer)
        # Sweet spot: 100-500 chars
        if 100 <= length <= 500:
            value = 1.0
        elif length < 100:
            value = length / 100
        else:
            value = max(0.0, 1.0 - (length - 500) / 1000)
        return [ScoreDimension(name="response_length", value=value, source=self.name)]

    def detect_issues(self, episode: Episode) -> list[Issue]:
        answer = episode.final_answer or ""
        issues = []
        if len(answer) < 20:
            issues.append(Issue(Severity.ERROR, "Answer Quality", "Response is extremely short"))
        if len(answer) > 2000:
            issues.append(Issue(Severity.WARNING, "Efficiency", "Response is excessively long"))
        return issues

# Register it
from agent_eval.scorers.registry import default_registry
default_registry.register("response_length", ResponseLengthScorer)

# Now usable via CLI: --scorers response_length
```

### 6. Writing a Custom Adapter (New Framework)

Implement the `TraceAdapter` protocol:

```python
from agent_eval.core.models import Episode, Step, StepKind

class CrewAIAdapter:
    """Convert CrewAI execution logs into Episodes."""

    @property
    def framework_name(self) -> str:
        return "crewai"

    def load_episode(self, source: str, **kwargs) -> Episode:
        # Parse your framework's log format
        # Convert to Episode with Steps
        steps = [
            Step(kind=StepKind.MESSAGE, agent_id="agent1", agent_name="Researcher",
                 content="Searching for information..."),
            Step(kind=StepKind.TOOL_CALL, agent_id="agent1", agent_name="Researcher",
                 tool_name="web_search", tool_args={"query": "..."}, tool_succeeded=True),
            # ... more steps
        ]
        return Episode(
            episode_id="crewai-run-001",
            steps=steps,
            source_framework="crewai",
            final_answer="The result is ...",
        )

    def load_episodes(self, source: str, **kwargs) -> list[Episode]:
        # Load multiple from a directory
        ...
```

### 7. RL Integration with TRL (GRPO)

Use evaluation scores as reward signals for TRL's GRPOTrainer:

```python
from agent_eval.rl.trl_bridge import GRPORewardBridge
from agent_eval.rewards import WeightedSumReward
from agent_eval.scorers.rules import IssueDetectorScorer

bridge = GRPORewardBridge(
    scorers=[IssueDetectorScorer()],
    reward_fn=WeightedSumReward(),
)

# Get a TRL-compatible function
reward_fn = bridge.as_trl_reward_fn()

# Use with TRL
from trl import GRPOTrainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    # ...
)
```

### 8. Prompt Optimization Loop

Run iterative prompt tuning with pluggable callbacks:

```python
from agent_eval.rl.tuning_loop import TuningLoop
from agent_eval.core.score import ScoreDimension, ScoreVector

def my_test_runner(prompt: str) -> list[dict]:
    """Run your agent with this prompt on test cases."""
    # Return list of test results
    return [{"input": "q1", "output": run_agent(prompt, "q1")}]

def my_evaluator(prompt: str, results: list[dict]) -> ScoreVector:
    """Evaluate the test results."""
    # Run through your evaluation pipeline
    score = compute_score(results)
    dims = [ScoreDimension(name="quality", value=score, max_value=1.0)]
    return ScoreVector(episode_id="eval", dimensions=dims)

def my_editor(prompt, score_vector, results):
    """Use an LLM to propose an improved prompt."""
    new_prompt = call_llm(f"Improve this prompt based on score {score_vector.overall}: {prompt}")
    return new_prompt, "Improved clarity and specificity"

loop = TuningLoop(
    test_runner=my_test_runner,
    evaluator=my_evaluator,
    editor=my_editor,
    reward_fn=lambda sv: sv.overall,
    max_iterations=10,
    target_reward=0.9,
)

result = loop.run("You are a helpful assistant...")
print(f"Best prompt (iteration {result.best_iteration}): {result.best_prompt}")
print(f"Best reward: {result.best_reward:.3f}")
print(f"Converged: {result.converged}")
```

### 9. DSPy MIPROv2 Integration

Use agent_eval scores as the optimization metric for DSPy:

```python
from agent_eval.rl.dspy_bridge import DSPyMetricBridge
from agent_eval.rewards import WeightedSumReward
from agent_eval.scorers.rules import IssueDetectorScorer

metric = DSPyMetricBridge(
    scorers=[IssueDetectorScorer()],
    reward_fn=WeightedSumReward(),
    input_field="question",
    output_field="answer",
    threshold=0.7,  # returns True/False for MIPROv2
)

# Use with DSPy
import dspy
teleprompter = dspy.MIPROv2(metric=metric.as_dspy_metric())
optimized_program = teleprompter.compile(program, trainset=examples)
```

### 10. HTML Reports

Generate self-contained HTML reports:

```python
from agent_eval.reporting.html import format_html_report, format_html_batch

# Single report
html = format_html_report(result, verbose=True)
with open("report.html", "w") as f:
    f.write(html)

# Batch comparison
html = format_html_batch(results)
with open("batch_report.html", "w") as f:
    f.write(html)
```

### 11. Composite Rewards for Multi-Objective RL

Combine multiple reward signals with custom weights:

```python
from agent_eval.rewards import WeightedSumReward, DeductionReward, CompositeReward

composite = CompositeReward(
    components=[
        (WeightedSumReward(weights={"accuracy": 2.0, "coherence": 1.0}), 3.0),
        (DeductionReward(critical_penalty=0.5), 1.0),
    ],
    normalize=True,
)

reward = composite.compute(score_vector)
breakdown = composite.compute_breakdown(score_vector)  # per-component diagnostics
```

## Built-in Scorers

| Name | Class | What It Checks |
|------|-------|----------------|
| `numeric` | `NumericConsistencyScorer` | Detects fabricated numbers in answers that don't appear in tool results |
| `issue_detector` | `IssueDetectorScorer` | 6 categories: Answer Quality, Tool Usage, Agent Coordination, Efficiency, Data Accuracy, Errors |
| `rule_based` | `RuleBasedScorer` | Deduction-based scoring (100pt base, severity penalties, bonuses) |
| `llm_judge` | `LLMJudgeScorer` | LLM-as-Judge on configurable dimensions (requires `llm_fn` argument) |
| `deepeval` | `DeepEvalScorer` | Wraps any DeepEval metric (requires `pip install deepeval`) |
| `ragas` | `RagasScorer` | Wraps Ragas metrics for RAG evaluation (requires `pip install ragas`) |

## Optional Dependencies

```bash
# Core only (zero dependencies)
PYTHONPATH=src agent-eval evaluate ...

# For RL features
pip install trl torch numpy

# For DeepEval metrics
pip install deepeval

# For Ragas metrics
pip install ragas

# For reward server
pip install fastapi uvicorn

# Everything (if installing as package)
pip install -e ".[all]"
```

## Next Steps and Research Directions

The following areas are open for exploration and would strengthen the evaluator's capabilities. Each includes pointers to relevant papers and tools for reference.

### 1. Multi-Agent Credit Assignment

**Problem**: When a multi-agent team fails, which agent caused the failure? Current scoring gives a single team-level score but doesn't attribute blame.

**Research**:
- [Agent Lightning](https://arxiv.org/abs/2508.03680) (2025) — RL for any agent framework via per-agent credit assignment. Shows how to decompose team rewards into individual agent contributions.
- [AgentRewardBench](https://arxiv.org/abs/2504.08942) (2025) — Benchmark showing no single LLM judge works for all agent tasks. Motivates multi-metric evaluation.

**Implementation path**: Add a `CreditAssignmentScorer` that analyses per-agent step sequences and attributes issues to specific agents. Could use step-level LLM judgements or heuristic turn analysis.

### 2. Reflective Prompt Evolution

**Problem**: The current `TuningLoop` uses a simple edit-and-score cycle. More sophisticated approaches evolve prompts through self-reflection.

**Research**:
- [GEPA](https://arxiv.org/pdf/2507.19457) (2025) — Reflective prompt evolution outperforms GRPO by 19%. Uses a genetic algorithm with LLM-driven mutation and crossover on prompt populations.
- [Multi-Module GRPO](https://arxiv.org/pdf/2508.04660) (2025) — Combines prompt optimization with GRPO weight updates. Shows synergies between prompt and weight tuning.

**Implementation path**: Extend `TuningLoop` with population-based strategies. Add a `PopulationEditor` protocol that maintains a pool of prompts and applies crossover/mutation between iterations.

### 3. Process Reward Models (PRMs)

**Problem**: Current scoring evaluates the final output. PRMs score each intermediate reasoning step, enabling more fine-grained RL training.

**Research**:
- [RLHF Book](https://rlhfbook.com/) — Comprehensive treatment of RLHF pipelines, including PRM vs ORM (Outcome Reward Models) tradeoffs.
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (OpenAI, 2023) — Foundational PRM paper showing step-level supervision improves mathematical reasoning.

**Implementation path**: The `Step`-level trajectory data is already captured. Add a `ProcessRewardScorer` that assigns per-step rewards based on tool correctness, reasoning quality, or LLM confidence. Feed per-step rewards to TRL via a custom `StepRewardBridge`.

### 4. Tool Use Evaluation Benchmarks

**Problem**: How well does an agent choose and use tools? Current `IssueDetectorScorer` checks failure rates but not strategic tool selection.

**Research**:
- [ToolBench](https://arxiv.org/abs/2305.16504) (2023) — Large-scale benchmark for tool use. Evaluates tool selection accuracy, argument correctness, and multi-step planning.
- [Gorilla](https://arxiv.org/abs/2305.15334) (2023) — LLM tool-use evaluation. Measures hallucination in API calls.

**Implementation path**: Add a `ToolStrategyScorer` that evaluates: (1) Did the agent use the right tools? (2) Were arguments well-formed? (3) Were unnecessary tools called? This needs a ground-truth tool plan or heuristic baselines.

### 5. RAG-Specific Evaluation

**Problem**: When agents use retrieval tools, evaluation should measure retrieval quality alongside generation quality.

**Research**:
- [Ragas](https://docs.ragas.io/) — Reference-free RAG evaluation (faithfulness, answer relevancy, context precision). Already wrapped as a scorer.
- [ARES](https://arxiv.org/abs/2311.09476) (2023) — Automated RAG Evaluation System using LLM judges with confidence calibration.

**Implementation path**: The `RagasScorer` wrapper exists but needs real integration testing. Extend the `Episode` model to capture retrieval context (retrieved documents, embeddings) as step metadata. Build a `RetrievalQualityScorer` that measures precision/recall of retrieved chunks.

### 6. Observability Integration

**Problem**: Production agent systems need continuous monitoring, not just offline evaluation.

**Research / Tools**:
- [Langfuse](https://langfuse.com/) — Open-source LLM observability platform with OpenTelemetry-compatible tracing.
- [TruLens](https://www.trulens.org/) — Evaluation and tracking for LLM applications.
- [OpenTelemetry Semantic Conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Standardized attributes for LLM traces.

**Implementation path**: Add an `OpenTelemetryAdapter` that reads OTLP traces with GenAI semantic conventions. This would allow evaluating any LLM application instrumented with OpenTelemetry, regardless of framework.

### 7. Safety and Alignment Evaluation

**Problem**: Agents may produce harmful, biased, or misaligned outputs that domain-specific scorers don't catch.

**Research / Tools**:
- [DeepEval](https://docs.confident-ai.com/) — 50+ metrics including toxicity, bias, hallucination. Already wrapped as a scorer.
- [LlamaGuard](https://arxiv.org/abs/2312.06674) (Meta, 2023) — Safety classifier for LLM inputs/outputs.
- [Constitutional AI](https://arxiv.org/abs/2212.08073) (Anthropic, 2022) — Self-supervised alignment through critique and revision.

**Implementation path**: Add a `SafetyScorer` that wraps LlamaGuard or similar classifiers. Integrate with DeepEval's toxicity and bias metrics. Use as a hard constraint in `CompositeReward` (zero reward if safety fails).

### 8. Automated Test Case Generation

**Problem**: Evaluating agents requires good test cases. Manually writing them is slow and biased.

**Research**:
- [GAIA](https://arxiv.org/abs/2311.12983) (Meta, 2023) — Benchmark for General AI Assistants with naturally difficult questions.
- [AgentBench](https://arxiv.org/abs/2308.03688) (2023) — Multi-environment benchmark for LLM agents.

**Implementation path**: Add a `TestCaseGenerator` that uses an LLM to generate diverse, challenging test cases for a given domain. Integrate with `TuningLoop` so the test set evolves alongside the prompt.

### Research Reading List (Priority Order)

1. **Agent Lightning** (2508.03680) — Most relevant for multi-agent RL credit assignment
2. **GEPA** (2507.19457) — Prompt evolution, directly applicable to TuningLoop
3. **Multi-Module GRPO** (2508.04660) — Combining prompt + weight optimization
4. **AgentRewardBench** (2504.08942) — Why multi-metric evaluation matters
5. **RLHF Book** (rlhfbook.com) — Comprehensive RLHF pipeline reference
6. **Let's Verify Step by Step** (2305.20050) — PRM foundations
7. **ARES** (2311.09476) — RAG evaluation with confidence calibration
8. **LlamaGuard** (2312.06674) — Safety evaluation for agents

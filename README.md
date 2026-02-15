# agent-eval

Framework-agnostic LLM/Agent evaluator with RL support.

```
Adapters --> Episode/Step --> Scorers --> ScoreVector --> Rewards --> RL Training
```

Converts traces from any multi-agent framework (AutoGen, LangGraph) into a canonical trajectory format, runs pluggable scorers, and produces multi-dimensional evaluation results. Scores can feed human-readable reports, RL reward functions, or prompt optimization loops.

## Quick Start

```bash
# No installation needed — run from the repo
cd evallab
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

All 432 tests run without installing the package:

```bash
cd evallab

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

## Choosing Scorers for Your Use Case

The scorers you pass to `EvalPipeline` determine what gets evaluated. Each scorer checks for different things, and you should pick them based on **what your agent actually does**.

Ask these questions about your agent:

| Question | If yes, add this scorer |
|----------|----------------------|
| Does it retrieve data via tools and report numbers? | `NumericConsistencyScorer` — catches fabricated numbers |
| Does it use tools at all? | `IssueDetectorScorer` — checks tool failures, coordination, errors |
| Is it a multi-agent system? | `IssueDetectorScorer` — detects stalls, agent imbalance, delegation issues |
| Do I need subjective quality assessment? | `LLMJudgeScorer` — evaluates coherence, relevance, or custom criteria |
| Does it do RAG (retrieval + generation)? | `DeepEvalScorer` or `RagasScorer` — faithfulness, context quality |
| Do I just need a quick pass/fail? | `IssueDetectorScorer` alone is enough |

**Always include `IssueDetectorScorer`** — it's the general-purpose quality gate that catches errors, missing answers, tool failures, and coordination problems. The other scorers add domain-specific checks on top.

**How they combine:** Each scorer produces its own `ScoreDimension` values and `Issue` objects. The pipeline collects all of them, feeds the issues into `RuleBasedScorer` (automatically) to produce a 0-100 score, and returns everything in a `ScoreVector`. More scorers = more thorough evaluation, but also slower.

**Examples by agent type:**

```python
# Chat agent (no tools, just conversation)
# -> Only need general issue detection
scorers = [IssueDetectorScorer()]

# Data analysis agent (queries databases, reports numbers)
# -> Need numeric check + general issues
scorers = [NumericConsistencyScorer(), IssueDetectorScorer()]

# Research agent (searches web, synthesizes information)
# -> Need general issues + LLM judge for answer quality
scorers = [IssueDetectorScorer(), LLMJudgeScorer(llm_fn=call_llm)]

# RAG agent (retrieves documents, answers based on them)
# -> Need faithfulness check + general issues
scorers = [IssueDetectorScorer(), DeepEvalScorer(metric_name="FaithfulnessMetric")]

# Production multi-agent system (full evaluation)
# -> Everything: numbers, rules, subjective quality
scorers = [
    NumericConsistencyScorer(),
    IssueDetectorScorer(),
    LLMJudgeScorer(
        llm_fn=call_llm,
        dimensions={
            "accuracy": "How factually accurate is the response?",
            "helpfulness": "How helpful is the response to the user?",
        },
    ),
]
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

# NumericConsistencyScorer: these agents retrieve financial data via tools,
#   so we need to verify reported numbers match tool outputs.
# IssueDetectorScorer: multi-agent system, so we also check for coordination
#   problems, tool failures, and answer quality.
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

# Single agent, no numerical reporting -> IssueDetectorScorer alone is enough.
# Add NumericConsistencyScorer if this agent retrieves and reports numbers.
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

# Use in a pipeline — combine rule-based + subjective evaluation.
# The LLM judge adds nuanced quality assessment that rules can't catch
# (e.g. "was the tone appropriate?" or "was the reasoning sound?").
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

## Evaluation Techniques Guide

This section explains every evaluation technique available in agent-eval, how each one works internally, and when to use it. Use this to pick the right combination of scorers for your use case.

### Quick Decision Guide

| I want to... | Use this | Dependencies |
|--------------|----------|-------------|
| Catch hallucinated numbers | `NumericConsistencyScorer` | None |
| Run a general quality check | `IssueDetectorScorer` | None |
| Get a 0-100 score with a letter grade | `RuleBasedScorer` (used automatically by the pipeline) | None |
| Evaluate subjective quality (coherence, helpfulness) | `LLMJudgeScorer` | Any LLM API |
| Use DeepEval metrics (faithfulness, toxicity, bias) | `DeepEvalScorer` | `deepeval` |
| Evaluate RAG retrieval quality | `RagasScorer` | `ragas` |
| Validate that my reward function ranks outputs correctly | PPE Benchmark Suite | None |

### Scorer Details

#### 1. NumericConsistencyScorer (`numeric`)

**What it does:** Detects fabricated or hallucinated numbers by cross-referencing numbers in the agent's final answer against numbers that actually appeared in tool results.

**How it works:**
1. Extracts all numbers from `episode.final_answer` using regex patterns that handle currencies (`$5.5B`), comma-separated numbers (`283,399,382.94`), and M/B/K notation (`283M`).
2. Extracts all numbers from tool call results (recursively traverses nested dicts/lists).
3. For each answer number above `min_value` (default `1.0`), checks if any tool number is within `tolerance` (default 5% relative error).
4. Numbers in the answer that have no match in tool results are flagged as **CRITICAL** "Data Fabrication" issues.
5. Score = fraction of answer numbers that matched tool results.

**When to use:**
- Agents that retrieve numerical data via tools and report it in answers (financial data, statistics, measurements).
- Catching when an LLM invents numbers not present in tool outputs.
- **Not useful** for agents that do pure text generation or don't use tools.

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tolerance` | `0.05` | Acceptable relative error (5%). Increase for approximate answers. |
| `min_value` | `1.0` | Skip numbers below this (avoids flagging `0`, `1`, etc.) |

**Example:**
```python
from agent_eval.scorers.numeric import NumericConsistencyScorer

# Strict: 1% tolerance
scorer = NumericConsistencyScorer(tolerance=0.01)

# Lenient: 10% tolerance, ignore small numbers
scorer = NumericConsistencyScorer(tolerance=0.10, min_value=10.0)

dims = scorer.score(episode)         # [ScoreDimension(name="numeric_accuracy", value=0.8)]
issues = scorer.detect_issues(episode)  # [Issue(severity=CRITICAL, category="Data Fabrication", ...)]
```

---

#### 2. IssueDetectorScorer (`issue_detector`)

**What it does:** Comprehensive rule-based quality gate that scans agent trajectories for problems across 6 categories. This is the primary "health check" scorer.

**How it works — 6 detection passes:**

| Category | What it checks | Severity |
|----------|---------------|----------|
| **Errors** | Exceptions, tracebacks, tool execution errors, API/connection/timeout errors, warnings in raw content | CRITICAL / ERROR / WARNING |
| **Answer Quality** | Missing final answer; answer shorter than 100 characters | CRITICAL / WARNING |
| **Agent Coordination** | Repeated tool call patterns (stalls); agent turn imbalance (max > 5x min); only 1 active agent | ERROR / WARNING |
| **Tool Usage** | No tools called (likely hallucination); low tool diversity; high failure rate | CRITICAL / ERROR / WARNING |
| **Efficiency** | Execution time exceeds limit; too many LLM calls | WARNING |
| **Data Accuracy** | "No data returned" messages; fact-check steps with FAIL verdict | WARNING / ERROR |

The score is computed as: `1.0 - (penalty_sum / 100)` where CRITICAL=25pts, ERROR=10pts, WARNING=5pts.

**When to use:**
- As the **default scorer in every pipeline**. It catches the most common failure modes.
- Multi-agent systems where coordination problems matter.
- Any agent that uses tools.

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_execution_seconds` | `300` | Flag runs longer than this |
| `max_llm_calls` | `30` | Flag runs with excessive LLM calls |
| `tool_failure_rate_threshold` | `0.3` | Flag when >30% of tool calls fail |

**Example:**
```python
from agent_eval.scorers.rules import IssueDetectorScorer

scorer = IssueDetectorScorer(
    max_execution_seconds=120,   # Stricter: 2 min limit
    max_llm_calls=20,            # Stricter: 20 LLM calls max
    tool_failure_rate_threshold=0.2,  # Flag at 20% failure rate
)

issues = scorer.detect_issues(episode)
for issue in issues:
    print(f"[{issue.severity.name}] {issue.category}: {issue.description}")
```

---

#### 3. RuleBasedScorer (`rule_based`)

**What it does:** Produces a 0-100 overall quality score with a letter grade. Uses a deduction-and-bonus formula.

**How it works:**
- Start at **100 points**
- **Deductions:** -25 per CRITICAL, -10 per ERROR, -5 per WARNING
- **Bonuses:** +5 for detailed answers (>500 chars), +3 for diverse tool use (>=3 tools), +2 for all tools succeeding
- Clamp to [0, 100]
- Grade: A (>=90), B (>=80), C (>=70), D (>=60), F (<60)

**When to use:**
- You typically don't call this directly. The `EvalPipeline` uses it internally to aggregate issues from all other scorers into a single score.
- Use it directly when you have issues from an `IssueDetectorScorer` and want a final grade.

**Example:**
```python
from agent_eval.scorers.rules import RuleBasedScorer, IssueDetectorScorer

detector = IssueDetectorScorer()
grader = RuleBasedScorer()

issues = detector.detect_issues(episode)
dims = grader.score_with_issues(episode, issues)
# dims[0] = ScoreDimension(name="overall_score", value=85, max_value=100)

print(grader.get_grade(dims[0].value))  # "B"
```

---

#### 4. LLMJudgeScorer (`llm_judge`)

**What it does:** Uses any LLM as an evaluator to score agent episodes on configurable quality dimensions. This is the most flexible scorer — it can assess anything you can describe in natural language.

**How it works:**
1. Builds a readable transcript from episode steps (messages, tool calls with results, fact-checks).
2. Sends the transcript to your LLM with a structured prompt asking it to score each dimension 0.0-1.0 with reasoning.
3. Parses the JSON response (handles markdown code fences, raw JSON, brace extraction).
4. Returns ScoreDimensions with [0, 1] scores per dimension.

**Default dimensions** (used when you don't specify custom ones):

| Dimension | What it assesses |
|-----------|-----------------|
| `relevance` | How relevant is the response to the user's question? |
| `groundedness` | Are claims grounded in tool results? Avoids fabrication? |
| `completeness` | Does the response thoroughly address all aspects? |
| `coherence` | Is the response well-organized and logically structured? |
| `tool_usage` | Were appropriate tools used effectively with correct parameters? |

**When to use:**
- When you need **subjective evaluation** beyond what rules can catch.
- Custom evaluation criteria specific to your domain.
- Evaluating reasoning quality, tone, formatting, or any nuanced aspect.
- **Trade-off:** Slower and costs money (LLM API calls), but much more flexible than rule-based scorers.

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm_fn` | *required* | `Callable[[str, str], str]` — any function that takes (system_prompt, user_prompt) and returns text |
| `dimensions` | 5 defaults above | `dict[str, str]` — dimension name to description |
| `batch_dimensions` | `True` | Score all dimensions in one LLM call (faster) vs. one call per dimension (more reliable) |
| `max_transcript_chars` | `8000` | Truncate long transcripts to fit context windows |
| `low_score_threshold` | `0.4` | Scores below this produce WARNING issues |

**Example with custom domain-specific dimensions:**
```python
from agent_eval.scorers.llm_judge import LLMJudgeScorer

judge = LLMJudgeScorer(
    llm_fn=call_llm,  # Your LLM wrapper function
    dimensions={
        "regulatory_compliance": "Does the response follow financial regulations?",
        "risk_disclosure": "Are investment risks properly disclosed?",
        "data_citation": "Does the response cite specific data sources?",
    },
    batch_dimensions=True,
    max_transcript_chars=12000,
)
```

---

#### 5. DeepEvalScorer (`deepeval`)

**What it does:** Wraps any [DeepEval](https://docs.confident-ai.com/) metric into the agent-eval scorer protocol. DeepEval provides 50+ LLM-powered evaluation metrics.

**How it works:**
1. Dynamically imports the specified DeepEval metric class.
2. Converts the Episode to a DeepEval `LLMTestCase` with `input` (task description), `actual_output` (final answer), and `retrieval_context` (tool results).
3. Runs `metric.measure(test_case)` and returns the score as a ScoreDimension.

**Useful DeepEval metrics:**

| Metric | What it measures |
|--------|-----------------|
| `AnswerRelevancyMetric` | Is the answer relevant to the question? |
| `FaithfulnessMetric` | Is the answer faithful to the retrieved context? |
| `HallucinationMetric` | Does the answer contain hallucinated content? |
| `ToxicityMetric` | Does the answer contain toxic content? |
| `BiasMetric` | Does the answer exhibit bias? |
| `GEval` | Custom LLM evaluation with your own criteria |

**When to use:**
- When you already use DeepEval and want to integrate its metrics.
- For RAG evaluation (faithfulness, relevancy).
- For safety checks (toxicity, bias).

**Example:**
```python
from agent_eval.scorers.deepeval import DeepEvalScorer

scorer = DeepEvalScorer(
    metric_name="FaithfulnessMetric",
    metric_kwargs={"threshold": 0.7, "model": "gpt-4o-mini"},
)

# Requires: pip install agent-eval[deepeval]
```

---

#### 6. RagasScorer (`ragas`)

**What it does:** Wraps [Ragas](https://docs.ragas.io/) metrics for evaluating RAG (Retrieval-Augmented Generation) quality.

**How it works:**
1. Converts the Episode to a HuggingFace Dataset with `question`, `answer`, and `contexts` fields.
2. Runs `ragas.evaluate()` with the specified metric.
3. Returns the score as a ScoreDimension.

**Available metrics:**

| Metric | What it measures |
|--------|-----------------|
| `faithfulness` | Are claims in the answer supported by the context? |
| `answer_relevancy` | Is the answer relevant to the question? |
| `context_precision` | Is the retrieved context relevant? (Less noise) |
| `context_recall` | Does the context contain all needed information? |
| `answer_similarity` | Semantic similarity to a reference answer |
| `answer_correctness` | Factual correctness of the answer |

**When to use:**
- Agents that use retrieval tools (search, database queries, document lookup).
- When you need to evaluate both retrieval quality AND answer quality.
- **Not useful** for agents that don't do retrieval.

**Example:**
```python
from agent_eval.scorers.ragas import RagasScorer

scorer = RagasScorer(metric_name="faithfulness")

# Requires: pip install agent-eval[ragas]
```

---

### Reward Functions (for RL Training)

Reward functions convert multi-dimensional ScoreVectors into scalar rewards suitable for reinforcement learning.

#### WeightedSumReward

Weighted average of normalized dimension values. Use when all dimensions matter but some matter more.

```python
from agent_eval.rewards import WeightedSumReward

# Accuracy matters 3x more than speed
reward = WeightedSumReward(weights={"accuracy": 3.0, "speed": 1.0})
scalar = reward.compute(score_vector)  # Float in [0, 1]
```

#### DeductionReward

Starts at 1.0, deducts per issue severity. Use when you care primarily about avoiding mistakes.

```python
from agent_eval.rewards import DeductionReward

reward = DeductionReward(
    critical_penalty=0.25,  # -0.25 per CRITICAL
    error_penalty=0.10,     # -0.10 per ERROR
    warning_penalty=0.05,   # -0.05 per WARNING
)
```

#### CompositeReward

Combines multiple reward functions with weights. Use for multi-objective optimization.

```python
from agent_eval.rewards import WeightedSumReward, DeductionReward, CompositeReward

composite = CompositeReward(
    components=[
        (WeightedSumReward(weights={"accuracy": 2.0}), 3.0),  # 75% weight
        (DeductionReward(), 1.0),                               # 25% weight
    ],
    normalize=True,
)

reward = composite.compute(score_vector)
breakdown = composite.compute_breakdown(score_vector)  # {"WeightedSumReward": 0.8, "DeductionReward": 0.6}
```

---

### PPE Benchmark Suite (Experimental)

**Purpose:** Validates whether your reward function correctly ranks good outputs above bad ones. Based on [Preference Proxy Evaluations (arXiv:2410.14872)](https://arxiv.org/abs/2410.14872).

**When to use:** Before deploying a reward function for RL training, benchmark it to make sure it actually distinguishes good outputs from bad ones.

**6 metrics available:**

| Metric | Type | Measures | Ideal |
|--------|------|----------|-------|
| Pairwise Accuracy | Pair-based | % of pairs ranked correctly | 1.0 |
| Separability | Pair-based | % of pairs with reward gap > margin | 1.0 |
| Brier Score | Pair-based | Calibrated prediction error | 0.0 |
| Best-of-K | Sample-based | Quality of reward's top pick vs. actual best | 1.0 |
| Spearman Correlation | Sample-based | Rank correlation with ground truth | 1.0 |
| Kendall Tau | Sample-based | Rank correlation (robust to outliers) | 1.0 |

**Example:**
```python
from agent_eval.experimental.ppe.synthetic import SyntheticDatasetBuilder
from agent_eval.experimental.ppe.runner import BenchmarkRunner
from agent_eval.rewards import WeightedSumReward, DeductionReward

# Build a benchmark dataset from your evaluation results
builder = SyntheticDatasetBuilder(score_vectors=my_score_vectors)
dataset = builder.build_dataset(n_pairs=100, k=5, n_samples=50)

# Compare reward functions
runner = BenchmarkRunner(dataset)
results = runner.run_comparison([
    (WeightedSumReward(), "Weighted Sum"),
    (DeductionReward(), "Deduction"),
])

# See which reward function ranks outputs more accurately
from agent_eval.experimental.ppe.report import comparison_to_text
print(comparison_to_text(results))
```

---

### Recommended Scorer Combinations

**Minimal (fast, zero dependencies):**
```python
scorers = [IssueDetectorScorer()]
```
Good for: CI/CD gates, quick quality checks, development iteration.

**Standard (catches most issues):**
```python
scorers = [NumericConsistencyScorer(), IssueDetectorScorer()]
```
Good for: Agents that use tools and report data. This is the default used by the CLI.

**Comprehensive (adds subjective evaluation):**
```python
scorers = [NumericConsistencyScorer(), IssueDetectorScorer(), LLMJudgeScorer(llm_fn=call_llm)]
```
Good for: Production evaluation, detailed quality reports, when you need nuanced assessment.

**RAG-focused:**
```python
scorers = [
    IssueDetectorScorer(),
    LLMJudgeScorer(llm_fn=call_llm, dimensions={"groundedness": "...", "completeness": "..."}),
    DeepEvalScorer(metric_name="FaithfulnessMetric"),  # or RagasScorer(metric_name="faithfulness")
]
```
Good for: Agents that retrieve and synthesize information from documents or databases.

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

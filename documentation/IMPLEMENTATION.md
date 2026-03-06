# agent_eval — Implementation Tracker

## Overview

Framework-agnostic LLM/Agent evaluator with RL support.
Architecture: `Adapters → Episode/Step → Scorers → ScoreVector → Rewards → RL`

---

## Progress Summary

| Phase | Description | Status | Tests | Files |
|-------|-------------|--------|-------|-------|
| **1** | Core + AutoGen Adapter + Deterministic Scorers | COMPLETE | 131 | 27 |
| **2** | LLM Judge + DeepEval + CLI | COMPLETE | 46 | 12 |
| **3** | RL Bridge (GRPO Priority) | COMPLETE | 45 | 10 |
| **4** | LangGraph Adapter + DSPy + HTML Reports | COMPLETE | 53 | 8 |

| **5a** | Intrinsic Reasoning & Orchestration Scorers | COMPLETE | 42 | 4 |

**Total: 526 tests passing** | Run with: `PYTHONPATH=src pytest tests/ -v`

---

## Phase 1: Core + AutoGen Adapter + Deterministic Scorers

### Status: COMPLETE (131 tests passing)

### Files Created

#### Package Structure
| File | Status | Description |
|------|--------|-------------|
| `pyproject.toml` | Done | PEP 621 config, hatchling build, optional deps (`dev`, `rl`, `deepeval`, `ragas`, `all`) |
| `src/agent_eval/__init__.py` | Done | Package root, exports `__version__` |
| `src/agent_eval/py.typed` | Done | PEP 561 marker for type checkers |

#### Core (`src/agent_eval/core/`)
| File | Status | Description |
|------|--------|-------------|
| `__init__.py` | Done | Re-exports all core types |
| `models.py` | Done | `Episode`, `Step`, `StepKind` dataclasses. Episode has `agents`, `duration_seconds`, `steps_by_kind`, `steps_by_agent` helpers |
| `score.py` | Done | `Severity` enum, `Issue`, `ScoreDimension` (with `normalized` property), `ScoreVector` (with `overall`, `dimension_by_name`, `to_dict`) |
| `protocols.py` | Done | `TraceAdapter`, `Scorer`, `RewardFunction` — PEP 544 runtime-checkable protocols |
| `exceptions.py` | Done | `AgentEvalError`, `AdapterError`, `ScorerError`, `PipelineError` |

#### AutoGen Adapter (`src/agent_eval/adapters/autogen/`)
| File | Status | Ported From |
|------|--------|-------------|
| `__init__.py` | Done | Exports `AutoGenAdapter` |
| `event_parser.py` | Done | `log_evaluator.py:_extract_json_events` — brace-counting JSON state machine |
| `tool_failure.py` | Done | `log_evaluator.py:_is_tool_call_failed` — two-tier failure detection (dict keys + string patterns) |
| `adapter.py` | Done | `log_evaluator.py:_extract_metrics` — converts events to Episode/Steps, resolves agent names, extracts timestamps and final answer |

#### Scorers — Numeric (`src/agent_eval/scorers/numeric/`)
| File | Status | Ported From |
|------|--------|-------------|
| `__init__.py` | Done | Exports `NumericConsistencyScorer` |
| `extraction.py` | Done | `numeric_validator.py` — `extract_numbers_from_text`, `extract_numbers_from_tool_results`, `extract_answer_block` |
| `consistency.py` | Done | `numeric_validator.py:validate_numeric_consistency` — wrapped as Scorer protocol with `score()` and `detect_issues()` |

#### Scorers — Rules (`src/agent_eval/scorers/rules/`)
| File | Status | Ported From |
|------|--------|-------------|
| `__init__.py` | Done | Exports `RuleBasedScorer`, `IssueDetectorScorer` |
| `issue_detector.py` | Done | `log_evaluator.py:_detect_issues` — 6 categories: Errors, Answer Quality, Agent Coordination, Tool Usage, Efficiency, Data Accuracy |
| `deduction.py` | Done | `log_evaluator.py:_calculate_score` + `_get_grade` — base-100 deduction scoring with bonuses |

#### Rewards (`src/agent_eval/rewards/`)
| File | Status | Description |
|------|--------|-------------|
| `__init__.py` | Done | Exports `WeightedSumReward`, `DeductionReward` |
| `aggregators.py` | Done | `WeightedSumReward` (configurable dimension weights), `DeductionReward` (severity-based, matches original scoring formula) |

#### Pipeline (`src/agent_eval/pipeline/`)
| File | Status | Description |
|------|--------|-------------|
| `__init__.py` | Done | Exports `EvalPipeline` |
| `runner.py` | Done | `EvalPipeline` orchestrates Adapter → Scorers → ScoreVector → EvalResult. `EvalResult` dataclass with episode, score_vector, grade, summary |

#### Reporting (`src/agent_eval/reporting/`)
| File | Status | Description |
|------|--------|-------------|
| `__init__.py` | Done | — |
| `text.py` | Done | `format_report()` and `format_comparison_report()` — terminal text output |
| `json_report.py` | Done | `to_json()` and `to_json_batch()` — JSON serialization |

#### Stub Packages (empty `__init__.py` for future phases)
- `src/agent_eval/adapters/langgraph/`
- `src/agent_eval/scorers/llm_judge/`
- `src/agent_eval/scorers/deepeval/`
- `src/agent_eval/scorers/ragas/`
- `src/agent_eval/rl/`
- `src/agent_eval/cli/`
- `src/agent_eval/cli/commands/`
- `src/agent_eval/utils/`

#### Tests (`tests/`)
| File | Status | Covers |
|------|--------|--------|
| `conftest.py` | Done | Shared fixtures: `sample_log_path`, `sample_log_content`, `minimal_episode`, `multi_agent_episode` |
| `fixtures/sample_log/event.txt` | Done | Full sample AutoGen log with Messages, ToolCalls, LLMCalls, FactCheckResult, and ANSWER block |
| `fixtures/sample_langgraph_trace.json` | Done | Placeholder LangGraph trace |
| `core/test_models.py` | Done | StepKind, Step, Episode (properties, filtering) |
| `core/test_score.py` | Done | Severity, Issue, ScoreDimension (normalized), ScoreVector (overall, to_dict) |
| `adapters/autogen/test_event_parser.py` | Done | JSON extraction: single, multiple, nested, malformed, filtering, fact-check events |
| `adapters/autogen/test_tool_failure.py` | Done | Tier 1 (dict keys, status codes) + Tier 2 (string patterns) + false positive guards |
| `adapters/autogen/test_adapter.py` | Done | Load from file/dir, step kinds, tool detection, agent resolution, timestamps, final answer, batch loading |
| `scorers/numeric/test_extraction.py` | Done | Number extraction (decimals, M/B/K, currencies), tool result extraction, answer block extraction |
| `scorers/numeric/test_consistency.py` | Done | Fabrication detection, tolerance, score dimensions, edge cases |
| `scorers/rules/test_deduction.py` | Done | Perfect score, critical/mixed deductions, clamping, bonuses, grade mapping |
| `scorers/rules/test_issue_detector.py` | Done | All 6 issue categories: answer quality, tool usage, coordination, efficiency, data accuracy, error detection |
| `rewards/test_aggregators.py` | Done | WeightedSumReward, DeductionReward with various issue combinations |
| `pipeline/test_runner.py` | Done | Full pipeline evaluation, batch, dimension presence |

---

### Remaining Phase 1 Work
- [x] Run all tests — **131 passed, 0 failed** (2.14s, no install needed: `PYTHONPATH=src pytest tests/ -v`)
- [x] Fixes applied during test run:
  - `_resolve_log_path` now accepts any file, not just `event.txt`
  - Number extraction regex extended for plain large decimals (`283399382.94`)
  - `pyproject.toml` target lowered to Python 3.10 (matching project venv)
  - Fixture moved to `fixtures/sample_log/event.txt` to match real directory layout

---

## Porting Map (Existing Code → New Location)

| Existing File | What Was Ported | New Location |
|---|---|---|
| `log_evaluator.py:_extract_json_events` | JSON event parsing with brace-counting state machine | `adapters/autogen/event_parser.py` |
| `log_evaluator.py:_is_tool_call_failed` | Two-tier tool failure detection | `adapters/autogen/tool_failure.py` |
| `log_evaluator.py:_extract_metrics` + event-to-agent mapping | Event → Step conversion, agent name resolution | `adapters/autogen/adapter.py` |
| `log_evaluator.py:_calculate_score` | Deduction-based scoring (100pt base) | `scorers/rules/deduction.py` |
| `log_evaluator.py:_detect_issues` | Rule-based issue detection (6 categories) | `scorers/rules/issue_detector.py` |
| `numeric_validator.py` | Number extraction + consistency checking | `scorers/numeric/extraction.py` + `consistency.py` |
| `test_log_evaluator.py` (1,357 lines) | All test cases adapted to new interfaces | `tests/` (distributed across modules) |

---

## Phase 2: LLM Judge + DeepEval + CLI

### Status: COMPLETE (46 additional tests, 177 total passing)

### Files Created

#### LLM Judge Scorer (`src/agent_eval/scorers/llm_judge/`)
| File | Status | Description |
|------|--------|-------------|
| `__init__.py` | Done | Exports `LLMJudgeScorer` |
| `prompts.py` | Done | `JUDGE_SYSTEM_PROMPT`, dimension templates, `DEFAULT_DIMENSIONS` (relevance, groundedness, completeness, coherence, tool_usage), `build_transcript`, `format_single_prompt`, `format_multi_prompt` |
| `judge.py` | Done | `LLMJudgeScorer` — model-agnostic via `Callable[[str, str], str]`, supports batch and individual dimension modes, JSON extraction from markdown fences, fallback on parse failure |

#### Optional Metric Wrappers
| File | Status | Description |
|------|--------|-------------|
| `scorers/deepeval/wrapper.py` | Done | `DeepEvalScorer` wrapping any DeepEval metric, Episode → `LLMTestCase` conversion, import guard with `_DEEPEVAL_AVAILABLE` |
| `scorers/ragas/wrapper.py` | Done | `RagasScorer` wrapping Ragas metrics, Episode → HuggingFace Dataset, import guard with `_RAGAS_AVAILABLE` |

#### Scorer Registry (`src/agent_eval/scorers/`)
| File | Status | Description |
|------|--------|-------------|
| `registry.py` | Done | `ScorerRegistry` with `register()`, `get()`, `list_scorers()`, lazy `_ensure_builtins()` auto-registration, `default_registry` singleton |

#### Pipeline Extensions (`src/agent_eval/pipeline/`)
| File | Status | Description |
|------|--------|-------------|
| `batch.py` | Done | `BatchResult` with scores, avg/best/worst, grade distribution, total_issues; `evaluate_batch()` and `evaluate_paths()` |
| `comparison.py` | Done | `DimensionDelta`, `ComparisonResult` with improvements/regressions; `compare_results()` and `compare_batch()` |

#### CLI (`src/agent_eval/cli/`) — argparse-based, zero dependencies
| File | Status | Description |
|------|--------|-------------|
| `main.py` | Done | `build_parser()` and `app()` entry point with subcommands |
| `commands/__init__.py` | Done | — |
| `commands/evaluate.py` | Done | `agent-eval evaluate <paths>` with `--framework`, `--agents`, `--format`, `--output`, `--brief`, `--scorers` |
| `commands/compare.py` | Done | `agent-eval compare <path_a> <path_b>` with `--label-a`, `--label-b`, `--format`, text and JSON output |

#### Phase 2 Tests
| File | Status | Covers |
|------|--------|--------|
| `tests/scorers/llm_judge/test_judge.py` | Done | 12 tests — mock LLM functions, batch/individual modes, fallback, markdown JSON extraction |
| `tests/scorers/deepeval/test_wrapper.py` | Done | 2 tests — import guard when deepeval not installed, availability flag |
| `tests/scorers/test_registry.py` | Done | 8 tests — register, get, list, duplicate guard, overwrite, builtin instantiation |
| `tests/pipeline/test_batch.py` | Done | 6 tests — BatchResult empty/single/multiple, grade distribution, issue counts, to_dict |
| `tests/pipeline/test_comparison.py` | Done | 6 tests — DimensionDelta improved/regressed, compare_results, compare_batch |
| `tests/cli/test_evaluate_cmd.py` | Done | 12 tests — evaluate/compare CLI commands, text/JSON/brief output, file output |

### Key Design Decisions (Phase 2)
- **argparse instead of Typer** — zero external dependencies for CLI
- **LLM Judge uses `Callable[[str, str], str]`** — model-agnostic, any LLM provider works
- **DeepEval/Ragas wrappers** — graceful import guards, tests work without optional deps installed
- **Scorer Registry** — lazy auto-registration of 4 built-in scorers (numeric, issue_detector, rule_based, llm_judge)

---

## Phase 3: RL Bridge (GRPO Priority)

### Status: COMPLETE (45 additional tests, 222 total passing)

### Files Created

#### Rewards (`src/agent_eval/rewards/`)
| File | Status | Description |
|------|--------|-------------|
| `composite.py` | Done | `CompositeReward` — weighted combination of multiple RewardFunctions with optional normalization, `compute_breakdown()` for diagnostics |

#### RL Data Models (`src/agent_eval/rl/`)
| File | Status | Ported From |
|------|--------|-------------|
| `data_models.py` | Done | `prompt_tunning/data_models.py` — `TuningIteration` (generic, uses ScoreVector instead of hardcoded scope/groundedness/clarity), `TuningResult` with `reward_trajectory` and `converged` flag |

#### Tuning Loop (`src/agent_eval/rl/`)
| File | Status | Ported From |
|------|--------|-------------|
| `tuning_loop.py` | Done | `prompt_tunning/prompt_tuner.py` — `TuningLoop` with pluggable `TestRunner`, `Evaluator`, `Editor`, `SimilarityFn` protocols; `lambda_param` for similarity bonus; early stopping via `target_reward` |

#### TRL Bridge (`src/agent_eval/rl/`)
| File | Status | Description |
|------|--------|-------------|
| `trl_bridge.py` | Done | `GRPORewardBridge` — wraps scorers + reward function into TRL-compatible `(prompts, completions) -> list[float]`; `_default_episode_builder` for prompt/completion pairs; `as_trl_reward_fn()` for `GRPOTrainer` |

#### Reward Server (`src/agent_eval/rl/`)
| File | Status | Description |
|------|--------|-------------|
| `reward_server.py` | Done | FastAPI-based HTTP reward server (OpenRLHF pattern); `/reward` endpoint for batch reward computation; `/health` endpoint; `create_app()` and `run_server()` functions; import guard for FastAPI |

#### CLI Extension
| File | Status | Description |
|------|--------|-------------|
| `cli/commands/serve.py` | Done | `agent-eval serve` with `--host`, `--port`, `--scorers` options |

#### Utils (`src/agent_eval/utils/`)
| File | Status | Ported From |
|------|--------|-------------|
| `embeddings.py` | Done | `prompt_tunning/utils.py` — `cosine_similarity` (numpy-only, no Azure/LangChain dependency), `make_similarity_fn` factory for creating `SimilarityFn` from any embedding function |

#### Phase 3 Tests
| File | Status | Covers |
|------|--------|--------|
| `tests/rewards/test_composite.py` | Done | 7 tests — single/equal/unequal weights, no-normalize, empty raises, breakdown, three components |
| `tests/rl/test_data_models.py` | Done | 7 tests — TuningIteration creation/to_dict/defaults, TuningResult creation/history/to_dict/empty |
| `tests/rl/test_tuning_loop.py` | Done | 7 tests — basic run, early stopping, lambda similarity bonus, single iteration, history recorded, reward trajectory, no-target-no-convergence |
| `tests/rl/test_trl_bridge.py` | Done | 8 tests — default episode builder, single/batch rewards, mismatched lengths, as_trl_reward_fn, issue scorer, custom episode builder, multiple scorers |
| `tests/rl/test_reward_server.py` | Done | 5 tests — import guard flag, FastAPI create_app, /health endpoint, /reward endpoint (3 skip-or-run based on FastAPI availability) |
| `tests/cli/test_serve_cmd.py` | Done | 3 tests — parser defaults, custom host/port, custom scorers |
| `tests/utils/test_embeddings.py` | Done | 9 tests — numpy availability, identical/orthogonal/opposite/similar/zero/high-dim vectors, make_similarity_fn basic/different texts |

### Key Design Decisions (Phase 3)
- **CompositeReward** generalises the prompt_tuner's `reward = overall + lambda * similarity` into arbitrary weighted combinations
- **TuningLoop uses Protocol callbacks** — users supply TestRunner, Evaluator, Editor; no domain-specific hardcoding
- **GRPORewardBridge** converts (prompt, completion) pairs to Episodes, runs through scorers, and returns scalar rewards compatible with TRL's `GRPOTrainer(reward_funcs=[...])`
- **Reward server follows OpenRLHF pattern** — HTTP API for distributed RL, FastAPI as optional dependency
- **`cosine_similarity` accepts plain Python lists** — no mandatory numpy arrays in the signature; numpy used internally
- **`make_similarity_fn` factory** — wraps any `embed_fn(text) -> list[float]` into a `SimilarityFn` for TuningLoop

### Porting Map (Phase 3)
| Existing File | What Was Ported | New Location |
|---|---|---|
| `prompt_tunning/data_models.py:AggregateScores` | Replaced by `ScoreVector` (Phase 1) | `core/score.py` |
| `prompt_tunning/data_models.py:IterationRecord` | Generalized to use ScoreVector | `rl/data_models.py:TuningIteration` |
| `prompt_tunning/data_models.py:PromptTuningResult` | Added converged flag, reward_trajectory | `rl/data_models.py:TuningResult` |
| `prompt_tunning/prompt_tuner.py:PromptTuner` | Protocol callbacks replace type aliases | `rl/tuning_loop.py:TuningLoop` |
| `prompt_tunning/utils.py:cosine_similarity` | Framework-agnostic, plain Python lists | `utils/embeddings.py` |

---

## Phase 4: LangGraph Adapter + DSPy + HTML Reports

### Status: COMPLETE (53 additional tests, 275 total passing)

### Files Created

#### LangGraph Adapter (`src/agent_eval/adapters/langgraph/`)
| File | Status | Description |
|------|--------|-------------|
| `__init__.py` | Done | Exports `LangGraphAdapter`, checkpoint functions |
| `adapter.py` | Done | `LangGraphAdapter` — converts `astream_events` JSON traces to Episodes; maps `on_chain_start/end`, `on_chat_model_start/end`, `on_tool_start/end` to canonical StepKinds; extracts task, final answer, timestamps, token counts; supports `load_from_dict()` |
| `checkpoint.py` | Done | `load_checkpoints()`, `checkpoint_to_episode()`, `checkpoints_to_episodes()`, `latest_checkpoint_to_episode()` — reads LangGraph MemorySaver/SqliteSaver checkpoints; converts message-role history to Steps |

#### DSPy Bridge (`src/agent_eval/rl/`)
| File | Status | Description |
|------|--------|-------------|
| `dspy_bridge.py` | Done | `DSPyMetricBridge` — wraps scorers + reward function as DSPy metric function `(example, prediction, trace) -> float|bool`; configurable `input_field`/`output_field`; `threshold` mode for boolean metrics; `as_dspy_metric()` for MIPROv2 integration; import guard with `_DSPY_AVAILABLE` |

#### HTML Reports (`src/agent_eval/reporting/`)
| File | Status | Description |
|------|--------|-------------|
| `html.py` | Done | `format_html_report()` — self-contained HTML with inline CSS, score card with grade colors, metrics table, dimension progress bars, issue badges by severity, XSS-safe escaping; `format_html_batch()` — multi-run HTML comparison table sorted by score |

#### Phase 4 Tests
| File | Status | Covers |
|------|--------|--------|
| `tests/adapters/langgraph/test_adapter.py` | Done | 17 tests — init, load from fixture/dict/dir, step kinds, tool calls/results, LLM calls, token counts, timestamps, task/answer extraction, metadata, errors, tool failure detection |
| `tests/adapters/langgraph/test_checkpoint.py` | Done | 11 tests — load checkpoints, missing/invalid files, first/middle/final checkpoint episodes, task extraction, custom ID, batch conversion, latest checkpoint, empty raises |
| `tests/rl/test_dspy_bridge.py` | Done | 10 tests — episode builder, evaluate, call with dicts/objects, threshold true/false, custom fields, as_dspy_metric, custom episode builder, availability flag |
| `tests/reporting/test_html.py` | Done | 14 tests — HTML structure, score/grade display, framework, verbose/non-verbose metrics, dimensions, issues/no-issues, XSS escaping, severity colors, batch empty/single/multiple/sorted |

#### Updated Fixture
| File | Status | Description |
|------|--------|-------------|
| `tests/fixtures/sample_langgraph_trace.json` | Updated | Rich LangGraph trace with `astream_events` format (chain start/end, chat model start/end with usage, tool start/end) + 3 checkpoints with progressive message history |

### Key Design Decisions (Phase 4)
- **LangGraphAdapter supports both event-based and dict-based loading** — `load_episode(path)` for files, `load_from_dict(trace)` for programmatic use
- **Checkpoint reader is separate from adapter** — users can choose event-based reconstruction (more detail) or checkpoint-based (simpler, from MemorySaver)
- **DSPy bridge works without dspy installed** — import guard pattern, tests pass in all environments
- **HTML reports are self-contained** — no external CSS/JS, single-file output with inline styles
- **XSS protection** — all user content HTML-escaped in reports

---

## Post-Phase Work: AutoGen Logger + Examples

### Status: COMPLETE (332 total tests passing)

### Files Created

#### AutoGen Logger (`src/agent_eval/adapters/autogen/`)
| File | Status | Description |
|------|--------|-------------|
| `logger.py` | Done | `attach_logger()` — captures AutoGen `EVENT_LOGGER_NAME` events to JSONL files loadable by `AutoGenAdapter`. `_normalize()` handles dicts, Pydantic v1/v2, JSON strings. `LoggerHandle` with `.detach()` for clean teardown. |

#### Examples (`examples/`)
| File | Status | Description |
|------|--------|-------------|
| `try_autogen_eval.py` | Done | End-to-end example: parse log → run pipeline with NumericConsistencyScorer + IssueDetectorScorer → format report |

#### Updated Files
| File | Change |
|------|--------|
| `adapters/autogen/__init__.py` | Added exports: `LoggerHandle`, `attach_logger` |
| `.gitignore` | Added `TODO.md`, `temp` |

#### Tests
| File | Status | Covers |
|------|--------|--------|
| `tests/adapters/autogen/test_logger.py` | Done | 21 tests — `_normalize` (dict, pydantic v1/v2, JSON string, edge cases), `_JsonlEventHandler` (JSONL output, skip invalid, non-serializable fields, multi-line), `attach_logger` integration (file creation, detach, append mode, parent dirs, adapter compatibility, import guard) |

---

## Session: 2026-02-12 — dev

### Completed This Session
- Added `src/agent_eval/adapters/autogen/logger.py` — JSONL event logger with `attach_logger()` and `LoggerHandle`
- Added `tests/adapters/autogen/test_logger.py` — 21 tests covering normalization, handler, and integration
- Updated `src/agent_eval/adapters/autogen/__init__.py` — exported `LoggerHandle` and `attach_logger`
- Moved `try_autogen_eval.py` → `examples/try_autogen_eval.py`
- Updated `.gitignore` — added `TODO.md` and `temp`

### In Progress (uncommitted)
- None — all changes committed (ab30945)

### Remaining Steps
- [ ] Create additional example scripts in `examples/` (e.g., LangGraph eval, LLM Judge, RL tuning)
- [ ] Add JSONL format support to `AutoGenAdapter` (currently only event.txt brace-counting format)
- [ ] Document `attach_logger` usage in README or separate guide
- [x] Remove tracked `__pycache__/` files from git (root `.gitignore` already covers the pattern)

### Plan Accuracy Notes
- All 4 original phases are complete and accurate
- Logger is new work beyond the original plan scope — added as post-phase section

### Doc Sync
- Updated test count in Progress Summary: 275 → 332

---

## Experimental: PPE Reward Model Benchmarking

### Status: COMPLETE (87 additional tests, 432 total passing)

Based on arXiv:2410.14872 "How to Evaluate Reward Models for RLHF" — adapted to benchmark evallab's own RewardFunction implementations on synthetic preference data derived from ScoreVectors.

### Files Created

#### Package Structure (`src/agent_eval/experimental/ppe/`)
| File | Status | Description |
|------|--------|-------------|
| `__init__.py` | Done | Public API re-exports (20 symbols) |
| `models.py` | Done | `PreferencePair`, `BenchmarkSample`, `BenchmarkDataset`, `MetricResult`, `BenchmarkResult` dataclasses with validation and serialization |
| `metrics.py` | Done | 6 PPE metrics (pairwise_accuracy, best_of_k, spearman_correlation, kendall_tau, separability, brier_score) + 5 internal math helpers (_sigmoid, _rank, _pearson, _spearman_rho, _kendall_tau_b). Zero dependencies — all stdlib. |
| `runner.py` | Done | `BenchmarkRunner` — orchestrates metric evaluation, auto-selects applicable metrics based on dataset contents, configurable separability margin |
| `synthetic.py` | Done | `SyntheticDatasetBuilder` (seeded RNG, configurable ground truth, O(n^2) pair generation with quality gap filtering, random K-group sampling) + `perturb_score_vector` (Gaussian noise, clamped to valid range) |
| `report.py` | Done | `benchmark_to_text`, `comparison_to_text`, `benchmark_to_dict`, `comparison_to_dict` |

#### Tests (`tests/experimental/ppe/`)
| File | Status | Covers |
|------|--------|--------|
| `test_models.py` | Done | 14 tests — all dataclasses, validation, serialization, domain handling |
| `test_metrics.py` | Done | 32 tests — all math helpers (sigmoid, rank, spearman, kendall), all 6 metrics with perfect/inverted/tied/empty cases, per-domain breakdown |
| `test_runner.py` | Done | 7 tests — full/selective/pairs-only/samples-only metrics, comparison, auto-labeling, margin propagation |
| `test_synthetic.py` | Done | 15 tests — perturbation (clamping, determinism, issues), pairs (gap filter, ordering), samples (grouping, custom N), dataset builder, custom ground truth, seed reproducibility |
| `test_report.py` | Done | 8 tests — text/dict for single and comparison, rounding, empty handling, per-domain display |

### Key Design Decisions
- **Stdlib only** — all metrics implemented from scratch (Spearman, Kendall tau-b with tie corrections, numerically stable sigmoid)
- **Callable-based metrics** — accept `Callable[[ScoreVector], float]`, decoupled from RewardFunction protocol
- **Runner binds via `.compute`** — bridges protocol-based reward functions to the callable interface
- **Synthetic builder uses instance-level `random.Random`** — fully deterministic with seed, no global state mutation

---

## Session: 2026-02-12 — dev (PPE Benchmarking)

### Completed This Session
- Implemented full PPE (Preference Proxy Evaluations) experimental module (arXiv:2410.14872 adaptation)
- Created `src/agent_eval/experimental/` package with `ppe/` subpackage (7 source files)
- Created `tests/experimental/ppe/` with 5 test files (87 tests, all passing)
- 6 PPE metrics: pairwise_accuracy, best_of_k, spearman_correlation, kendall_tau, separability, brier_score
- Synthetic dataset generation from ScoreVectors with configurable ground truth
- BenchmarkRunner for single and comparison evaluation
- Text and dict report formatting
- Marked TODO items as done: "add reference to paper in experimental"

### In Progress (uncommitted)
- None — all changes committed

### Remaining Steps
- [ ] Create additional example scripts in `examples/` (e.g., LangGraph eval, LLM Judge, RL tuning)
- [x] ~~Add JSONL format support to `AutoGenAdapter`~~ — already done (`adapter.py` supports JSONL via `detect_format`)
- [ ] Document `attach_logger` usage in README or separate guide
- [ ] Add PPE benchmark example script to `examples/`

### Plan Accuracy Notes
- PPE module is new work beyond the original 4-phase plan — added as experimental section
- All prior phases remain complete and accurate

### Doc Sync
- Updated CLAUDE.md: test count 275 → 432, added experimental/ppe to Key Modules
- Updated IMPLEMENTATION.md: test count 332 → 432, added PPE phase section

---

## Phase 5a: Intrinsic Reasoning & Orchestration Effectiveness Scorers

### Status: COMPLETE (42 additional tests, 526 total passing)

Based on "Agentic Reasoning for Large Language Models" (Wei et al., 2026) — separates model capability (intrinsic reasoning) from workflow quality (orchestration), enabling the diagnosis "is my model bad or is my orchestration bad?"

### Files Created

#### Reasoning Scorer (`src/agent_eval/scorers/reasoning/`)
| File | Status | Description |
|------|--------|-------------|
| `__init__.py` | Done | Package init |
| `intrinsic.py` | Done | `IntrinsicReasoningScorer` — analyzes MESSAGE-level content for reasoning quality. 4 dimensions: `reasoning_depth` (reasoning marker ratio), `reasoning_coherence` (contradiction detection), `self_correction` (retraction/correction patterns), `plan_quality` (planning structure with actionability check). Issues: shallow reasoning WARNING, per-contradiction WARNING, no-planning-in-multistep INFO. |

#### Orchestration Scorer (`src/agent_eval/scorers/orchestration/`)
| File | Status | Description |
|------|--------|-------------|
| `__init__.py` | Done | Package init |
| `effectiveness.py` | Done | `OrchestrationScorer` — analyzes Episode-level structure (step ordering, agent transitions, tool patterns). 5 dimensions: `delegation_efficiency` (productive agent ratio), `tool_strategy` (retry/redundancy/diversity penalties), `coordination_overhead` (productive step ratio), `recovery_effectiveness` (failure recovery rate), `termination_quality` (wasted steps after answer). Issues: idle agent WARNING, tool retry loop ERROR, high overhead WARNING, unrecovered failure WARNING. |

#### Updated Files
| File | Change |
|------|--------|
| `scorers/registry.py` | Registered `intrinsic_reasoning` and `orchestration` in `_register_builtins()` |
| `cli/commands/evaluate.py` | Added `intrinsic` and `orchestration` to CLI scorer map |

#### Tests
| File | Status | Covers |
|------|--------|--------|
| `tests/scorers/reasoning/test_intrinsic.py` | Done | 19 tests — basics (name, protocol, dimensions, empty), reasoning depth (with/without/partial markers, shallow issue), coherence (no contradictions, with contradictions, single message, issue), self-correction (with/without, short ignored), plan quality (with/without, multistep issue, single step) |
| `tests/scorers/orchestration/test_effectiveness.py` | Done | 23 tests — basics (name, protocol, dimensions, empty), delegation (all productive, idle agent, idle issue), tool strategy (diverse, retry loop, retry issue, redundant, no tools), coordination overhead (all productive, high overhead, overhead issue), recovery (no failures, with/without recovery, partial, unrecovered issue), termination (clean, wasted steps, no steps) |

### Key Design Decisions
- **Zero breaking changes** — both scorers are additive, implementing the existing `Scorer` protocol
- **Content vs. structure separation** — IntrinsicReasoningScorer analyzes message text content; OrchestrationScorer analyzes step-level structure and patterns
- **Regex-based analysis** — no LLM calls required; both scorers are deterministic and zero-dependency
- **Scorer Registry names** — `intrinsic_reasoning` and `orchestration` (registry), `intrinsic` and `orchestration` (CLI shorthand)
- **Tool Strategy partially covers Phase 5 item #8** — redundancy detection, retry loop detection, and diversity scoring are now built in

---

## Phase 5: Strategic Roadmap

Based on domain research (18 libraries, 30+ papers, 15 adjacent projects, 8 community sources, 12 RL systems, 6 observability platforms evaluated).

### Execution Order

Items sorted by priority then effort. **Do the next unchecked item.**

| # | Item | Theme | Effort | Status | Why this order |
|---|------|-------|--------|--------|----------------|
| 1 | OTel GenAI Trace Adapter | Ingestion | Medium | **DONE** | Highest leverage — one adapter covers dozens of OTel-instrumented frameworks |
| 2 | OpenAI Agents SDK Adapter | Ingestion | Medium | Not Started | Largest single-framework user base (OpenAI ecosystem) |
| 3 | CrewAI Adapter | Ingestion | Medium | Not Started | Highest-starred OSS agent framework (~34k stars) |
| 4 | LlamaGuard Safety Scorer | Evaluation | Medium | Not Started | Critical for safe RL training — blocks responsible RL use |
| 5 | Per-Agent Credit Assignment | Evaluation | Large | Not Started | #1 stated gap; prerequisite for meaningful multi-agent RL |
| 6 | Process Reward Scorer | Evaluation | Large | Not Started | Unlocks step-level RL; build before RL bridges (synergy) |
| 7 | Google ADK Adapter | Ingestion | Medium | Not Started | Fast-growing Google ecosystem (~15.6k stars) |
| 8 | ~~Tool Strategy Scorer~~ | Evaluation | Medium | **Partially Done** | Covered by `OrchestrationScorer.tool_strategy` dimension |
| 9 | veRL Integration Bridge | RL | Medium | Not Started | Production-scale RL (ByteDance/NVIDIA) |
| 10 | OpenRLHF Integration Bridge | RL | Medium | Not Started | Distributed RL; pairs with process rewards |
| 11 | Population-Based Prompt Evolution | RL | Large | Not Started | Extends TuningLoop; avoids local optima |
| 12 | Synthetic Test Case Generator | Self-Sufficiency | Medium | Not Started | Prevents overfitting to static test sets |

### Theme Details

Below is reference detail for each item, grouped by theme.

#### 5A: Universal Trace Ingestion

_Make agent-eval the standard evaluation backend for any agent framework._

| Item | Description | Sources |
|------|-------------|---------|
| ~~OTel GenAI Trace Adapter~~ | **DONE.** `OTelTraceAdapter` reads OTLP JSON exports, maps `gen_ai.*` spans to Steps. Covers Langfuse, Phoenix, OpenLLMetry. Zero deps. 63 tests. | [OTel GenAI Semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/), [Agent Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) |
| OpenAI Agents SDK Adapter | Parse OpenAI Agents SDK structured traces (agents, handoffs, tool calls). Largest single-framework user base. | [GitHub](https://github.com/openai/openai-agents-python) |
| CrewAI Adapter | Parse CrewAI task execution logs. Highest-starred OSS agent framework (~34k stars). | [GitHub](https://github.com/crewAIInc/crewAI) |
| Google ADK Adapter | Parse Google ADK traces. Fast-growing Google ecosystem (~15.6k stars). | [GitHub](https://github.com/google/adk-python) |

#### 5B: Deeper Evaluation

_Move beyond episode-level scoring to fine-grained, multi-dimensional evaluation._

| Item | Description | Sources |
|------|-------------|---------|
| Per-Agent Credit Assignment | Decompose team-level ScoreVector into per-agent breakdowns. Heuristic + optional LLM-critic modes. Fills gap #1. | [Agent Lightning](https://github.com/microsoft/agent-lightning), [LLM-MCA](https://arxiv.org/abs/2502.16863) |
| Process Reward Scorer | Per-step rewards (heuristic, LLM-judge, implicit PRM). Unlocks step-level RL. Fills gap #3. | [PRIME](https://arxiv.org/abs/2502.01456), [AgentPRM](https://arxiv.org/abs/2502.10325) |
| Tool Strategy Scorer | Evaluate tool selection accuracy, argument correctness, redundant call detection. Fills gap #4. | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html), [tau-bench](https://github.com/sierra-research/tau-bench) |
| LlamaGuard Safety Scorer | LlamaGuard 3 wrapper for safety scoring. Critical for safe RL training. Fills gap #7. | [LlamaGuard 3](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/) |

#### 5C: RL Training Ecosystem

_Become the standard evaluation-to-reward bridge for all RL backends._

| Item | Description | Sources |
|------|-------------|---------|
| veRL Integration Bridge | Wrap EvalPipeline as veRL reward callable. Production-scale RL (ByteDance/NVIDIA). | [GitHub](https://github.com/volcengine/verl) |
| OpenRLHF Integration Bridge | Wrap EvalPipeline as OpenRLHF 0.8.0 external environment. | [GitHub](https://github.com/OpenRLHF/OpenRLHF) |
| Population-Based Prompt Evolution | Extend TuningLoop with population-based crossover/mutation (GEPA-style). Fills gap #2. | [GEPA](https://arxiv.org/pdf/2507.19457), [EvoPrompt](https://arxiv.org/abs/2309.08532) |

#### 5D: Self-Sufficiency

_Make agent-eval capable of generating its own test cases._

| Item | Description | Sources |
|------|-------------|---------|
| Synthetic Test Case Generator | LLM-driven test input generation. Co-evolves with TuningLoop. Fills gap #8. | [Ragas Synthetic Gen](https://docs.ragas.io/en/stable/getstarted/testset_generation/) |

### Competitive Landscape

| Tool | Stars | Evaluates Traces? | Produces RL Rewards? | Multi-Agent Credit? | Process Rewards? | Zero Core Deps? |
|------|-------|-------------------|---------------------|---------------------|-----------------|----------------|
| **agent-eval** | — | Yes | **Yes** | Planned | Planned | **Yes** |
| DeepEval | 13.8k | Yes (v3.0) | No | No | No | No |
| Ragas | 12.7k | Partial | No | No | No | No |
| Langfuse | 19k | Yes (observability) | No | No | No | No |
| TruLens | 3.1k | Yes | No | No | No | No |
| Phoenix | 7.8k | Yes (observability) | No | No | No | No |
| Inspect AI | 1.8k | Yes (safety) | No | No | No | No |
| Agent Lightning | — | Yes (RL-focused) | **Yes** | **Yes** | Partial | No |

**Unique positioning**: No existing tool combines trace-based agent evaluation with RL reward functions at zero core dependencies.

### Watch

| # | What | Why | Adopt When |
|---|------|-----|------------|
| 1 | Agent Lightning (MS Research) | Closest architectural cousin; credit assignment module | Stable release with documented API |
| 2 | HiPER (Hierarchical Plan-Execute RL) | HAE credit assignment at planner/executor levels | Public implementation released |
| 3 | MASPRM (Multi-Agent System PRM) | Multi-agent process reward model | Public implementation released |
| 4 | AgentPRM | Monte Carlo agent PRM for tool-use traces | Matures beyond research prototype |
| 5 | OpenPipe ART | Multi-turn agent GRPO trainer | Reaches beta stability |
| 6 | RL-Factory | Minimal reward-interface RL wrapper | Community adoption stabilizes |
| 7 | Inspect AI (UK AISI) | 100+ safety evaluations | Need for compliance-grade safety evals |
| 8 | Amazon Bedrock AgentCore Evals | 13 built-in evaluators; validates market demand | Customers request trace import |

### Skip

| # | What | Why |
|---|------|-----|
| 1 | lm-evaluation-harness (EleutherAI) | Benchmark-focused, not agent/trace evaluation |
| 2 | HELM (Stanford) | Foundation model benchmarking, not agent traces |
| 3 | Evidently AI | ML monitoring/drift detection, not RL rewards |
| 4 | Giskard v3 | Unstable API (v3 rewrite). Revisit when stable. |
| 5 | Promptfoo | TypeScript-native; not integrable into Python pipeline |
| 6 | LangSmith | Closed SaaS. LangGraph adapter covers open-source LangChain. |
| 7 | Braintrust | Closed SaaS. No open integration surface. |
| 8 | sdiehl/prm | Too narrow / prototype-quality. PRIME covers better. |
| 9 | AgentBench (Tsinghua) | Less maintained than SWE-bench/BFCL/tau-bench |

### Synergies

- **OTel Adapter + All Framework Adapters**: OTel may eventually subsume per-framework adapters as frameworks adopt OTel instrumentation. Build OTel first; add framework-specific adapters only where OTel coverage is insufficient.
- **Credit Assignment + Process Rewards**: Both decompose `ScoreVector` into sub-episode granularity. Share `step_scores` infrastructure.
- **Safety Scorer + Composite Reward**: LlamaGuard scores become hard constraints in `CompositeReward` (zero reward if unsafe), preventing RL reward-hacking.
- **Process Rewards + veRL/OpenRLHF**: Step-level rewards are the differentiating feature for scaled RL training systems. Build process rewards before RL bridges.
- **Test Case Generator + Prompt Evolution**: Co-evolving test cases and prompts prevents overfitting. Multiplicative value.

---

## Next Steps: Upgrade Roadmap Items

Migrated from `documentation/UPGRADE_ROADMAP.md` (previously `IMPLEMENTATION.md` was at repo root — moved to `documentation/`). Items below are **pending implementation**, reordered by impact for local/Docker selective-install workflow. TRL >=0.12 pin and `py.typed` marker were already completed and are omitted.

### Batch 1: Declare Missing Optional Dep Groups (Low Risk — Docker/Install Fix)

Directly fixes Docker selective-install pain: `pip install -e "/path/to/evallab[server]"` instead of pulling all deps or manually installing undeclared packages.

| # | Task | Detail | Status |
|---|------|--------|--------|
| 1 | Add `server` optional dep group | `server = ["fastapi>=0.100", "uvicorn>=0.20"]` in `[project.optional-dependencies]` | Done |
| 2 | Add `dspy` optional dep group | `dspy = ["dspy-ai>=2.5"]` in `[project.optional-dependencies]` | Done |
| 3 | Update `all` group | Include `server` and `dspy` in the `all` extras | Done |
| 4 | Verify | `PYTHONPATH=src venv/Scripts/python.exe -m pytest tests/ -v` | — |

**References:** [FastAPI](https://pypi.org/project/fastapi/), [DSPy](https://pypi.org/project/dspy/)

**Docker usage after this change:**
```dockerfile
# Selective installs — only pull what you need:
RUN pip install -e "/path/to/evallab[autogen]"          # just autogen adapter
RUN pip install -e "/path/to/evallab[server]"            # just reward server (no torch)
RUN pip install -e "/path/to/evallab[autogen,server]"    # combine as needed
RUN pip install -e "/path/to/evallab[dspy]"              # just DSPy bridge
RUN pip install -e "/path/to/evallab[all]"               # everything (now actually complete)
```

### Batch 2: Drop Black, Consolidate on Ruff Format (Low Risk)

| # | Task | Detail | Status |
|---|------|--------|--------|
| 1 | Remove `black>=24.0` from `[dev]` deps | Ruff formatter is a Black-compatible drop-in replacement (30x faster) | Not Started |
| 2 | Add `[tool.ruff.format]` config | `quote-style = "double"`, `line-ending = "auto"` | Not Started |
| 3 | Remove `[tool.black]` section | No longer needed after switch to ruff format | Not Started |
| 4 | Run `ruff format src/ tests/` | Reformat codebase with ruff | Not Started |
| 5 | Update README/CLAUDE.md commands | Replace `black src/` with `ruff format src/` | Not Started |
| 6 | Verify | `ruff check src/ tests/ && ruff format --check src/ tests/` | — |

**Migration notes:** Add `[tool.ruff.format]` with `quote-style = "double"` and `line-ending = "auto"`. Ruff's formatter is a Black-compatible drop-in (30x faster). See [Ruff formatter docs](https://docs.astral.sh/ruff/formatter/), [Black latest](https://pypi.org/project/black/).

### Batch 3: pytest-asyncio Migration (Medium Risk)

| # | Task | Detail | Status |
|---|------|--------|--------|
| 1 | Pin `pytest-asyncio>=1.0,<2.0` | Current `>=0.23` allows installing 1.x which may break deprecated `event_loop` fixture usage | Not Started |
| 2 | Search for `event_loop` fixture usage | Replace with `asyncio.get_running_loop()` if found | Not Started |
| 3 | Verify | `PYTHONPATH=src venv/Scripts/python.exe -m pytest tests/ -v` | — |

**Migration notes:** pytest-asyncio 1.0 removed the `event_loop` fixture and changed scoping behavior. The `asyncio_mode = "auto"` config key is still supported but behavior may differ. See [pytest-asyncio migration guide](https://pytest-asyncio.readthedocs.io/en/latest/how-to-guides/migrate_from_0_23.html), [ThinhDA blog](https://thinhdanggroup.github.io/pytest-asyncio-v1-migrate/).

### Batch 4: Bump Optional Dep Floors (Medium Risk)

| # | Task | Detail | Status |
|---|------|--------|--------|
| 1 | Pin `deepeval>=2.0` | deepeval 1.x→3.x had major API changes; test wrapper against 3.x | Not Started |
| 2 | Pin `ragas>=0.2` | Ragas 0.1→0.4 had significant API changes; test wrapper against 0.4.x | Not Started |
| 3 | Verify per-wrapper | Run wrapper tests with updated deps if installed | — |
| 4 | Verify full suite | `PYTHONPATH=src venv/Scripts/python.exe -m pytest tests/ -v` | — |

**Migration notes:** deepeval 1.x→3.x had major API changes (multi-turn evaluation, ArenaGEval, trace-based agent evaluation). Ragas 0.1→0.4 introduced a new metric system and custom metrics. Test `scorers/deepeval/wrapper.py` and `scorers/ragas/wrapper.py` against latest versions — import paths for metrics may have changed. See [DeepEval](https://pypi.org/project/deepeval/), [Ragas](https://pypi.org/project/ragas/).

### Batch 5: Python Version Bump (Higher Risk — Do Last)

| # | Task | Detail | Status |
|---|------|--------|--------|
| 1 | Update `requires-python = ">=3.11"` | Python 3.10 EOL Oct 2026; 3.11 brings 10-60% faster execution and better error messages | Not Started |
| 2 | Update `target-version` in `[tool.ruff]` | Change to `"py311"` (and `[tool.black]` if still present) | Not Started |
| 3 | Rebuild venv with Python 3.11+ | Ensure local environment matches new minimum | Not Started |
| 4 | Consider adopting 3.11+ features | `ExceptionGroup`, `TaskGroup` in async code | Not Started |
| 5 | Verify | Full test suite on Python 3.11+ | — |

**Migration notes:** Python 3.10 EOL is Oct 2026. 3.11 brings 10-60% faster execution and better error messages. No code changes expected since the project uses standard 3.10 features. See [Python EOL](https://endoflife.date/python), [Python 3.11 features](https://docs.python.org/3/whatsnew/3.11.html).

**Skipped upgrades (not worth pursuing):** hatchling latest (build-only dep, `requires = ["hatchling"]` already allows latest), switching build backend (hatchling is modern and well-configured), pinning numpy/torch exact versions (reduces compatibility for users with other packages), upgrading pytest beyond 9.0.x (already on latest major).

### Deferred (Low Priority for Solo/Local Use)

| # | Task | Detail | Status |
|---|------|--------|--------|
| 1 | Bump ruff floor to >=0.9 | Only matters for multi-contributor lint consistency | Not Started |
| 2 | Bump Python minimum to >=3.12 | Type parameter syntax, enhanced f-strings, incremental GC | Not Started |
| 3 | Add pre-commit hooks config | Automate ruff + formatting on commit | Not Started |
| 4 | Add CI pipeline (GitHub Actions) | Basic workflow: `pytest` + `ruff check` | Not Started |
| 5 | Test Python 3.13 free-threaded mode | Could benefit concurrent evaluation pipelines | Not Started |

### Batch Dependencies

- **Batch 2** (drop black) should happen before **Batch 5** (Python bump) to avoid updating black config that will be removed
- **Batch 3** (pytest-asyncio) is independent and can happen in any order
- **Batch 4** (dep floor bumps) is independent of other batches
- **Batch 5** (Python bump) should be last — highest-risk change

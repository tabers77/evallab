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

**Total: 332 tests passing** | Run with: `PYTHONPATH=src pytest tests/ -v`

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
| `fixtures/sample_autogen_log.txt` | Done | Full sample AutoGen log with Messages, ToolCalls, LLMCalls, FactCheckResult, and ANSWER block |
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

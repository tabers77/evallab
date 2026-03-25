# Evallab Issue: NumericConsistencyScorer False Positives on Derived/Calculated Numbers

## Problem

The `NumericConsistencyScorer` flags **all** numbers in the final answer that don't match tool output within 5% tolerance as "Data Fabrication: CRITICAL". It does not distinguish between:

1. **Tool-sourced numbers** — facts retrieved from DB/API tools (e.g., total spend = €94,004.24). These SHOULD be validated against tool output.
2. **Derived/calculated numbers** — estimates, percentages, and arithmetic the LLM computes from tool data (e.g., "save 5.5% → ~€953"). These are legitimately not in tool output.

This causes correct, high-quality answers to receive an F grade when the LLM produces scenario tables, projections, or any calculated recommendations.

## Reproduction

**Run:** `2026-03-23_15-55-26` in SalesNegotiator-engine

The orchestrator produced a negotiation scenario table:

```
| Defensive  | reduce blank add-on by 50% → ~4.5%; D&D 1%; freight 0-2% | ~5.5% to 7.5%  | ~€953 to €1,299  |
| Target     | reduce blank add-on by 100% → ~9.1%; freight 5-8%; D&D 2% | ~16.1% to 19.1% | ~€2,790 to €3,307 |
| Ambitious  | remove blank ~9.1%; freight 10-15%; D&D 3%                  | ~22.1% to 27.1% | ~€3,828 to €4,695 |
```

All 13 CRITICAL fabrication flags come from this table. The numbers are **proposed negotiation targets and computed savings**, not claims of fact from tool data. Examples:

| Flagged number | What it actually is | Closest tool number | Error |
|---|---|---|---|
| 50.00 | "reduce by 50%" — a proposed negotiation lever | 55.00 | 9.1% |
| 4.50 | "~4.5%" savings estimate | 5.00 | 10.0% |
| 5.50 | "~5.5%" total savings low end | 6.00 | 8.3% |
| 7.50 | "~7.5%" total savings high end | 8.00 | 6.2% |
| 1,299.00 | €1,299 = 7.5% of €17,322.77 (correct arithmetic) | 1,373.78 | 5.4% |
| 16.10 | "~16.1%" target scenario savings | 21.18 | 24.0% |
| 3,307.00 | €3,307 = 19.1% of €17,322.77 (correct arithmetic) | 3,557.04 | 7.0% |
| 15.00 | "10-15%" freight reduction target | 12.00 | 25.0% |
| 120.00 | Part of scenario parameters | 137.38 | 12.6% |

**Comparison:** 7 other runs have **0 fabrication issues** because their answers either lacked a final answer entirely or only cited direct tool output without scenario calculations.

## Affected Code

**File:** `evallab/src/agent_eval/scorers/numeric/consistency.py`

**Method:** `_find_fabrications()` (lines ~120-168)

```python
# Current logic (simplified):
for answer_num in answer_numbers:
    closest_error = min(abs(answer_num - tool_num) / max(abs(tool_num), 1e-9)
                        for tool_num in all_tool_numbers)
    if closest_error > self.tolerance:  # default 0.05 (5%)
        fabrications.append(...)  # CRITICAL
```

**Problem:** Every number extracted from the answer is compared against tool numbers with zero context about what the number represents.

## Suggested Fixes (Options)

### Option A: Number context classification (recommended)

Before comparing, classify each answer number as "factual" vs "derived":

- **Factual indicators:** preceded by keywords like "total spend", "revenue", "cost was", "data shows", specific supplier/period references
- **Derived indicators:** preceded by "~", "approximately", inside a scenario/projection table, percentage ranges ("X% to Y%"), preceded by "save", "reduce by", "target", "assume"

Only flag factual numbers as fabrication. Derived numbers could have a separate, softer check (e.g., "arithmetic consistency" — does 5.5% of €17,322.77 actually equal ~€953?).

### Option B: Arithmetic verification for derived numbers

Instead of checking if derived numbers exist in tool output, verify the arithmetic:

1. Identify the baseline number (€17,322.77 — which IS in tool output)
2. Verify that percentage * baseline = stated euro amount
3. Flag only if the arithmetic is wrong, not if the percentage itself isn't in tool output

### Option C: Tolerance tiers

Use different tolerance thresholds based on number magnitude or context:

- Exact match (1% tolerance) for large absolute numbers that likely come from DB (>10,000)
- Relaxed match (20% tolerance) or skip for small numbers (<100) that are likely percentages/parameters
- Skip numbers inside markdown table cells that contain "~" prefix (approximate marker)

### Option D: Allowlist for common non-tool numbers

Skip validation for:
- Round percentages (1, 2, 3, 5, 10, 15, 50, 100)
- Numbers preceded by "~" or "approximately"
- Numbers inside cells of tables whose headers contain "scenario", "target", "savings", "estimate"

## Impact

Without this fix, **any answer that includes projections, scenarios, or calculated recommendations will receive an F grade** regardless of actual quality. This penalizes the best answers (those that go beyond raw data to provide actionable analysis) the most severely.

The current run's answer was arguably the highest quality output across all 8 evaluated runs — it included supplier financials, market research, competitive analysis, and a structured negotiation playbook — yet scored the lowest.

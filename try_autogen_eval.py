from agent_eval.adapters.autogen.adapter import AutoGenAdapter
from agent_eval.pipeline.runner import EvalPipeline
from agent_eval.scorers.numeric.consistency import NumericConsistencyScorer
from agent_eval.scorers.rules.issue_detector import IssueDetectorScorer
from agent_eval.reporting.text import format_report

# 1. Parse the log into an Episode
adapter = AutoGenAdapter(agent_names=["SalesNegotiator", "FinanceExpert"])
episode = adapter.load_episode("tests/fixtures/sample_log/event.txt")

# Inspect the parsed episode
print(f"Episode ID: {episode.episode_id}")
print(f"Steps: {len(episode.steps)}")
for step in episode.steps:
    print(f"  {step.kind.name:12s} | {step.agent_name or 'N/A':25s} | {(step.tool_name or step.content or '')[:50]}")
print(f"Final answer: {episode.final_answer[:80]}...")
print(f"Duration: {episode.duration_seconds}s")

# 2. Run the evaluation pipeline
pipeline = EvalPipeline(
    adapter=adapter,
    scorers=[NumericConsistencyScorer(), IssueDetectorScorer()],
)
result = pipeline.evaluate(episode)

# 3. Inspect the result
print(f"\nGrade: {result.grade}")
print(f"Dimensions:")
for dim in result.score_vector.dimensions:
    print(f"  {dim.name}: {dim.value}/{dim.max_value}")
print(f"Issues: {len(result.score_vector.issues)}")
for issue in result.score_vector.issues:
    print(f"  [{issue.severity.name}] {issue.category}: {issue.description}")

# 4. Full formatted report
print("\n" + format_report(result, verbose=True))

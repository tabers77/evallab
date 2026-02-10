"""Tests for cli.commands.evaluate — evaluate command."""

from pathlib import Path

from agent_eval.cli.main import build_parser, app


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SAMPLE_LOG = FIXTURES_DIR / "sample_log" / "event.txt"


class TestEvaluateCommand:
    def test_parser_accepts_single_path(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate", str(SAMPLE_LOG)])
        assert args.command == "evaluate"
        assert args.paths == [str(SAMPLE_LOG)]

    def test_parser_accepts_multiple_paths(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate", str(SAMPLE_LOG), str(SAMPLE_LOG)])
        assert len(args.paths) == 2

    def test_parser_format_json(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate", str(SAMPLE_LOG), "--format", "json"])
        assert args.output_format == "json"

    def test_parser_brief_flag(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate", str(SAMPLE_LOG), "--brief"])
        assert args.brief is True

    def test_parser_custom_scorers(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate", str(SAMPLE_LOG), "--scorers", "numeric"])
        assert args.scorers == ["numeric"]

    def test_evaluate_text_output(self, capsys):
        app(["evaluate", str(SAMPLE_LOG)])
        captured = capsys.readouterr()
        assert "AGENT EVALUATION REPORT" in captured.out
        assert "OVERALL SCORE" in captured.out

    def test_evaluate_json_output(self, capsys):
        app(["evaluate", str(SAMPLE_LOG), "--format", "json"])
        captured = capsys.readouterr()
        import json

        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) == 1
        assert "episode_id" in data[0]

    def test_evaluate_brief_output(self, capsys):
        app(["evaluate", str(SAMPLE_LOG), "--brief"])
        captured = capsys.readouterr()
        assert "AGENT EVALUATION REPORT" in captured.out
        # Brief mode should NOT have detailed metrics
        assert "DETAILED METRICS" not in captured.out

    def test_evaluate_output_to_file(self, tmp_path):
        out_file = tmp_path / "report.txt"
        app(["evaluate", str(SAMPLE_LOG), "--output", str(out_file)])
        assert out_file.exists()
        content = out_file.read_text()
        assert "AGENT EVALUATION REPORT" in content


class TestCompareCommand:
    def test_parser_accepts_two_paths(self):
        parser = build_parser()
        args = parser.parse_args(["compare", str(SAMPLE_LOG), str(SAMPLE_LOG)])
        assert args.command == "compare"
        assert args.path_a == str(SAMPLE_LOG)
        assert args.path_b == str(SAMPLE_LOG)

    def test_compare_text_output(self, capsys):
        app(["compare", str(SAMPLE_LOG), str(SAMPLE_LOG)])
        captured = capsys.readouterr()
        assert "COMPARISON REPORT" in captured.out

    def test_compare_json_output(self, capsys):
        app(
            [
                "compare",
                str(SAMPLE_LOG),
                str(SAMPLE_LOG),
                "--format",
                "json",
            ]
        )
        captured = capsys.readouterr()
        import json

        data = json.loads(captured.out)
        assert "score_delta" in data

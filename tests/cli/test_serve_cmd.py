"""Tests for cli.commands.serve — serve command parser."""

from agent_eval.cli.main import build_parser


class TestServeCommand:
    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["serve"])
        assert args.command == "serve"
        assert args.host == "0.0.0.0"
        assert args.port == 8000
        assert args.scorers == ["issue_detector"]

    def test_parser_custom_host_port(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--host", "127.0.0.1", "--port", "9000"])
        assert args.host == "127.0.0.1"
        assert args.port == 9000

    def test_parser_custom_scorers(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--scorers", "numeric", "issue_detector"])
        assert args.scorers == ["numeric", "issue_detector"]

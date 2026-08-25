"""Tests for agent_eval.__version__ — instrument provenance.

``__version__`` was hardcoded and sat at "0.1.0" through the 0.2.x and 0.3.0
releases. That is a trap in this library specifically: its own audit finding
F10 is "no run records which instrument produced it", and a consumer reading
``agent_eval.__version__`` to record provenance stored a version that had not
shipped in months. It is now derived from the installed distribution.
"""

import agent_eval
from agent_eval import _resolve_version


class TestVersion:
    def test_is_a_non_empty_string(self):
        assert isinstance(agent_eval.__version__, str)
        assert agent_eval.__version__

    def test_is_not_the_stale_hardcoded_value(self):
        """0.1.0 is only correct if that is genuinely what is installed."""
        import importlib.metadata as md

        installed = None
        for dist in ("se-agent-eval", "agent-eval"):
            try:
                installed = md.version(dist)
                break
            except md.PackageNotFoundError:
                continue
        expected = installed if installed else "0.0.0.dev0"
        assert agent_eval.__version__ == expected

    def test_source_tree_reports_a_dev_marker(self, monkeypatch):
        """An uninstalled source tree must be honest, not claim a release."""
        import importlib.metadata as md

        def _missing(_name):
            raise md.PackageNotFoundError(_name)

        monkeypatch.setattr("agent_eval._version", _missing)
        assert _resolve_version() == "0.0.0.dev0"

    def test_falls_back_to_the_pre_rename_distribution_name(self, monkeypatch):
        """Environments installed before the 0.3.0 rename still resolve."""
        import importlib.metadata as md

        def _only_old_name(name):
            if name == "agent-eval":
                return "0.2.1"
            raise md.PackageNotFoundError(name)

        monkeypatch.setattr("agent_eval._version", _only_old_name)
        assert _resolve_version() == "0.2.1"

"""agent_eval: Framework-agnostic LLM/Agent evaluator with RL support."""

from importlib.metadata import PackageNotFoundError, version as _version


def _resolve_version() -> str:
    """Version of the installed distribution, or a dev marker in a source tree.

    Derived rather than hardcoded. This string sat at "0.1.0" through the 0.2.x
    and 0.3.0 releases — a trap in a library whose own audit finding F10 is "no
    run records which instrument produced it": a consumer reading
    ``agent_eval.__version__`` for provenance got a version that had not shipped
    in months. ``pyproject.toml`` is the single source of truth.

    Both distribution names are tried: the project was renamed to
    ``se-agent-eval`` in 0.3.0 to avoid dependency confusion with an unrelated
    public PyPI ``agent-eval``, and environments installed before the rename
    still carry the old name.
    """
    for dist in ("se-agent-eval", "agent-eval"):
        try:
            return _version(dist)
        except PackageNotFoundError:
            continue
    return "0.0.0.dev0"


__version__ = _resolve_version()

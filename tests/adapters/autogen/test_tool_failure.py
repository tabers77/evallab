"""Tests for adapters.autogen.tool_failure — tool failure detection."""

from agent_eval.adapters.autogen.tool_failure import is_tool_call_failed


class TestToolFailureDetection:
    # --- Tier 1: Structured payloads ---

    def test_dict_with_error_key(self):
        assert is_tool_call_failed({"error": "Connection timeout"}) is True

    def test_dict_with_errors_key(self):
        assert is_tool_call_failed({"errors": ["bad input"]}) is True

    def test_dict_with_exception_key(self):
        assert is_tool_call_failed({"exception": "ValueError"}) is True

    def test_dict_with_traceback_key(self):
        assert is_tool_call_failed({"traceback": "..."}) is True

    def test_dict_with_success_false(self):
        assert is_tool_call_failed({"success": False, "data": None}) is True

    def test_dict_with_ok_false(self):
        assert is_tool_call_failed({"ok": False}) is True

    def test_dict_with_status_error(self):
        assert is_tool_call_failed({"status": "error"}) is True

    def test_dict_with_http_error_status(self):
        assert is_tool_call_failed({"status_code": 404}) is True
        assert is_tool_call_failed({"status_code": 500}) is True

    # --- Tier 2: String patterns ---

    def test_string_validation_error(self):
        assert is_tool_call_failed("Validation error: invalid input") is True

    def test_string_tool_execution_error(self):
        assert is_tool_call_failed("ToolExecutionError occurred") is True

    def test_string_exception(self):
        assert is_tool_call_failed("Exception raised during execution") is True

    def test_string_http_errors(self):
        assert is_tool_call_failed("HTTP 404 error") is True
        assert is_tool_call_failed("HTTP 500 internal server error") is True

    def test_string_error_word(self):
        assert is_tool_call_failed("An error occurred") is True

    # --- Negative cases ---

    def test_successful_dict(self):
        assert is_tool_call_failed({"success": True, "data": "OK"}) is False

    def test_successful_string(self):
        assert (
            is_tool_call_failed("Query executed successfully with no errors") is False
        )

    def test_no_error_false_positive_guard(self):
        assert (
            is_tool_call_failed("Query completed successfully with no error found")
            is False
        )

    def test_ok_status_code(self):
        assert is_tool_call_failed({"status_code": 200, "data": "OK"}) is False

    def test_none_result(self):
        assert is_tool_call_failed(None) is False

    def test_numeric_result(self):
        assert is_tool_call_failed(42) is False

    def test_empty_string(self):
        assert is_tool_call_failed("") is False

    def test_status_ok(self):
        assert is_tool_call_failed({"status": "ok"}) is False

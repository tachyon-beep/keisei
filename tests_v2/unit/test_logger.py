"""
Tests for the unified logger output routing.

Smoke tests ensuring that the module-level logging functions do not crash
and that output is correctly routed to stderr.
"""

import pytest

from keisei.utils.unified_logger import (
    UnifiedLogger,
    create_module_logger,
    log_error_to_stderr,
    log_info_to_stderr,
    log_warning_to_stderr,
)


class TestLogFunctionsDoNotRaise:
    """Basic smoke tests: calling each log function must not raise."""

    def test_log_info_to_stderr_does_not_raise(self):
        log_info_to_stderr("TestComponent", "informational message")

    def test_log_warning_to_stderr_does_not_raise(self):
        log_warning_to_stderr("TestComponent", "warning message")

    def test_log_error_to_stderr_does_not_raise(self):
        log_error_to_stderr("TestComponent", "error message")

    def test_log_info_with_empty_string_does_not_raise(self):
        """An empty message string should be handled gracefully."""
        log_info_to_stderr("TestComponent", "")

    def test_log_error_with_exception_does_not_raise(self):
        """Passing an exception object should not crash."""
        log_error_to_stderr(
            "TestComponent", "something went wrong", RuntimeError("boom")
        )

    def test_rapid_log_calls_do_not_crash(self):
        """Calling log functions rapidly in a tight loop must not crash."""
        for i in range(100):
            log_info_to_stderr("StressTest", f"message {i}")


class TestLogOutputRoutedToStderr:
    """Verify that log output actually appears on stderr."""

    def test_info_appears_on_stderr(self, capfd):
        log_info_to_stderr("Router", "hello stderr")
        captured = capfd.readouterr()
        assert "hello stderr" in captured.err
        assert "INFO" in captured.err

    def test_warning_appears_on_stderr(self, capfd):
        log_warning_to_stderr("Router", "warning stderr")
        captured = capfd.readouterr()
        assert "warning stderr" in captured.err
        assert "WARNING" in captured.err

    def test_error_appears_on_stderr(self, capfd):
        log_error_to_stderr("Router", "error stderr")
        captured = capfd.readouterr()
        assert "error stderr" in captured.err
        assert "ERROR" in captured.err

    def test_component_name_appears_in_output(self, capfd):
        log_info_to_stderr("MyComponent", "check component name")
        captured = capfd.readouterr()
        assert "MyComponent" in captured.err


class TestUnifiedLoggerClass:
    """Tests for the UnifiedLogger class itself."""

    def test_create_module_logger_returns_logger(self):
        logger = create_module_logger("TestModule")
        assert isinstance(logger, UnifiedLogger)
        assert logger.name == "TestModule"

    def test_logger_info_does_not_raise(self):
        logger = create_module_logger("TestModule")
        logger.info("test info message")

    def test_logger_writes_to_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        logger = UnifiedLogger("FileTest", log_file_path=str(log_file))
        logger.info("file log message")
        contents = log_file.read_text()
        assert "file log message" in contents
        assert "INFO" in contents

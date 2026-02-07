"""
unified_logger.py: Unified logging infrastructure for consistent logging across the codebase.

This module provides a centralized logging interface to replace inconsistent print statements
and provide a unified approach to logging throughout the Keisei training system.
"""

import sys
from datetime import datetime
from typing import Optional


class UnifiedLogger:
    """
    A unified logging interface that provides consistent logging across the codebase.

    This logger automatically handles output to multiple targets:
    - File logging with timestamps
    - stderr output

    Designed to replace inconsistent print statements throughout the codebase.
    """

    def __init__(
        self,
        name: str,
        log_file_path: Optional[str] = None,
        enable_stderr: bool = True,
    ):
        """
        Initialize the unified logger.

        Args:
            name: Logger name (typically module name)
            log_file_path: Optional path to log file
            enable_stderr: Whether to output to stderr
        """
        self.name = name
        self.log_file_path = log_file_path
        self.enable_stderr = enable_stderr

    def _format_message(self, level: str, message: str) -> str:
        """Format a log message with timestamp and level."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [{self.name}] {level}: {message}"

    def _write_to_file(self, formatted_message: str) -> None:
        """Write message to log file if configured."""
        if self.log_file_path:
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(formatted_message + "\n")
                    f.flush()
            except (OSError, IOError):
                # Fallback to stderr if file logging fails
                print(
                    f"[LOGGER] Failed to write to log file: {formatted_message}",
                    file=sys.stderr,
                )

    def _write_to_console(self, formatted_message: str) -> None:
        """Write message to stderr."""
        if self.enable_stderr:
            print(formatted_message, file=sys.stderr)

    def info(self, message: str) -> None:
        """Log an info-level message."""
        formatted_message = self._format_message("INFO", message)
        self._write_to_file(formatted_message)
        self._write_to_console(formatted_message)

    def warning(self, message: str) -> None:
        """Log a warning-level message."""
        formatted_message = self._format_message("WARNING", message)
        self._write_to_file(formatted_message)
        self._write_to_console(formatted_message)

    def error(self, message: str) -> None:
        """Log an error-level message."""
        formatted_message = self._format_message("ERROR", message)
        self._write_to_file(formatted_message)
        self._write_to_console(formatted_message)

    def debug(self, message: str) -> None:
        """Log a debug-level message."""
        formatted_message = self._format_message("DEBUG", message)
        self._write_to_file(formatted_message)
        if self.enable_stderr:
            print(formatted_message, file=sys.stderr)

    def log(self, message: str, level: str = "INFO") -> None:
        """
        General log method for backward compatibility.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        level = level.upper()
        if level == "ERROR":
            self.error(message)
        elif level == "WARNING":
            self.warning(message)
        elif level == "DEBUG":
            self.debug(message)
        else:
            self.info(message)


def create_module_logger(
    module_name: str,
    log_file_path: Optional[str] = None,
) -> UnifiedLogger:
    """
    Create a logger for a specific module.

    Args:
        module_name: Name of the module (e.g., "SetupManager", "SessionManager")
        log_file_path: Optional path to log file

    Returns:
        UnifiedLogger instance configured for the module
    """
    return UnifiedLogger(
        name=module_name,
        log_file_path=log_file_path,
        enable_stderr=True,
    )


def log_error_to_stderr(
    component: str, message: str, exception: Optional[Exception] = None
) -> None:
    """
    Utility function for consistent error logging to stderr.

    Args:
        component: Name of the component reporting the error
        message: Error message
        exception: Optional exception object
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if exception:
        error_msg = f"[{timestamp}] [{component}] ERROR: {message}: {exception}"
    else:
        error_msg = f"[{timestamp}] [{component}] ERROR: {message}"

    print(error_msg, file=sys.stderr)


def log_warning_to_stderr(component: str, message: str) -> None:
    """
    Utility function for consistent warning logging to stderr.

    Args:
        component: Name of the component reporting the warning
        message: Warning message
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    warning_msg = f"[{timestamp}] [{component}] WARNING: {message}"
    print(warning_msg, file=sys.stderr)


def log_info_to_stderr(component: str, message: str) -> None:
    """
    Utility function for consistent info logging to stderr.

    Args:
        component: Name of the component reporting the info
        message: Info message
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_msg = f"[{timestamp}] [{component}] INFO: {message}"
    print(info_msg, file=sys.stderr)

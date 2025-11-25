"""Status spinner for the OpenStack Upgrade Assistant."""

import sys
import threading
import time
from typing import Optional


class StatusSpinner:
    """A simple console status spinner that runs in a background thread.

    This spinner is designed to not interfere with prompts or other output.
    It automatically stops and clears itself when needed.

    Attributes:
        message: Status message to display
        _running: Flag indicating if spinner is active
        _thread: Background thread running the spinner
        _spinner_chars: Characters to cycle through for animation
    """

    def __init__(self, message: str = "Thinking"):
        """Initialize the spinner.

        Args:
            message: Status message to display next to the spinner
        """
        self.message = message
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def start(self) -> None:
        """Start the spinner in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the spinner and clear the line."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)

        # Clear the spinner line by overwriting with spaces
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()

    def _spin(self) -> None:
        """Internal method that runs in background thread to animate the spinner."""
        idx = 0
        while self._running:
            spinner_char = self._spinner_chars[idx % len(self._spinner_chars)]
            sys.stdout.write(f'\r{spinner_char} {self.message}...')
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)

    def __enter__(self):
        """Context manager entry - start the spinner."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop the spinner."""
        self.stop()
        return False


# Global spinner instance that can be controlled from anywhere
_global_spinner: Optional[StatusSpinner] = None


def get_global_spinner() -> Optional[StatusSpinner]:
    """Get the current global spinner instance.

    Returns:
        The global spinner instance, or None if no spinner is active
    """
    return _global_spinner


def set_global_spinner(spinner: Optional[StatusSpinner]) -> None:
    """Set the global spinner instance.

    Args:
        spinner: Spinner instance to set as global, or None to clear
    """
    global _global_spinner
    _global_spinner = spinner


def stop_global_spinner() -> None:
    """Stop the global spinner if one is running."""
    global _global_spinner
    if _global_spinner:
        _global_spinner.stop()
        _global_spinner = None

import time


class Timer(object):
    """Timer for measuring and reporting performance."""

    def __init__(self) -> None:
        self.started = time.perf_counter()

    @property
    def sec(self) -> float:
        return time.perf_counter() - self.started

    def __str__(self):
        return f"{self.sec:1.01f} sec"


class Progress(object):
    """Prints messages to console as needed.
    """
    def print(self, msg):
        print(msg)


class DummyProgress(Progress):
    """Silent progress indicator. Doesn't output anything."""
    def print(self, msg):
        pass

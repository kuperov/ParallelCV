class Progress(object):
    """Progress indicator.

    Prints messages to console as needed.
    """

    def print(self, msg):
        print(msg)


class DummyProgress(Progress):
    """Silent progress indicator. Doesn't output anything."""

    def print(self, msg):
        pass

class CountError(Exception):
    exit_code: int = 1


class UsageError(CountError):
    exit_code = 2


class VisionError(CountError):
    exit_code = 3

"""Public session failures expose one abort class/message, never private details."""


class ProtocolError(ValueError):
    """Generic failure at the confidential workload boundary."""


ABORT = "certification session aborted"


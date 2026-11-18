"""Deadline-aware reads for peer sockets/pipes and finite local files."""
from __future__ import annotations

import io
import select
import socket
import time


class DeadlineReader:
    def __init__(self, source, deadline: float):
        self.source, self.deadline = source, deadline

    def read(self, n: int):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("submission deadline expired")
        if isinstance(self.source, socket.socket):
            old = self.source.gettimeout()
            try:
                self.source.settimeout(remaining)
                result = self.source.recv(n)
            finally:
                self.source.settimeout(old)
        elif isinstance(self.source, io.BytesIO):
            result = self.source.read(n)
        elif isinstance(self.source, (io.BufferedReader, io.FileIO)):
            ready, _, _ = select.select([self.source.fileno()], [], [], remaining)
            if not ready:
                raise TimeoutError("submission deadline expired")
            # read1 makes at most one underlying read; a buffered read(n) could
            # otherwise wait for n bytes even after select reports some data.
            read = getattr(self.source, 'read1', self.source.read)
            result = read(n)
        else:
            # Do not claim to interrupt an arbitrary user-supplied blocking read.
            raise ValueError("deadline requires a socket, finite file, pipe or BytesIO")
        if time.monotonic() > self.deadline:
            raise TimeoutError("submission deadline expired")
        return result


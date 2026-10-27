"""Read-only settlement gate for explicitly owned, separately grouped API children."""

from __future__ import annotations

import os
import socket
import subprocess
from dataclasses import dataclass, field


@dataclass(frozen=True)
class OwnedAPIGate:
    process: subprocess.Popen[bytes]
    port: int

    @classmethod
    def register(cls, process: subprocess.Popen[bytes], port: int) -> OwnedAPIGate:
        """Register while alive; caller must have used start_new_session=True.

        Only checks the supplied owned child. Does not discover arbitrary detached
        workers, stop processes, reserve a port or grant a reusable execution lease.
        """
        if os.name != "posix" or not 1 <= port <= 65535:
            raise ValueError("An owned POSIX process and explicit loopback port are required.")
        if process.poll() is not None or os.getpgid(process.pid) != process.pid:
            raise ValueError("Register a live owned child in its own process group.")
        return cls(process, port)

    def __call__(self) -> None:
        """Refuse unknown exit, surviving group members or an occupied listener."""
        if self.process.poll() != 0:
            raise ValueError("Owned API is active or did not exit cleanly.")
        try:
            os.killpg(self.process.pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("Owned process group still has surviving workers.")
        # Binding, rather than an HTTP request, sends no data to an unknown service.
        # REUSEADDR permits TIME_WAIT, but not another active listening socket.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", self.port))
            except OSError as exc:
                raise ValueError("Owned API loopback port is not available.") from exc


@dataclass
class OwnedAPIs:
    """One trial's append-only ownership set; later APIs invalidate earlier settlement."""

    _gates: list[OwnedAPIGate] = field(default_factory=list, init=False)

    def register(self, process: subprocess.Popen[bytes], port: int) -> None:
        self._gates.append(OwnedAPIGate.register(process, port))

    def __call__(self) -> None:
        for gate in self._gates:
            gate()

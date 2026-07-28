#!/usr/bin/env python3
"""Smoke-test that malformed UDP input cannot terminate the legacy engine."""

from __future__ import annotations

import argparse
import socket
import subprocess
import time
from pathlib import Path

MAGIC = b"\x57\x04\x00\x00"


def _available_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _header(seat: int, function: int) -> bytes:
    return MAGIC + bytes((seat, function, 0, 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    args = parser.parse_args()
    engine = args.engine.resolve()
    if not engine.is_file():
        parser.error(f"engine binary does not exist: {engine}")

    local_port = _available_udp_port()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as peer:
        peer.bind(("127.0.0.1", 0))
        peer.settimeout(3.0)
        remote_port = int(peer.getsockname()[1])
        process = subprocess.Popen(
            [
                str(engine),
                "--seat",
                "0",
                "--local-port",
                str(local_port),
                "--remote-port",
                str(remote_port),
                "--log-level",
                "2",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            # Startup readiness is signalled by a COMM_READY datagram.
            ready, _ = peer.recvfrom(200)
            if len(ready) < 8 or ready[:4] != MAGIC:
                raise RuntimeError("engine emitted an invalid readiness packet")

            malformed_packets = [
                b"x",
                _header(9, 0),  # invalid seat
                _header(0, 3) + b"\xff",  # truncated move
                _header(0, 4) + b"\xff",  # invalid event
                _header(0, 7) + bytes([0xFF]) * 30,  # invalid lineup values
                _header(0, 8) + b"\x01\x01\x01\x01",  # impossible init flags
            ]
            for packet in malformed_packets:
                peer.sendto(packet, ("127.0.0.1", local_port))
            time.sleep(0.25)
            if process.poll() is not None:
                output = process.communicate(timeout=1)[0]
                raise RuntimeError(f"engine exited after malformed input:\n{output}")
        finally:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=3)

    print("legacy protocol smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

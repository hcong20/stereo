"""Simple CAN sender wrapper using python-can for runtime integration.

Provides a lightweight `CANSender` to open a bus once and send distance
measurements as a 2-byte unsigned little-endian integer representing
millimetres in a standard CAN data payload.
"""
from __future__ import annotations

from dataclasses import dataclass
import struct
import typing


@dataclass
class CANConfig:
    channel: str = "can0"
    arbitration_id: int = 0x401
    bustype: str = "socketcan"


class CANSender:
    """Open a python-can bus and provide a simple `send_distance_m` method."""

    def __init__(self, cfg: CANConfig):
        self.cfg = cfg
        self.bus = None
        try:
            import can

            self.can = can
        except Exception:  # pragma: no cover - optional runtime dependency
            raise

        # Open the bus lazily on first send to avoid startup exceptions when not used.
        try:
            self.bus = self.can.interface.Bus(channel=self.cfg.channel, bustype=self.cfg.bustype)
        except Exception:
            # Let caller handle failures — raise with context preserved.
            raise

    def send_distance_mm(self, distance_mm: int) -> None:
        """Send a distance given in millimeters as an unsigned 2-byte little-endian integer."""
        if distance_mm is None:
            return
        if not (distance_mm > 0 and distance_mm != float("inf")):
            return
        dist_mm = distance_mm if isinstance(distance_mm, int) else int(distance_mm)
        # Ignore values outside the unsigned 16-bit range to avoid runtime errors.
        if dist_mm < 0 or dist_mm > 0xFFFF:
            return
        data = struct.pack("<H", dist_mm)
        msg = self.can.Message(arbitration_id=int(self.cfg.arbitration_id), data=data, is_extended_id=False)
        try:
            self.bus.send(msg)
        except Exception:
            # Non-fatal: ignore send errors to avoid disrupting runtime loop.
            pass

    def shutdown(self) -> None:
        """Close the bus cleanly if available."""
        try:
            if self.bus is not None:
                try:
                    self.bus.shutdown()
                except Exception:
                    pass
        finally:
            self.bus = None

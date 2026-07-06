"""Safety and lightweight cost controls for AgentFlow."""

from __future__ import annotations

import re


class SafetyController:
    """Redact secrets and estimate prompt cost pressure."""

    SECRET_PATTERNS = [
        re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
        re.compile(r"AIza[A-Za-z0-9_\-]{8,}"),
        re.compile(r"postgresql://[^\s]+:[^\s]+@"),
    ]

    def redact(self, text: str) -> str:
        redacted = str(text)
        for pattern in self.SECRET_PATTERNS:
            redacted = pattern.sub("[REDACTED_SECRET]", redacted)
        return redacted

    def estimate_units(self, text: str) -> int:
        """Approximate token pressure with a conservative char/4 estimate."""
        return max(1, len(str(text)) // 4)


_safety_controller: SafetyController | None = None


def get_safety_controller() -> SafetyController:
    global _safety_controller
    if _safety_controller is None:
        _safety_controller = SafetyController()
    return _safety_controller

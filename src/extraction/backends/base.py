from __future__ import annotations

from typing import Any, Protocol, Sequence


class VLMBackend(Protocol):
    """Minimal generation contract of one scene graph request."""

    model_id: str

    def generate(
        self,
        images: Sequence[Any],
        prompt: str,
        max_new_tokens: int,
    ) -> str: ...

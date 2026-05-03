"""
Session state schema — persisted in sessions.state (JSON column).

Loaded from DB each turn, injected into the root agent's instruction via ADK's
{var} template mechanism, then saved back after the turn completes.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class SessionState(BaseModel):
    user_id: str
    plan_tier: Literal["free", "pro", "enterprise"] = "free"
    last_agent: str | None = None
    turn_count: int = 0

    def to_db_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_adk_state(self) -> dict[str, Any]:
        """Return a flat dict suitable for ADK session.state injection."""
        return {
            "user_id": self.user_id,
            "plan_tier": self.plan_tier,
            "last_agent": self.last_agent or "none",
            "turn_count": str(self.turn_count),
        }

    @classmethod
    def from_db_dict(cls, data: dict[str, Any]) -> SessionState:
        return cls.model_validate(data)

"""
Account tools — used by AccountAgent.

Mock data is used for the take-home; the ADK tool wiring is what's evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BuildSummary:
    build_id: str
    pipeline: str
    status: str  # passed | failed | cancelled
    branch: str
    started_at: str   # ISO-8601 string (serialisable for ADK tool responses)
    duration_seconds: int


@dataclass
class AccountStatus:
    user_id: str
    plan_tier: str
    concurrent_builds_used: int
    concurrent_builds_limit: int
    storage_used_gb: float
    storage_limit_gb: float


_MOCK_BUILDS: dict[str, list[dict[str, object]]] = {
    "default": [
        {
            "build_id": "bld_001",
            "pipeline": "ci/test",
            "status": "failed",
            "branch": "main",
            "started_at": "2026-05-01T10:00:00Z",
            "duration_seconds": 142,
        },
        {
            "build_id": "bld_002",
            "pipeline": "ci/test",
            "status": "passed",
            "branch": "feat/login",
            "started_at": "2026-05-01T09:30:00Z",
            "duration_seconds": 178,
        },
        {
            "build_id": "bld_003",
            "pipeline": "ci/lint",
            "status": "failed",
            "branch": "fix/auth",
            "started_at": "2026-05-01T08:00:00Z",
            "duration_seconds": 23,
        },
        {
            "build_id": "bld_004",
            "pipeline": "ci/test",
            "status": "passed",
            "branch": "main",
            "started_at": "2026-04-30T20:00:00Z",
            "duration_seconds": 195,
        },
        {
            "build_id": "bld_005",
            "pipeline": "ci/deploy",
            "status": "cancelled",
            "branch": "main",
            "started_at": "2026-04-30T19:00:00Z",
            "duration_seconds": 60,
        },
    ]
}

_PLAN_LIMITS: dict[str, dict[str, object]] = {
    "free":       {"concurrent_builds_limit": 1, "storage_limit_gb": 5.0},
    "pro":        {"concurrent_builds_limit": 5, "storage_limit_gb": 50.0},
    "enterprise": {"concurrent_builds_limit": 20, "storage_limit_gb": 500.0},
}


async def get_recent_builds(user_id: str, limit: int = 5) -> list[dict[str, object]]:
    """
    Return the most recent builds for a user, newest first.

    Args:
        user_id: The user's account ID.
        limit: Maximum number of builds to return (default 5).

    Returns:
        List of build summaries as dicts with keys: build_id, pipeline,
        status, branch, started_at, duration_seconds.
    """
    builds = _MOCK_BUILDS.get(user_id, _MOCK_BUILDS["default"])
    return [dict(b) for b in builds[:limit]]


async def get_account_status(user_id: str) -> dict[str, object]:
    """
    Return current account status including plan tier and resource usage.

    Args:
        user_id: The user's account ID.

    Returns:
        Dict with keys: user_id, plan_tier, concurrent_builds_used,
        concurrent_builds_limit, storage_used_gb, storage_limit_gb.
    """
    plan = "pro" if user_id.startswith("enterprise_") else "free"
    limits = _PLAN_LIMITS.get(plan, _PLAN_LIMITS["free"])
    return {
        "user_id": user_id,
        "plan_tier": plan,
        "concurrent_builds_used": 1,
        "concurrent_builds_limit": limits["concurrent_builds_limit"],
        "storage_used_gb": 2.3,
        "storage_limit_gb": limits["storage_limit_gb"],
    }

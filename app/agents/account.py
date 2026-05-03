"""AccountAgent — handles user account and build queries via mock tools."""
from __future__ import annotations

from google.adk.agents import LlmAgent

from app.agents.tools.account_tools import get_account_status, get_recent_builds
from app.settings import settings

ACCOUNT_INSTRUCTION = """You are the Helix account specialist.

Answer questions about the user's account using the available tools:
- get_recent_builds: fetches the user's recent CI/CD builds
- get_account_status: fetches plan tier and resource usage

The user's user_id is available in context — always pass it to the tools.
Present results in a clear, concise format. Never fabricate data.
"""

account_agent = LlmAgent(
    name="account_agent",
    model=settings.adk_model,
    description=(
        "Handles questions about the user's account: recent builds, build status, "
        "plan tier, resource usage, and account limits. Use for: 'show my builds', "
        "'what is my plan', 'how many builds did I run', account status queries."
    ),
    instruction=ACCOUNT_INSTRUCTION,
    tools=[get_recent_builds, get_account_status],
)

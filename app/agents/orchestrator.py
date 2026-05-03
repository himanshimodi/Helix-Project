"""
SROP Root Orchestrator — Google ADK agent.

Routes every user turn to KnowledgeAgent or AccountAgent via ADK AgentTool.
Routing is performed by LLM tool-selection — not string parsing.

The instruction uses {var} placeholders that ADK resolves from session.state
at runtime, so the root agent is created once and reused across all turns.
State variables injected per turn: user_id, plan_tier, last_agent, turn_count.
"""
from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from app.agents.account import account_agent
from app.agents.knowledge import knowledge_agent
from app.settings import settings

ROOT_INSTRUCTION = """You are the Helix Support Concierge — a routing agent.

Current user context:
- user_id: {user_id}
- plan_tier: {plan_tier}
- last_agent_used: {last_agent?}
- conversation_turn: {turn_count}

Route every user message to the correct specialist tool:
- HOW to do something, WHAT a feature is, docs/config questions → knowledge_agent
- Their builds, account status, plan usage, pipeline results → account_agent
- Greetings, thanks, or clearly off-topic → respond directly (no tool call)

Rules:
1. Always call a tool when intent matches knowledge or account.
2. Never answer knowledge or account questions yourself — delegate.
3. Pass the user_id from context when the account tool needs it.
4. If unsure, prefer knowledge_agent.
"""

root_agent = LlmAgent(
    name="srop_root",
    model=settings.adk_model,
    instruction=ROOT_INSTRUCTION,
    tools=[
        AgentTool(agent=knowledge_agent),
        AgentTool(agent=account_agent),
    ],
)

"""POST /v1/sessions — create a session."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Session, User
from app.db.session import get_db
from app.srop.state import SessionState

router = APIRouter(tags=["sessions"])

_VALID_TIERS = {"free", "pro", "enterprise"}


class CreateSessionRequest(BaseModel):
    model_config = {"extra": "forbid"}

    user_id: str = Field(min_length=1, max_length=64)
    plan_tier: str = Field(default="free")


class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: str


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    """Create a session. Upserts the user if not seen before."""
    plan_tier = body.plan_tier if body.plan_tier in _VALID_TIERS else "free"

    user = await db.get(User, body.user_id)
    if user is None:
        user = User(user_id=body.user_id, plan_tier=plan_tier)
        db.add(user)
    else:
        user.plan_tier = plan_tier

    session_id = str(uuid.uuid4())
    state = SessionState(user_id=body.user_id, plan_tier=plan_tier)  # type: ignore[call-arg]
    session = Session(
        session_id=session_id,
        user_id=body.user_id,
        state=state.to_db_dict(),
    )
    db.add(session)
    await db.commit()

    return CreateSessionResponse(session_id=session_id, user_id=body.user_id)

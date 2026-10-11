import logging

from fastapi import APIRouter

from app.schemas.dashboard import DashboardSummaryResponse
from app.services.dashboard import build_dashboard_summary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummaryResponse)
def get_summary():
    return DashboardSummaryResponse(**build_dashboard_summary())

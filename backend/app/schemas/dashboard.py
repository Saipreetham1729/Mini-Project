from typing import Dict, Optional

from pydantic import BaseModel

from app.schemas.benchmark import PrfResultOut


class DashboardSummaryResponse(BaseModel):
    total_questions: int
    total_claims: int
    labeled_claims: int
    unlabeled_claims: int
    status_counts: Dict[str, int]
    failure_type_counts: Dict[str, int]
    model_counts: Dict[str, int]
    avg_latency_seconds: Optional[float] = None
    prf_by_method: Dict[str, PrfResultOut]

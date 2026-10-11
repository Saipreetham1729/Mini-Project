from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import benchmark, dashboard, documents, evaluation, mitigation, query, verification
from app.config import settings
from app.db.database import init_db

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

app.include_router(documents.router)
app.include_router(query.router)
app.include_router(verification.router)
app.include_router(mitigation.router)
app.include_router(evaluation.router)
app.include_router(benchmark.router)
app.include_router(dashboard.router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "app": settings.app_name,
        "llm_provider": settings.llm_provider,
        "llm_model_name": settings.llm_model_name if settings.llm_provider == "openai" else None,
        "is_mock": settings.llm_provider == "mock",
    }

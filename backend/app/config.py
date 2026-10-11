from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "ContextGuard"
    debug: bool = True
    llm_api_key: str = ""
    llm_provider: str = "mock"  # "mock" (offline testing) or "openai" (OpenAI or any OpenAI-compatible API)
    llm_model_name: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_request_timeout: int = 60

    # Storage paths, relative to backend/
    upload_dir: Path = Path("storage/uploads")
    processed_dir: Path = Path("storage/processed")
    vector_db_dir: Path = Path("storage/vectordb")

    # Chunking defaults
    chunk_size_words: int = 200
    chunk_overlap_words: int = 40

    # Embedding / retrieval defaults
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_top_k: int = 5

    # NLI faithfulness metric (Phase 8) -- xsmall variant chosen for a light download
    nli_model_name: str = "cross-encoder/nli-deberta-v3-xsmall"

    # Basic evidence verification thresholds (Phase 5 baseline -- cosine similarity).
    # These are heuristic starting points; expect to tune them once you have
    # real LLM answers to look at. Proper faithfulness metrics arrive in Phase 8.
    support_threshold: float = 0.55
    partial_support_threshold: float = 0.35

    # Likely failure-type classification thresholds (Phase 6, heuristic only).
    retrieval_relevance_threshold: float = 0.35  # below this, treat retrieval as having found nothing relevant
    strong_retrieval_threshold: float = 0.6      # above this, retrieval is considered to have succeeded overall
    conflict_dissimilarity_threshold: float = 0.3  # below this pairwise similarity, relevant chunks are flagged as possibly divergent

    # Self-consistency metric (extended scope): independently re-generate the
    # answer N times at a higher temperature, then measure how often each
    # original claim is entailed across those independent samples. Low
    # agreement across samples is a hallucination-risk signal even without
    # any retrieved evidence to check against.
    self_consistency_samples: int = 3
    self_consistency_temperature: float = 0.8


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.processed_dir.mkdir(parents=True, exist_ok=True)
settings.vector_db_dir.mkdir(parents=True, exist_ok=True)

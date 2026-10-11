"""
Shared prompt text. Lives in its own module -- with no other project
imports -- specifically so a lightweight consumer (self_consistency) can use
the exact same system prompt as the main pipeline without transitively
importing vector_store/embedding_service and their heavy ML dependencies.
"""

RAG_SYSTEM_PROMPT = (
    "You are an assistant that answers questions based on the provided document context. "
    "Use the context to inform your answer. If the context is insufficient, do your best "
    "to answer using the available information, and note if you are uncertain."
)

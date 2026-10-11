import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, UploadFile

from app.config import settings
from app.schemas.document import (
    ChunksListResponse,
    DocumentUploadResponse,
    EmbedResponse,
    ProcessResponse,
    RetrievedChunkOut,
    RetrieveRequest,
    RetrieveResponse,
)
from app.services import vector_store
from app.services.chunker import chunk_document
from app.services.embedding_service import embed_query, embed_texts
from app.services.pdf_extractor import extract_text_by_page

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now.")

    doc_id = uuid.uuid4().hex[:12]
    dest_path = settings.upload_dir / f"{doc_id}.pdf"

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    dest_path.write_bytes(contents)
    logger.info("Saved upload doc_id=%s filename=%s", doc_id, file.filename)

    return DocumentUploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        message="Uploaded. Call /documents/{doc_id}/process next.",
    )


@router.post("/{doc_id}/process", response_model=ProcessResponse)
def process_document(doc_id: str):
    pdf_path = settings.upload_dir / f"{doc_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"No uploaded document with doc_id={doc_id}")

    try:
        pages = extract_text_by_page(pdf_path)
        chunks = chunk_document(
            pages,
            doc_id=doc_id,
            chunk_size=settings.chunk_size_words,
            overlap=settings.chunk_overlap_words,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    out_path = settings.processed_dir / f"{doc_id}_chunks.json"
    out_path.write_text(json.dumps(chunks, indent=2))

    return ProcessResponse(
        doc_id=doc_id,
        page_count=len(pages),
        chunk_count=len(chunks),
        message="Processed. Call /documents/{doc_id}/chunks to inspect them.",
    )


@router.get("/{doc_id}/chunks", response_model=ChunksListResponse)
def get_chunks(doc_id: str):
    chunks_path = settings.processed_dir / f"{doc_id}_chunks.json"
    if not chunks_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No processed chunks for doc_id={doc_id}. Run /process first.",
        )

    chunks = json.loads(chunks_path.read_text())
    return ChunksListResponse(doc_id=doc_id, chunk_count=len(chunks), chunks=chunks)


@router.post("/{doc_id}/embed", response_model=EmbedResponse)
def embed_document(doc_id: str):
    chunks_path = settings.processed_dir / f"{doc_id}_chunks.json"
    if not chunks_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No processed chunks for doc_id={doc_id}. Run /process first.",
        )

    chunks = json.loads(chunks_path.read_text())
    if not chunks:
        raise HTTPException(status_code=422, detail="Document has no chunks to embed.")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    vector_store.delete_doc(doc_id)  # idempotent: safe to re-embed after re-processing
    vector_store.add_chunks(doc_id, chunks, embeddings)

    return EmbedResponse(
        doc_id=doc_id,
        embedded_chunk_count=len(chunks),
        message="Embedded and stored. Call /documents/{doc_id}/retrieve to search.",
    )


@router.post("/{doc_id}/retrieve", response_model=RetrieveResponse)
def retrieve_chunks(doc_id: str, body: RetrieveRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query_embedding = embed_query(body.query)
    results = vector_store.query(doc_id, query_embedding, top_k=body.top_k)

    return RetrieveResponse(
        doc_id=doc_id,
        query=body.query,
        results=[RetrievedChunkOut(**r) for r in results],
    )

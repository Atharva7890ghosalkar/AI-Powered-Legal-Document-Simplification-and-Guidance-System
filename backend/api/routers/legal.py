import logging
import json
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.api.deps import get_legal_assistant_service
from src.models.schemas import AskRequest, AskResponse, UploadResponse

if TYPE_CHECKING:
    from src.services.legal_assistant_service import LegalAssistantService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["legal"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    service: "LegalAssistantService" = Depends(get_legal_assistant_service),
) -> UploadResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return service.process_document(file_bytes=content, filename=file.filename or "document.pdf")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Document processing failed")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {exc}") from exc


@router.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    service: "LegalAssistantService" = Depends(get_legal_assistant_service),
) -> AskResponse:
    try:
        return service.answer_question(request)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Question answering failed")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc


@router.post("/ask/stream")
def ask_question_stream(
    request: AskRequest,
    service: "LegalAssistantService" = Depends(get_legal_assistant_service),
) -> StreamingResponse:
    try:
        stream, chunks, citations = service.answer_question_stream(request)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Question streaming setup failed")
        raise HTTPException(status_code=500, detail=f"Failed to initialize stream: {exc}") from exc

    def event_stream():
        accumulated = []
        context_event = {
            "type": "context",
            "retrieved_context": [chunk.model_dump() for chunk in chunks],
            "citations": [citation.model_dump() for citation in citations],
        }
        yield f"data: {json.dumps(context_event)}\n\n"

        try:
            for token in stream:
                accumulated.append(token)
                token_event = {"type": "token", "delta": token}
                yield f"data: {json.dumps(token_event)}\n\n"
        except Exception as exc:  # noqa: BLE001
            error_event = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(error_event)}\n\n"
            return

        done_event = {
            "type": "done",
            "answer": "".join(accumulated),
            "citations": [citation.model_dump() for citation in citations],
            "retrieved_context": [chunk.model_dump() for chunk in chunks],
        }
        yield f"data: {json.dumps(done_event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

import logging
import uuid

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ChatRequest, ChatResponse, HealthResponse
from api.dependencies import ApplicationServices

logger = logging.getLogger(__name__)


def create_app(services: ApplicationServices) -> FastAPI:
    app = FastAPI(
        title="Enterprise Advanced RAG API",
        description="Enterprise RAG with hybrid retrieval, conversational memory and document management.",
        version="1.2.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )
    app.state.services = services

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health_check():
        return HealthResponse(status="healthy", service="Enterprise Advanced RAG")

    @app.post("/api/v1/chat", response_model=ChatResponse, tags=["RAG"])
    def chat(payload: ChatRequest, request: Request):
        service = request.app.state.services
        session_id = payload.session_id or str(uuid.uuid4())
        try:
            service.validate_question(payload.question)
            result = service.conversational_rag.ask(
                question=payload.question, session_id=session_id
            )
            return ChatResponse(
                session_id=session_id,
                question=payload.question,
                rewritten_question=result.get("rewritten_question"),
                answer=result.get("answer", ""),
                sources=result.get("sources", []),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception:
            logger.exception("RAG request failed")
            raise HTTPException(status_code=500, detail="An internal error occurred while processing your question.")

    @app.delete("/api/v1/sessions/{session_id}", tags=["Conversation"])
    def clear_session(session_id: str, request: Request):
        try:
            request.app.state.services.conversational_rag.clear_session(session_id)
            return {"message": "Conversation history cleared.", "session_id": session_id}
        except Exception:
            logger.exception("Failed to clear conversation session")
            raise HTTPException(status_code=500, detail="Unable to clear conversation history.")

    @app.post("/api/v1/documents/upload", tags=["Documents"])
    async def upload_document(request: Request, file: UploadFile = File(...)):
        service = request.app.state.services
        if service.document_manager is None:
            raise HTTPException(status_code=503, detail="Document management service is unavailable.")
        try:
            max_bytes = service.document_manager.validator.max_file_size_bytes
            file_bytes = await file.read(max_bytes + 1)
            if len(file_bytes) > max_bytes:
                raise HTTPException(status_code=413, detail="Uploaded file exceeds the maximum allowed size.")
            return service.upload_document(file.filename or "", file_bytes)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception:
            logger.exception("Document upload failed")
            raise HTTPException(status_code=500, detail="Document processing or indexing failed.")
        finally:
            await file.close()

    @app.get("/api/v1/documents", tags=["Documents"])
    def list_documents(request: Request):
        try:
            return {"documents": request.app.state.services.document_history()}
        except Exception:
            logger.exception("Failed to list documents")
            raise HTTPException(status_code=500, detail="Unable to retrieve document history.")

    @app.get("/api/v1/documents/stats", tags=["Documents"])
    def document_stats(request: Request):
        try:
            return request.app.state.services.document_stats()
        except Exception:
            logger.exception("Failed to retrieve document statistics")
            raise HTTPException(status_code=500, detail="Unable to retrieve document statistics.")

    @app.get("/api/v1/documents/{document_id}", tags=["Documents"])
    def get_document(document_id: str, request: Request):
        try:
            record = request.app.state.services.get_document(document_id)
            if record is None:
                raise HTTPException(status_code=404, detail="Document not found.")
            return record
        except HTTPException:
            raise
        except Exception:
            logger.exception("Failed to retrieve document")
            raise HTTPException(status_code=500, detail="Unable to retrieve document.")

    @app.delete("/api/v1/documents/{document_id}", tags=["Documents"])
    def delete_document(document_id: str, request: Request):
        try:
            result = request.app.state.services.delete_document(document_id)
            if result is None:
                raise HTTPException(status_code=404, detail="Document not found.")
            return result
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception:
            logger.exception("Document deletion failed")
            raise HTTPException(status_code=500, detail="Document deletion failed. Check server logs and reconcile indexes before retrying.")

    return app


def build_production_app():
    from api.bootstrap import build_services
    return create_app(build_services())

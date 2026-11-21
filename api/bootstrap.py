from pathlib import Path

import yaml

from api.dependencies import ApplicationServices

from src.generation.rag_chain import EnterpriseRAGChain

from src.memory.conversation_store import ConversationStore
from src.memory.conversation_manager import ConversationManager
from src.memory.query_rewriter import QueryRewriter

from src.pipeline.conversational_rag import ConversationalRAG

from src.guardrails.input_guard import InputGuard

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.ingestion_pipeline import IngestionPipeline

from src.parsing.chunk_manager import ChunkManager

from src.document_management.indexing_service import IndexingService
from src.document_management.document_manager import DocumentManager


def load_yaml_config(config_path: str) -> dict:

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def build_services() -> ApplicationServices:

    print("\n========== INITIALIZING ENTERPRISE RAG ==========")

    # 1. Load configuration.
    conversation_config = load_yaml_config(
        "config/conversation_config.yaml"
    )["conversation"]

    guardrail_config = load_yaml_config(
        "config/guardrails_config.yaml"
    )["guardrails"]

    # 2. Initialize the existing RAG pipeline.
    print("[1/6] Loading RAG pipeline...")

    rag_chain = EnterpriseRAGChain(
        config_path="config/generation_config.yaml"
    )

    print("[SUCCESS] RAG pipeline initialized.")

    # 3. Initialize conversation memory.
    print("[2/6] Initializing conversation memory...")

    storage_config = conversation_config["storage"]
    memory_config = conversation_config["memory"]

    if storage_config["type"] != "sqlite":
        raise ValueError(
            "Only SQLite conversation storage is configured."
        )

    conversation_store = ConversationStore(
        db_path=storage_config["database_path"]
    )

    conversation_manager = ConversationManager(
        store=conversation_store,
        max_turns=memory_config["max_turns"]
    )

    # 4. Initialize query rewriter.
    print("[3/6] Initializing query rewriter...")

    rewrite_config = conversation_config["query_rewriting"]

    if rewrite_config.get("enabled", True):

        groq_llm = rag_chain.generator.llm_client.llm

        query_rewriter = QueryRewriter(
            llm=groq_llm
        )

    else:

        class IdentityQueryRewriter:

            @staticmethod
            def rewrite(question, conversation_history):
                return question

        query_rewriter = IdentityQueryRewriter()

    # 5. Initialize input guardrail.
    print("[4/6] Initializing input guardrail...")

    input_guard = InputGuard(
        max_length=guardrail_config["input"]["max_length"]
    )

    # 6. Connect conversational RAG.
    print("[5/6] Connecting conversational RAG...")

    conversational_rag = ConversationalRAG(
        rag_chain=rag_chain,
        conversation_manager=conversation_manager,
        query_rewriter=query_rewriter
    )

    # 7. Initialize incremental document upload.
    print("[6/6] Initializing document upload service...")

    # Reuse the existing retrieval components.
    dense_retriever = rag_chain.retriever.dense
    sparse_retriever = rag_chain.retriever.sparse
    hybrid_retriever = rag_chain.retriever.hybrid

    faiss_store = dense_retriever.vectorstore

    # Reuse the embedding model already loaded by DenseRetriever.
    embedding_model = dense_retriever.embedding_model

    document_loader = DocumentLoader()
    chunk_manager = ChunkManager()

    ingestion_pipeline = IngestionPipeline(
        load_documents=document_loader.load_file,
        chunk_manager=chunk_manager,
        embedding_model=embedding_model
    )

    indexing_service = IndexingService(
        faiss_store=faiss_store,
        sparse_retriever=sparse_retriever,
        vectorstore_path=dense_retriever.vectorstore_path,
        hybrid_retriever=hybrid_retriever
    )

    document_manager = DocumentManager(
        ingestion_pipeline=ingestion_pipeline,
        indexing_service=indexing_service,
        upload_directory="data/uploads",
        max_file_size_mb=20
    )

    services = ApplicationServices(
        conversational_rag=conversational_rag,
        input_guard=input_guard,
        document_manager=document_manager
    )

    print("\n========== RAG INITIALIZATION COMPLETE ==========\n")

    return services
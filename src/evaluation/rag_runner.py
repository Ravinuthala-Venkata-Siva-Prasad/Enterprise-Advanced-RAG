from __future__ import annotations
import os, time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.retrieval.retrieval_pipeline import AdvancedRetrievalPipeline

SYSTEM_PROMPT = """You are an enterprise document assistant. Answer using only the supplied context. Do not use outside knowledge or invent facts. If the answer is missing, say: 'I cannot find this information in the provided documents.' Cite each source used by its exact filename in square brackets, e.g. [policy.pdf]. Do not cite files not supplied. Be concise."""

def _payload(result):
    doc = result.get("document", result) if isinstance(result, dict) else result
    if isinstance(doc, dict):
        text = doc.get("page_content") or doc.get("text") or ""
        meta = doc.get("metadata", {}) or {}
        source = meta.get("source", doc.get("source")) if isinstance(meta, dict) else None
    else:
        text = getattr(doc, "page_content", "")
        meta = getattr(doc, "metadata", {}) or {}
        source = meta.get("source") if isinstance(meta, dict) else None
    return str(text), str(source) if source else None

class RAGEvaluationRunner:
    """Runs the existing AdvancedRetrievalPipeline followed by Groq generation."""
    def __init__(self):
        load_dotenv()
        key = os.getenv("GROQ_API_KEY")
        if not key: raise RuntimeError("GROQ_API_KEY is missing from .env")
        self.retriever = AdvancedRetrievalPipeline()
        self.llm = ChatGroq(api_key=key, model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                            temperature=float(os.getenv("GROQ_TEMPERATURE", "0")))

    def run(self, question: str) -> dict:
        start = time.perf_counter()
        retrieved = self.retriever.retrieve(question)
        contexts, sources, blocks = [], [], []
        for i, result in enumerate(retrieved):
            content, source = _payload(result)
            if content.strip():
                contexts.append(content)
                label = source or f"retrieved_document_{i+1}"
                blocks.append(f"SOURCE: {label}\nCONTENT:\n{content}")
            if source: sources.append(source)
        context = "\n\n---\n\n".join(blocks) or "[No relevant document passages were retrieved.]"
        response = self.llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                                    HumanMessage(content=f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}")])
        answer = response.content if isinstance(response.content, str) else str(response.content)
        return {"question": question, "answer": answer, "contexts": contexts,
                "sources": sources, "latency_seconds": round(time.perf_counter()-start, 3),
                "retrieved_count": len(retrieved)}

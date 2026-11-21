
import uuid
import requests
import streamlit as st

from frontend.api_client import RAGAPIClient


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Enterprise Advanced RAG",
    page_icon="🧠",
    layout="wide"
)


# ==================================================
# 2. SESSION STATE INITIALIZATION
# ==================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_client" not in st.session_state:
    st.session_state.api_client = RAGAPIClient()

if "upload_success" not in st.session_state:
    st.session_state.upload_success = False

client = st.session_state.api_client


# ==================================================
# 3. HELPER FUNCTIONS
# ==================================================

def get_valid_sources(sources):
    """
    Filter invalid or empty source entries.
    Supports string sources and dictionaries.
    """

    valid_sources = []

    if not isinstance(sources, list):
        return valid_sources

    for source in sources:

        if isinstance(source, str):

            source = source.strip()

            if source:
                valid_sources.append(source)

        elif isinstance(source, dict):

            source_value = (
                source.get("source")
                or source.get("file_path")
                or source.get("filename")
                or source.get("content")
            )

            if source_value and str(source_value).strip():
                valid_sources.append(str(source_value).strip())

    # Remove duplicate sources while preserving order
    return list(dict.fromkeys(valid_sources))


def display_sources(sources):
    """
    Display retrieved sources as plain text.
    """

    valid_sources = get_valid_sources(sources)

    if valid_sources:

        with st.expander("📚 Retrieved Sources"):

            for index, source in enumerate(valid_sources, 1):
                st.text(f"{index}. {source}")

    else:

        st.caption("No source information was returned by the API.")


def display_search_query(rewritten_question):
    """
    Display rewritten retrieval query.
    """

    if rewritten_question:

        with st.expander("🔍 Search Query"):
            st.text(str(rewritten_question))


def display_index_statistics():
    """
    Display FAISS, BM25 and uploaded file statistics.
    """

    try:

        stats = client.get_document_stats()

        col1, col2 = st.columns(2)

        col1.metric(
            "FAISS Vectors",
            stats.get("faiss_vectors", 0)
        )

        col2.metric(
            "BM25 Documents",
            stats.get("bm25_documents", 0)
        )

        st.metric(
            "Uploaded Files",
            stats.get("uploaded_files", 0)
        )

    except requests.RequestException as exc:

        st.warning(
            f"Could not load document statistics: {exc}"
        )


# ==================================================
# 4. SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🧠 Enterprise RAG")

    st.caption(
        "Document Intelligence Assistant"
    )

    st.divider()

    # ----------------------------------------------
    # API CONNECTION STATUS
    # ----------------------------------------------

    st.subheader("🔌 Connection Status")

    api_connected = False

    try:

        health = client.health_check()

        if health.get("status") == "healthy":

            st.success("API Connected")

            api_connected = True

        else:

            st.warning("API is not healthy")

    except requests.RequestException:

        st.error("API Disconnected")

    st.divider()

    # ----------------------------------------------
    # INDEX STATISTICS
    # ----------------------------------------------

    st.subheader("📊 Index Statistics")

    if api_connected:

        display_index_statistics()

    else:

        st.info(
            "Connect the API to view index statistics."
        )

    st.divider()

    # ----------------------------------------------
    # DOCUMENT UPLOAD
    # ----------------------------------------------

    st.subheader("📁 Upload Documents")

    st.caption(
        "Supported formats: PDF, DOCX, TXT, CSV, HTML"
    )

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=[
            "pdf",
            "docx",
            "txt",
            "csv",
            "html",
            "htm"
        ],
        accept_multiple_files=False,
        key="document_upload"
    )

    if st.button(
        "Upload and Index",
        use_container_width=True,
        disabled=not api_connected or uploaded_file is None
    ):

        try:

            with st.spinner(
                "Processing document, generating embeddings "
                "and updating indexes..."
            ):

                result = client.upload_document(
                    uploaded_file
                )

            st.success(
                "Document uploaded and indexed successfully!"
            )

            st.write(
                "**Filename:**",
                result.get("filename", uploaded_file.name)
            )

            st.write(
                "**Chunks indexed:**",
                result.get("total_chunks", 0)
            )

            indexing = result.get("indexing", {})

            st.write(
                "**FAISS vectors:**",
                indexing.get("faiss_vectors", 0)
            )

            st.write(
                "**BM25 documents:**",
                indexing.get("bm25_documents", 0)
            )

            st.session_state.upload_success = True

            st.rerun()

        except requests.HTTPError as exc:

            status = (
                exc.response.status_code
                if exc.response is not None
                else "Unknown"
            )

            try:

                detail = exc.response.json().get(
                    "detail",
                    "Upload failed."
                )

            except Exception:

                detail = "Upload failed."

            if status == 400:

                st.warning(detail)

            elif status == 413:

                st.error(
                    "File exceeds the maximum upload size."
                )

            else:

                st.error(
                    f"Upload failed ({status}): {detail}"
                )

        except requests.Timeout:

            st.error(
                "Upload timed out. Check the API before retrying."
            )

        except requests.ConnectionError:

            st.error(
                "Cannot connect to the FastAPI backend."
            )

        except Exception as exc:

            st.error(
                f"Unexpected error: {exc}"
            )

    if st.session_state.upload_success:

        st.info(
            "Index statistics have been refreshed."
        )

        st.session_state.upload_success = False

    st.divider()

    # ----------------------------------------------
    # DOCUMENT MANAGEMENT
    # ----------------------------------------------

    st.subheader("🗂️ Document Management")

    if api_connected:
        try:
            documents = client.get_documents()
            st.caption(f"Registered documents: {len(documents)}")

            if documents:
                display_rows = [
                    {
                        "Filename": item.get("filename", ""),
                        "Document ID": item.get("document_id", ""),
                        "Chunks": item.get("total_chunks", 0),
                        "Status": item.get("status", "unknown"),
                        "Uploaded": item.get("uploaded_at", "")[:19].replace("T", " "),
                    }
                    for item in documents
                ]
                st.dataframe(display_rows, use_container_width=True, hide_index=True)

                document_options = {
                    f"{item.get('filename')} ({item.get('document_id')})": item.get("document_id")
                    for item in documents
                }
                selected_label = st.selectbox(
                    "Select a document to delete",
                    options=list(document_options.keys()),
                    key="delete_document_selection",
                )
                selected_document_id = document_options[selected_label]

                if st.button(
                    "Prepare document deletion",
                    use_container_width=True,
                ):
                    st.session_state["pending_delete_document_id"] = selected_document_id
                    st.rerun()

                if st.session_state.get("pending_delete_document_id") == selected_document_id:
                    st.warning(
                        "This permanently removes the uploaded file and its indexed chunks from FAISS and BM25."
                    )
                    confirm_col, cancel_col = st.columns(2)
                    if confirm_col.button("Confirm deletion", type="primary", use_container_width=True):
                        try:
                            deleted = client.delete_document(selected_document_id)
                            st.session_state.pop("pending_delete_document_id", None)
                            st.success(
                                f"Deleted {deleted.get('filename')}. "
                                f"Removed {deleted.get('removed_chunks', 0)} chunks."
                            )
                            st.rerun()
                        except requests.HTTPError as exc:
                            detail = "Document deletion failed."
                            try:
                                detail = exc.response.json().get("detail", detail)
                            except Exception:
                                pass
                            st.error(detail)
                    if cancel_col.button("Cancel", use_container_width=True):
                        st.session_state.pop("pending_delete_document_id", None)
                        st.rerun()
            else:
                st.info("No documents are registered yet. Upload a document to begin.")
        except requests.RequestException as exc:
            st.warning(f"Could not load document history: {exc}")
    else:
        st.info("Connect the API to manage documents.")

    st.divider()

    # ----------------------------------------------
    # CONVERSATION MANAGEMENT
    # ----------------------------------------------

    st.subheader("💬 Conversation")

    st.caption(
        f"Session ID: {st.session_state.session_id}"
    )

    if st.button(
        "New Conversation",
        use_container_width=True
    ):

        try:

            client.clear_session(
                st.session_state.session_id
            )

        except requests.RequestException:

            st.warning(
                "Could not clear server history. "
                "A new local session will be created."
            )

        st.session_state.session_id = str(uuid.uuid4())

        st.session_state.messages = []

        st.rerun()

    st.divider()

    # ----------------------------------------------
    # PROJECT FEATURES
    # ----------------------------------------------

    st.subheader("🚀 Project Features")

    st.markdown("""
    - Hybrid Retrieval
    - FAISS Vector Database
    - BM25 Sparse Retrieval
    - Cross-Encoder Reranking
    - Groq LLM Generation
    - Conversational Memory
    - Document Upload
    - Incremental Indexing
    - Retrieval Evaluation
    - Guardrails
    """)


# ==================================================
# 5. MAIN HEADER
# ==================================================

st.title(
    "🧠 Enterprise Advanced RAG"
)

st.write(
    "Ask questions about your enterprise documents "
    "and receive context-grounded answers."
)

st.divider()


# ==================================================
# 6. DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        if message["role"] == "assistant":

            sources = message.get(
                "sources",
                []
            )

            display_sources(sources)

            rewritten_question = message.get(
                "rewritten_question"
            )

            display_search_query(
                rewritten_question
            )


# ==================================================
# 7. CHAT INPUT
# ==================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# ==================================================
# 8. PROCESS USER QUESTION
# ==================================================

if question:

    # ----------------------------------------------
    # SAVE USER MESSAGE
    # ----------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })

    with st.chat_message("user"):

        st.markdown(question)

    # ----------------------------------------------
    # GENERATE ASSISTANT RESPONSE
    # ----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                # Call FastAPI backend
                result = client.ask_question(

                    question=question,

                    session_id=st.session_state.session_id

                )

                # ----------------------------------
                # EXTRACT API RESPONSE
                # ----------------------------------

                answer = result.get(
                    "answer",
                    "The API returned no answer."
                )

                sources = result.get(
                    "sources",
                    []
                )

                rewritten_question = result.get(
                    "rewritten_question"
                )

                # ----------------------------------
                # DISPLAY ANSWER
                # ----------------------------------

                st.markdown(answer)

                # ----------------------------------
                # DISPLAY RETRIEVED SOURCES
                # ----------------------------------

                display_sources(sources)

                # ----------------------------------
                # DISPLAY REWRITTEN QUERY
                # ----------------------------------

                display_search_query(
                    rewritten_question
                )

                # ----------------------------------
                # SAVE ASSISTANT MESSAGE
                # ----------------------------------

                valid_sources = get_valid_sources(
                    sources
                )

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer,

                    "sources": valid_sources,

                    "rewritten_question": rewritten_question

                })

            # --------------------------------------
            # ERROR HANDLING
            # --------------------------------------

            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. "
                    "Try again or increase the API timeout."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to the FastAPI backend. "
                    "Check that the API server is running."
                )

            except requests.exceptions.HTTPError as exc:

                status_code = (

                    exc.response.status_code

                    if exc.response is not None

                    else "unknown"

                )

                st.error(
                    f"API request failed. HTTP status: {status_code}"
                )

            except Exception as exc:

                st.error(
                    f"Unexpected error: {exc}"
                )

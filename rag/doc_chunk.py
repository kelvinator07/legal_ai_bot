import logging
import os
import warnings

warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*migration guide.*")

# Suppress the harmless "UNEXPECTED position_ids" load report from transformers
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

logger = logging.getLogger(__name__)

_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(_RAG_DIR, "knowledge-base")
VECTOR_DB_DIR = os.path.join(_RAG_DIR, "chat_vector_db")

def model_name() -> str:
    return os.getenv("LEGAL_AGENT_MODEL", "gpt-4.1-mini")

MODEL = model_name()

_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _db_exists() -> bool:
    """Check whether a persisted Chroma collection already has documents."""
    if not os.path.isdir(VECTOR_DB_DIR):
        return False
    try:
        store = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=_embeddings)
        return store._collection.count() > 0
    except Exception:
        return False


if _db_exists():
    logger.info("[RAG] Existing vector store found at %s — reusing it", VECTOR_DB_DIR)
    _vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=_embeddings)
    logger.info("[RAG] Loaded vectorstore with %d documents", _vectorstore._collection.count())
else:
    logger.info("[RAG] No existing vector store found — building from knowledge base at %s", KNOWLEDGE_BASE_DIR)
    _loader = DirectoryLoader(
        KNOWLEDGE_BASE_DIR,
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    _documents = _loader.load()
    logger.info("[RAG] Loaded %d documents", len(_documents))

    _text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=[
            "\n## ",   # Major headings (Parts/Chapters)
            "\n### ",  # Sub-headings (Sections)
            "\n\n",
            "\n",
            ". ",
            " ",
        ]
    )
    _chunks = _text_splitter.split_documents(_documents)
    logger.info("[RAG] Split into %d chunks (chunk_size=%d, overlap=%d)", len(_chunks), 1500, 300)

    _vectorstore = Chroma.from_documents(
        documents=_chunks, embedding=_embeddings, persist_directory=VECTOR_DB_DIR
    )
    logger.info("[RAG] Vectorstore created with %d documents", _vectorstore._collection.count())
    del _loader, _documents, _text_splitter, _chunks

_llm = ChatOpenAI(temperature=0.7, model_name=MODEL, api_key=os.getenv("OPENAI_API_KEY"))

_memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
    output_key='answer'
)

# Chroma scores are not normalized to [0, 1]; similarity_score_threshold mis-filters.
_retriever = _vectorstore.as_retriever(search_kwargs={"k": 8})

_conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=_llm,
    retriever=_retriever,
    memory=_memory,
    return_source_documents=True,
    verbose=False
)

logger.info("[RAG] ConversationalRetrievalChain initialized")

SYSTEM_PROMPT_TEMPLATE = """Summarize the following retrieved excerpts for another model that will answer the user.

- Use ONLY the Context. Do not cite statutes, sections, or holdings that are not in the Context.
- If the Context is empty or irrelevant to the question, reply exactly with:
  "The documents I have access to do not cover this area of law."
- Be concise: relevant themes or acts, short quotes or tight paraphrases tied to the Context, then bullet next steps only if the text supports them. Omit outcome/timeline unless explicitly in the Context.
- Do not add a legal-disclaimer; the outer assistant handles user-facing tone and disclaimers.

Context:
{context}"""


# --- RAG path guard rails ---
MAX_RAG_QUESTION_CHARS = 1_000
MAX_RAG_FINAL_ANSWER_CHARS = 1_000

_EMPTY_QUESTION_REPLY = (
    "Please ask a specific question about Nigerian law so I can search the knowledge base."
)
_NO_RETRIEVAL_CONTEXT_REPLY = (
    "The documents I have access to do not cover this area of law."
)
_RAG_PROCESSING_ERROR_REPLY = (
    "Something went wrong while searching the legal knowledge base. Please try again."
)


def _sanitize_rag_text(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.replace("\x00", "").strip()


def _gradio_messages_to_lc_history(
    messages: list[dict | BaseMessage] | None,
) -> list[tuple[str, str]]:
    """Convert Gradio 6+ {'role','content'} dicts to (user, assistant) tuples for ConversationalRetrievalChain."""
    if not messages:
        return []
    turns: list[tuple[str, str]] = []
    pending_user: str | None = None
    for msg in messages:
        if isinstance(msg, BaseMessage):
            if msg.type == "human":
                pending_user = str(msg.content)
            elif msg.type == "ai" and pending_user is not None:
                turns.append((pending_user, str(msg.content)))
                pending_user = None
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role is None or content is None:
            continue
        text = content if isinstance(content, str) else str(content)
        if role == "user":
            pending_user = text
        elif role == "assistant" and pending_user is not None:
            turns.append((pending_user, text))
            pending_user = None
    return turns


def rag_query_answer(
    question: str,
    chat_history: list[dict | BaseMessage] | None = None,
):
    q = _sanitize_rag_text(question)
    if not q:
        return _EMPTY_QUESTION_REPLY
    logger.info("[GUARDRAILS] Sanitizing RAG input (length=%d)", len(q))
    if len(q) > MAX_RAG_QUESTION_CHARS:
        q = q[:MAX_RAG_QUESTION_CHARS]
        logger.info("[GUARDRAILS] Input capped at %d chars", MAX_RAG_QUESTION_CHARS)

    lc_history = _gradio_messages_to_lc_history(chat_history)
    try:
        logger.info("[RAG] Invoking conversation chain")
        out = _conversation_chain.invoke({"question": q, "chat_history": lc_history})
    except Exception:
        logger.exception("[RAG] Conversation chain failed")
        return _RAG_PROCESSING_ERROR_REPLY

    source_docs = out.get("source_documents") or []
    logger.info("[RAG] Retrieved %d source documents", len(source_docs))
    context_parts: list[str] = []
    for d in source_docs:
        page = getattr(d, "page_content", None)
        if isinstance(page, str) and page.strip():
            context_parts.append(page)
    context = "\n\n".join(context_parts).strip()
    if not context:
        return _NO_RETRIEVAL_CONTEXT_REPLY

    draft = (out.get("answer") or "").strip()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    if draft:
        system_prompt += (
            "\n\nFirst-pass retrieval draft (polish using the same constraints; "
            "do not add facts beyond the Context):\n\n"
            f"{draft}"
        )
    try:
        logger.info("[RAG] Running second-pass LLM summarization")
        response = _llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=q)]
        )
    except Exception:
        logger.exception("[RAG] Second-pass summarization failed")
        return draft if draft else _RAG_PROCESSING_ERROR_REPLY

    final = response.content
    if final is None:
        return draft if draft else _NO_RETRIEVAL_CONTEXT_REPLY
    if not isinstance(final, str):
        final = str(final)
    final = final.strip()
    if not final:
        return draft if draft else _NO_RETRIEVAL_CONTEXT_REPLY
    if len(final) > MAX_RAG_FINAL_ANSWER_CHARS:
        final = final[: MAX_RAG_FINAL_ANSWER_CHARS - 1].rstrip() + "…"
        logger.info("[GUARDRAILS] Final answer capped at %d chars", MAX_RAG_FINAL_ANSWER_CHARS)
    return final

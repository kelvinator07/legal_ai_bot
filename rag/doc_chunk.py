import os
import glob
import tiktoken
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory


load_dotenv(override=True)

_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(_RAG_DIR, "knowledge-base")
VECTOR_DB_DIR = os.path.join(_RAG_DIR, "chat_vector_db")

MODEL = "gpt-4o-mini"
openai_api_key = os.getenv('OPENROUTER_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

# How many characters in all the documents?
knowledge_base_path = os.path.join(KNOWLEDGE_BASE_DIR, "*")
files = glob.glob(knowledge_base_path, recursive=True)
print(f"Found {len(files)} files in the knowledge base")

entire_knowledge_base = ""

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        entire_knowledge_base += f.read()
        entire_knowledge_base += "\n\n"

print(f"Total characters in knowledge base: {len(entire_knowledge_base):,}")


# How many tokens in all the documents?
encoding = tiktoken.encoding_for_model(MODEL)
tokens = encoding.encode(entire_knowledge_base)
token_count = len(tokens)
print(f"Total tokens for {MODEL}: {token_count:,}")


# Load in everything in the knowledgebase using LangChain's loaders
documents = []
loader = DirectoryLoader(
    KNOWLEDGE_BASE_DIR,
    glob="*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
folder_docs = loader.load()
for doc in folder_docs:
    documents.append(doc)    

print(f"Loaded {len(documents)} documents")


# Divide into chunks using the RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,          # Larger chunks preserve more section context
    chunk_overlap=300,         # More overlap to avoid splitting mid-provision
    separators=[
        "\n## ",              # Major headings (Parts/Chapters)
        "\n### ",             # Sub-headings (Sections)
        "\n\n",               # Paragraph breaks
        "\n",                 # Line breaks
        ". ",                 # Sentences
        " ",                  # Words
    ]
)
chunks = text_splitter.split_documents(documents)

print(f"Documents divided into {len(chunks)} chunks")


# Pick an embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# create a vector store
if os.path.exists(VECTOR_DB_DIR):
    print("Deleting old vector store...")
    Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(
    documents=chunks, embedding=embeddings, persist_directory=VECTOR_DB_DIR
)
collection = vectorstore._collection
count = collection.count()
print(f"Vectorstore created with {count} documents")

# investigate the vectors
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")

llm = ChatOpenAI(temperature=0.7, model_name=MODEL, base_url="https://openrouter.ai/api/v1", api_key=openai_api_key)

memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
    output_key='answer'
)

# Chroma scores are not normalized to [0, 1]; similarity_score_threshold mis-filters.
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=False
)


SYSTEM_PROMPT_TEMPLATE = """
You are a Nigerian legal information assistant. You help users understand their legal rights 
under Nigerian law by citing specific statutes, sections, and case precedents.

RULES:
1. ONLY cite laws, sections, and cases that appear in the provided context below.
2. Structure your response as: (a) applicable law, (b) relevant sections with quotes, 
   (c) practical next steps the user can take. (d) Expected outcome (e) Timeline
3. If the context does not contain relevant law for the question, say: 
   "The documents I have access to do not cover this area of law."
4. Always end with: "Disclaimer: This is legal information, not legal advice. 
   Please consult a qualified Nigerian lawyer for advice specific to your situation."
Keep it short, concise, and straight to the point.

Context:
{context}
"""


def expand_query(question: str) -> str:
    """Use LLM to add legal terms to the query for better retrieval."""
    expansion_prompt = f"""Given this user question about Nigerian law, 
    rewrite it to include relevant legal terminology that would help 
    find matching statutes. Keep it concise.
    
    Question: {question}
    Rewritten query:"""
    response = llm.invoke([HumanMessage(content=expansion_prompt)])
    return response.content


llm_question = expand_query("I bought a phone in Nigeria that broke after 1 week. Shop won't refund. What are my rights?")
llm_question


def _gradio_messages_to_lc_history(
    messages: list[dict | BaseMessage] | None,
) -> list[tuple[str, str]]:
    """Gradio 6+ uses {'role','content'}; ConversationalRetrievalChain expects (user, assistant) tuples per turn."""
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
    lc_history = _gradio_messages_to_lc_history(chat_history)
    out = conversation_chain.invoke({"question": question, "chat_history": lc_history})
    source_docs = out.get("source_documents") or []
    context = "\n\n".join(d.page_content for d in source_docs)
    draft = (out.get("answer") or "").strip()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    if draft:
        system_prompt += (
            "\n\nFirst-pass retrieval answer (polish and follow the rules above; "
            "do not add facts beyond the context):\n\n"
            f"{draft}"
        )
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    return response.content

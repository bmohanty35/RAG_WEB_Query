import os
from dotenv import load_dotenv
import hashlib

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# ✅ FIX: correct env variable
os.environ["GOOGLE_API_KEY"] = os.getenv("gemini_key")


# Initialize LLM
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# Generate unique vector DB per URL
def get_persist_dir(url):
    return f"./vectordb_{hashlib.md5(url.encode()).hexdigest()}"


# Load website data
def load_data(url):
    loader = WebBaseLoader(url)
    data = loader.load()

    if not data:
        raise ValueError("No data loaded from URL")

    return data[0].page_content


# Split text
def split_text(text):
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.create_documents([text])


# ✅ FIX: pass url here
def create_vector_store(docs, url):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=get_persist_dir(url)
    )

    return vectordb


# Build RAG chain
def build_chain(vectordb):
    retriever = vectordb.as_retriever(search_type="similarity")

    sys_prompt = """
    You are a helpful assistant.
    Answer ONLY from the given context.
    If answer not found, say:
    "Not found in the provided content."
    """

    human_prompt = """
    Context:
    {context}

    Question:
    {question}
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(sys_prompt),
        HumanMessagePromptTemplate.from_template(human_prompt)
    ])

    llm = get_llm()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# Main RAG function
def run_rag(url, query):
    text = load_data(url)
    docs = split_text(text)

    # ✅ FIX: pass url here
    vectordb = create_vector_store(docs, url)

    chain = build_chain(vectordb)

    answer = chain.invoke(query)
    return answer
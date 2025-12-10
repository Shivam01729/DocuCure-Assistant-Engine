from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from .config import AppConfig
from .prompts import SYSTEM_PROMPT
from .ingestion import get_embedding_model

def get_vector_store():
    """
    Connects to an existing Pinecone index and returns the VectorStore object.
    """
    embeddings = get_embedding_model()
    
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=AppConfig.INDEX_NAME,
        embedding=embeddings
    )
    return docsearch

def get_rag_chain():
    """
    Constructs and returns the RAG execution chain.
    """
    docsearch = get_vector_store()
    
    # Configure retriever
    retriever = docsearch.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 3}
    )
    
    # Initialize LLM
    llm = ChatOpenAI(model=AppConfig.CHAT_MODEL, api_key=AppConfig.OPENAI_API_KEY)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )
    
    # Build chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

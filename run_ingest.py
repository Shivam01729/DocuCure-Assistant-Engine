from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.medical_bot.config import AppConfig
from src.medical_bot.ingestion import load_source_documents, split_documents, get_embedding_model
from src.medical_bot.formatting import sanitize_document_metadata

def run_ingestion():
    print("Starting ingestion process...")
    
    # 1. Load Data
    print("Loading PDFs...")
    raw_docs = load_source_documents(data_dir='data/')
    
    # 2. Process Data (Clean metadata)
    print("Sanitizing metadata...")
    clean_docs = sanitize_document_metadata(raw_docs)
    
    # 3. Split Text
    print("Splitting documents...")
    text_chunks = split_documents(clean_docs)
    print(f"Generated {len(text_chunks)} text chunks.")
    
    # 4. Initialize Embeddings
    print("Initializing embedding model...")
    embeddings = get_embedding_model()
    
    # 5. Initialize Pinecone
    print(f"Connecting to Pinecone index '{AppConfig.INDEX_NAME}'...")
    pc = Pinecone(api_key=AppConfig.PINECONE_API_KEY)
    
    # Check/Create Index
    if not pc.has_index(AppConfig.INDEX_NAME):
        print(f"Index '{AppConfig.INDEX_NAME}' not found. Creating...")
        pc.create_index(
            name=AppConfig.INDEX_NAME,
            dimension=AppConfig.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=AppConfig.AWS_REGION),
        )
    
    # 6. Upsert to Vector Store
    print("Upserting vectors to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=AppConfig.INDEX_NAME,
        embedding=embeddings, 
    )
    print("Ingestion complete!")

if __name__ == "__main__":
    run_ingestion()

from typing import List
from langchain.schema import Document

def sanitize_document_metadata(original_docs: List[Document]) -> List[Document]:
    """
    Refines a list of Documents to keep only essential metadata (source)
    and the original content, decoupling from unnecessary details.
    
    Args:
        original_docs: List of LangChain Documents with potentially cluttered metadata.
        
    Returns:
        A list of clean Document objects.
    """
    clean_docs: List[Document] = []
    for doc in original_docs:
        # Extract only the 'source' from metadata, default to 'unknown' if missing
        source_path = doc.metadata.get("source", "unknown")
        
        new_doc = Document(
            page_content=doc.page_content,
            metadata={"source": source_path}
        )
        clean_docs.append(new_doc)
    return clean_docs

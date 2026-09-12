import os
from pinecone import Pinecone

# Initialize the Pinecone client using environment variables
# Ensure PINECONE_API_KEY and PINECONE_INDEX_NAME are set in your .env file
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME", "researchsynth-tensions")
index = pc.Index(index_name)

def clear_namespace(namespace: str) -> None:
    """
    Wipes all vectors in a specific namespace to provide a clean slate for a new run.
    """
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        # If the namespace doesn't exist yet (very first time running), ignore the error
        pass

def upsert_findings(records: list[dict], namespace: str) -> None:
    """
    Stores vector embeddings and their associated metadata in Pinecone.
    
    Args:
        records: A list of dictionaries containing 'id', 'values' (3072 dims), and 'metadata'.
        namespace: The active namespace string isolating the current run.
    """
    if records:
        index.upsert(vectors=records, namespace=namespace)

def search_within_subtopic(query_vector: list[float], subtopic: str, namespace: str, top_k: int = 5) -> dict:
    """
    Searches for the closest matching claims strictly within a specific subtopic.
    """
    return index.query(
        vector=query_vector,
        top_k=top_k,
        filter={"subtopics": {"$in": [subtopic]}},
        include_metadata=True,
        namespace=namespace
    )
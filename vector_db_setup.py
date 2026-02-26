import os
import json
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration ---
# Use an open-source, local embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "vector_store"
COLLECTION_NAME = "bank_knowledge_base"
INPUT_FILE = "preprocessed_data/bank_data_advanced.jsonl" # Use the deduplicated JSONL

def setup_vector_db():
    """Initializes ChromaDB and indexes the preprocessed bank data."""
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file {INPUT_FILE} not found. Run preprocessing first.")
        return

    # 1. Initialize ChromaDB client with persistence
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

    # 2. Setup embedding function
    print(f"📥 Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    # 3. Create or get collection (Delete existing if found for clean re-index)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"🗑️ Deleted existing collection '{COLLECTION_NAME}' for fresh start.")
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"} # Use cosine similarity
    )

    # 4. Load data from JSONL
    print(f"📄 Reading records from {INPUT_FILE}...")
    documents = []
    metadatas = []
    ids = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            # The 'content' field is what we embed
            documents.append(record["content"])
            # 'metadata' and 'structured_data' are stored for retrieval
            # We combine them into a single metadata dict
            meta = record["metadata"]
            # Add some structured data keys to metadata for filtering if needed
            # (ChromaDB metadata must be simple types: str, int, float, bool)
            metadatas.append(meta)
            ids.append(record["hash_id"])

    # 5. Add records to the collection in batches (Chroma has limits per add)
    batch_size = 500
    print(f"🏗️ Indexing {len(documents)} records in batches of {batch_size}...")
    
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        print(f"✅ Indexed records {i} to {end}")

    print(f"🚀 Vector DB setup complete! Stored in '{PERSIST_DIRECTORY}'")

if __name__ == "__main__":
    setup_vector_db()

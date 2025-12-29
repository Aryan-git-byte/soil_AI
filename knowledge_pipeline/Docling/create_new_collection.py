from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(url="http://localhost:6333")

COLLECTION = "farmbot_knowledge_v2"
VECTOR_SIZE = 384  # bge-small-en-v1.5

# Delete if exists (safe)
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(
        size=VECTOR_SIZE,
        distance=models.Distance.COSINE
    )
)

print("✅ Fresh collection created:", COLLECTION)

import os
from pathlib import Path
from typing import List, Dict
import torch
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from tqdm import tqdm
import hashlib

class PDFEmbeddingSystem:
    def __init__(self, pdf_dir: str = "./pdf", collection_name: str = "pdf_documents"):
        self.pdf_dir = Path(pdf_dir)
        self.collection_name = collection_name
        
        # Initialize Docling converter
        print("Initializing Docling...")
        self.converter = DocumentConverter()
        
        # Initialize embedding model with GPU support
        print("Loading embedding model on GPU...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=self.device)
        self.embedding_dim = 768
        
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Initialize Qdrant client (server mode)
        print("Connecting to Qdrant server...")
        self.client = QdrantClient(url="http://localhost:6333")
        
        # Create collection if it doesn't exist
        self._setup_collection()
    
    def _setup_collection(self):
        """Setup Qdrant collection with vector and keyword search"""
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)
        
        if not collection_exists:
            print(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            
            # Create payload index for keyword search
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="text",
                field_schema="text"
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="filename",
                field_schema="keyword"
            )
        else:
            print(f"Collection {self.collection_name} already exists")
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def _get_chunk_id(self, filename: str, chunk_idx: int) -> str:
        """Generate unique ID for chunk"""
        hash_input = f"{filename}_{chunk_idx}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def process_pdf(self, pdf_path: Path) -> List[Dict]:
        """Process a single PDF with Docling"""
        print(f"Processing: {pdf_path.name}")
        
        try:
            # Convert PDF using Docling
            result = self.converter.convert(str(pdf_path))
            
            # Extract text from document
            full_text = result.document.export_to_markdown()
            
            # Chunk the text
            chunks = self._chunk_text(full_text)
            
            # Prepare chunk data
            chunk_data = []
            for idx, chunk in enumerate(chunks):
                chunk_data.append({
                    "text": chunk,
                    "filename": pdf_path.name,
                    "chunk_idx": idx,
                    "total_chunks": len(chunks)
                })
            
            return chunk_data
            
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {str(e)}")
            return []
    
    def embed_and_store(self, chunk_data: List[Dict], batch_size: int = 32):
        """Embed chunks and store in Qdrant"""
        if not chunk_data:
            return
        
        texts = [c["text"] for c in chunk_data]
        
        # Generate embeddings in batches
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Prepare points for Qdrant
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunk_data, embeddings)):
            point_id = self._get_chunk_id(chunk["filename"], chunk["chunk_idx"])
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "text": chunk["text"],
                    "filename": chunk["filename"],
                    "chunk_idx": chunk["chunk_idx"],
                    "total_chunks": chunk["total_chunks"]
                }
            ))
        
        # Upload to Qdrant
        print(f"Uploading {len(points)} points to Qdrant...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def process_all_pdfs(self):
        """Process all PDFs in the directory"""
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            chunk_data = self.process_pdf(pdf_path)
            if chunk_data:
                self.embed_and_store(chunk_data)
        
        print("\n✓ All PDFs processed successfully!")
        self._print_stats()
    
    def _print_stats(self):
        """Print collection statistics"""
        collection_info = self.client.get_collection(self.collection_name)
        print(f"\nCollection Statistics:")
        print(f"  Total points: {collection_info.points_count}")
        print(f"  Vector dimension: {self.embedding_dim}")
    
    def search_vector(self, query: str, limit: int = 5):
        """Vector similarity search"""
        query_embedding = self.model.encode(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit
        )
        
        return results
    
    def search_keyword(self, keyword: str, limit: int = 5):
        """Keyword search in text payload"""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="text",
                        match=MatchValue(value=keyword)
                    )
                ]
            ),
            limit=limit
        )
        
        return results[0]  # Returns (points, next_offset)
    
    def hybrid_search(self, query: str, limit: int = 5):
        """Combine vector and keyword search"""
        print(f"\nSearching for: '{query}'")
        print("=" * 60)
        
        # Vector search
        print("\n📊 Vector Search Results:")
        vector_results = self.search_vector(query, limit)
        
        for idx, hit in enumerate(vector_results, 1):
            print(f"\n{idx}. Score: {hit.score:.4f}")
            print(f"   File: {hit.payload['filename']}")
            print(f"   Chunk: {hit.payload['chunk_idx'] + 1}/{hit.payload['total_chunks']}")
            print(f"   Text: {hit.payload['text'][:200]}...")
        
        return vector_results


# Main execution
if __name__ == "__main__":
    # Initialize the system
    system = PDFEmbeddingSystem(pdf_dir="./pdf")
    
    # Process all PDFs
    system.process_all_pdfs()
    
    # Example searches
    print("\n" + "=" * 60)
    print("EXAMPLE SEARCHES")
    print("=" * 60)
    
    # Try a sample search
    sample_queries = [
        "machine learning algorithms",
        "data processing pipeline",
        "neural networks"
    ]
    
    for query in sample_queries:
        system.hybrid_search(query, limit=3)
        print("\n")
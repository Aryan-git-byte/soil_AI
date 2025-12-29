import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from typing import List, Dict
import sys

class PDFSearcher:
    def __init__(self, collection_name: str = "pdf_documents"):
        self.collection_name = collection_name
        
        # Initialize embedding model with GPU support
        print("Loading embedding model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=self.device)
        
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Connect to Qdrant
        print("Connecting to Qdrant server...")
        self.client = QdrantClient(url="http://localhost:6333")
        
        # Verify collection exists
        self._verify_collection()
    
    def _verify_collection(self):
        """Check if collection exists"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            print(f"✓ Connected to collection: {self.collection_name}")
            print(f"  Total documents: {collection_info.points_count}")
        except Exception as e:
            print(f"✗ Error: Collection '{self.collection_name}' not found!")
            print(f"  Make sure you've run the indexing script first.")
            sys.exit(1)
    
    def vector_search(self, query: str, limit: int = 5, score_threshold: float = 0.0):
        """Semantic vector search"""
        print(f"\n🔍 Searching: '{query}'")
        print("=" * 80)
        
        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            score_threshold=score_threshold
        )
        
        if not results:
            print("No results found.")
            return []
        
        # Display results
        print(f"\n📊 Found {len(results)} results:\n")
        for idx, hit in enumerate(results, 1):
            print(f"{idx}. Score: {hit.score:.4f}")
            print(f"   📄 File: {hit.payload['filename']}")
            print(f"   📍 Chunk: {hit.payload['chunk_idx'] + 1}/{hit.payload['total_chunks']}")
            print(f"   📝 Text: {hit.payload['text'][:300]}...")
            print()
        
        return results
    
    def keyword_search(self, keyword: str, limit: int = 10):
        """Full-text keyword search"""
        print(f"\n🔍 Keyword Search: '{keyword}'")
        print("=" * 80)
        
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="text",
                        match=MatchText(text=keyword)
                    )
                ]
            ),
            limit=limit
        )
        
        if not results:
            print("No results found.")
            return []
        
        # Display results
        print(f"\n📊 Found {len(results)} results:\n")
        for idx, point in enumerate(results, 1):
            print(f"{idx}. 📄 File: {point.payload['filename']}")
            print(f"   📍 Chunk: {point.payload['chunk_idx'] + 1}/{point.payload['total_chunks']}")
            print(f"   📝 Text: {point.payload['text'][:300]}...")
            print()
        
        return results
    
    def search_by_filename(self, filename: str, limit: int = 10):
        """Search within a specific file"""
        print(f"\n🔍 Searching in file: '{filename}'")
        print("=" * 80)
        
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="filename",
                        match=MatchText(text=filename)
                    )
                ]
            ),
            limit=limit
        )
        
        if not results:
            print(f"No chunks found for file: {filename}")
            return []
        
        # Sort by chunk index
        results = sorted(results, key=lambda x: x.payload['chunk_idx'])
        
        print(f"\n📊 Found {len(results)} chunks:\n")
        for point in results:
            print(f"Chunk {point.payload['chunk_idx'] + 1}/{point.payload['total_chunks']}:")
            print(f"{point.payload['text'][:300]}...")
            print()
        
        return results
    
    def hybrid_search(self, query: str, limit: int = 5):
        """Combine vector and keyword search with re-ranking"""
        print(f"\n🔍 Hybrid Search: '{query}'")
        print("=" * 80)
        
        # Vector search
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        vector_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit * 2  # Get more results for re-ranking
        )
        
        # Keyword search
        keyword_results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="text",
                        match=MatchText(text=query)
                    )
                ]
            ),
            limit=limit * 2
        )
        
        # Combine and deduplicate results
        combined = {}
        
        # Add vector results with scores
        for hit in vector_results:
            point_id = hit.id
            combined[point_id] = {
                'payload': hit.payload,
                'vector_score': hit.score,
                'keyword_match': False,
                'combined_score': hit.score
            }
        
        # Boost scores for keyword matches
        keyword_ids = {point.id for point in keyword_results}
        for point_id in keyword_ids:
            if point_id in combined:
                combined[point_id]['keyword_match'] = True
                combined[point_id]['combined_score'] *= 1.5  # Boost by 50%
        
        # Sort by combined score
        sorted_results = sorted(
            combined.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )[:limit]
        
        if not sorted_results:
            print("No results found.")
            return []
        
        # Display results
        print(f"\n📊 Found {len(sorted_results)} results:\n")
        for idx, (point_id, data) in enumerate(sorted_results, 1):
            keyword_badge = "🔑" if data['keyword_match'] else ""
            print(f"{idx}. Score: {data['combined_score']:.4f} {keyword_badge}")
            print(f"   📄 File: {data['payload']['filename']}")
            print(f"   📍 Chunk: {data['payload']['chunk_idx'] + 1}/{data['payload']['total_chunks']}")
            print(f"   📝 Text: {data['payload']['text'][:300]}...")
            print()
        
        return sorted_results
    
    def list_all_files(self):
        """List all indexed files"""
        print("\n📚 Indexed Files:")
        print("=" * 80)
        
        # Get all points
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000  # Get all
        )
        
        # Extract unique filenames
        files = {}
        for point in results:
            filename = point.payload['filename']
            if filename not in files:
                files[filename] = {
                    'chunks': 0,
                    'total_chunks': point.payload['total_chunks']
                }
            files[filename]['chunks'] += 1
        
        print(f"\nTotal files: {len(files)}\n")
        for idx, (filename, info) in enumerate(sorted(files.items()), 1):
            print(f"{idx}. {filename}")
            print(f"   Chunks: {info['chunks']}/{info['total_chunks']}")
        print()
        
        return files
    
    def interactive_mode(self):
        """Interactive search mode"""
        print("\n" + "=" * 80)
        print("📖 PDF SEARCH SYSTEM - Interactive Mode")
        print("=" * 80)
        print("\nCommands:")
        print("  1. search <query>      - Vector search")
        print("  2. keyword <word>      - Keyword search")
        print("  3. file <filename>     - Search in specific file")
        print("  4. hybrid <query>      - Hybrid search (vector + keyword)")
        print("  5. list                - List all indexed files")
        print("  6. exit                - Exit")
        print()
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if not command:
                    continue
                
                if command.lower() == 'exit':
                    print("Goodbye!")
                    break
                
                if command.lower() == 'list':
                    self.list_all_files()
                    continue
                
                parts = command.split(maxsplit=1)
                if len(parts) < 2:
                    print("Invalid command. Use: <command> <query>")
                    continue
                
                cmd, query = parts
                
                if cmd.lower() == 'search':
                    self.vector_search(query, limit=5)
                elif cmd.lower() == 'keyword':
                    self.keyword_search(query, limit=5)
                elif cmd.lower() == 'file':
                    self.search_by_filename(query, limit=10)
                elif cmd.lower() == 'hybrid':
                    self.hybrid_search(query, limit=5)
                else:
                    print(f"Unknown command: {cmd}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")


# Main execution
if __name__ == "__main__":
    # Initialize searcher
    searcher = PDFSearcher(collection_name="pdf_documents")
    
    # Check if command line arguments provided
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print("Running quick search...")
        searcher.hybrid_search(query, limit=5)
    else:
        # Start interactive mode
        searcher.interactive_mode()
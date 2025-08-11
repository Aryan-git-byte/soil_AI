# app/models/knowledge_base.py
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

class KnowledgeBase:
    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(exist_ok=True)
        
        # Initialize sentence transformer for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Knowledge storage
        self.documents = {}
        self.embeddings = {}
        self.metadata = {}
        
        # Initialize with default agricultural knowledge
        self._initialize_default_knowledge()
        
    def _initialize_default_knowledge(self):
        """Initialize with essential agricultural knowledge"""
        default_knowledge = {
            "soil_ph": {
                "content": """Soil pH is crucial for crop growth. Most crops prefer pH 6.0-7.0:
                - pH < 6.0: Acidic soil, may need lime application
                - pH 6.0-7.0: Optimal range for most crops
                - pH > 7.0: Alkaline soil, may need sulfur or organic matter
                
                pH affects nutrient availability:
                - Low pH: Iron, manganese available; phosphorus, calcium limited
                - High pH: Phosphorus, iron, manganese limited; calcium available
                
                Testing: Use digital pH meter or soil test strips monthly.""",
                "category": "soil_management",
                "crops": ["all"],
                "priority": "high"
            },
            "npk_nutrients": {
                "content": """Essential plant nutrients (NPK):
                
                Nitrogen (N):
                - Function: Leaf growth, chlorophyll production
                - Deficiency signs: Yellow leaves, stunted growth
                - Sources: Urea, ammonium sulfate, compost
                
                Phosphorus (P):
                - Function: Root development, flowering, fruiting
                - Deficiency signs: Purple leaves, poor root growth
                - Sources: Bone meal, rock phosphate, DAP fertilizer
                
                Potassium (K):
                - Function: Disease resistance, water regulation
                - Deficiency signs: Brown leaf edges, weak stems
                - Sources: Potash, wood ash, kelp meal
                
                Application timing: Split doses during growing season.""",
                "category": "fertilization",
                "crops": ["all"],
                "priority": "high"
            },
            "irrigation_scheduling": {
                "content": """Smart irrigation principles:
                
                Soil moisture monitoring:
                - 0-30%: Drought stress, immediate irrigation needed
                - 30-70%: Optimal range for most crops
                - 70-100%: Risk of waterlogging, reduce irrigation
                
                Timing:
                - Early morning (6-10 AM): Reduces evaporation, prevents fungal issues
                - Avoid midday and evening watering
                
                Methods:
                - Drip irrigation: Most efficient, 90% efficiency
                - Sprinkler: Good coverage, 75-85% efficiency
                - Flood irrigation: Traditional, 60% efficiency
                
                Weather consideration: Skip irrigation before expected rain.""",
                "category": "irrigation",
                "crops": ["all"],
                "priority": "high"
            },
            "pest_management": {
                "content": """Integrated Pest Management (IPM):
                
                Prevention:
                - Crop rotation to break pest cycles
                - Maintain plant health through proper nutrition
                - Use pest-resistant varieties
                - Encourage beneficial insects
                
                Monitoring:
                - Regular field scouting (2-3 times per week)
                - Yellow sticky traps for flying pests
                - Pheromone traps for specific insects
                
                Control methods (in order of preference):
                1. Biological: Beneficial insects, microbial pesticides
                2. Cultural: Crop rotation, intercropping
                3. Mechanical: Hand picking, barriers
                4. Chemical: Selective pesticides as last resort
                
                Common pests: Aphids, caterpillars, mites, thrips""",
                "category": "pest_management",
                "crops": ["all"],
                "priority": "medium"
            },
            "disease_management": {
                "content": """Plant disease prevention and control:
                
                Prevention:
                - Proper spacing for air circulation
                - Avoid overhead watering late in day
                - Remove infected plant debris
                - Use disease-resistant varieties
                
                Common fungal diseases:
                - Powdery mildew: White powdery coating on leaves
                - Blight: Brown/black spots on leaves and stems
                - Root rot: Yellowing, wilting despite moist soil
                
                Treatment:
                - Fungicides: Copper-based (organic), systemic chemicals
                - Cultural: Improve drainage, reduce humidity
                - Pruning: Remove infected parts, sterilize tools
                
                Weather conditions favoring disease: High humidity + moderate temperatures""",
                "category": "disease_management",
                "crops": ["all"],
                "priority": "medium"
            },
            "crop_rotation": {
                "content": """Crop rotation principles:
                
                Benefits:
                - Breaks pest and disease cycles
                - Improves soil structure and fertility
                - Reduces weed pressure
                - Optimizes nutrient use
                
                Basic rotation groups:
                1. Legumes (nitrogen fixers): Beans, peas, clover
                2. Brassicas (heavy feeders): Cabbage, broccoli, cauliflower
                3. Root crops: Carrots, potatoes, radishes
                4. Grasses/Cereals: Corn, wheat, oats
                
                Simple 4-year rotation:
                Year 1: Legumes → Year 2: Brassicas → Year 3: Root crops → Year 4: Cereals
                
                Avoid: Same family crops in consecutive seasons""",
                "category": "cropping_systems",
                "crops": ["all"],
                "priority": "medium"
            }
        }
        
        for doc_id, doc_data in default_knowledge.items():
            self.add_document(doc_id, doc_data["content"], doc_data)
    
    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """Add a document to the knowledge base"""
        # Store document
        self.documents[doc_id] = content
        self.metadata[doc_id] = metadata or {}
        self.metadata[doc_id]["added_date"] = datetime.now().isoformat()
        
        # Generate embedding
        embedding = self.embedder.encode(content)
        self.embeddings[doc_id] = embedding
        
        # Save to disk
        self._save_knowledge()
    
    def search(self, query: str, top_k: int = 5, category: str = None) -> List[Dict]:
        """Search for relevant documents"""
        if not self.documents:
            return []
        
        # Generate query embedding
        query_embedding = self.embedder.encode(query)
        
        # Calculate similarities
        similarities = []
        for doc_id, doc_embedding in self.embeddings.items():
            # Skip if category filter doesn't match
            if category and self.metadata.get(doc_id, {}).get("category") != category:
                continue
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            similarities.append({
                "doc_id": doc_id,
                "similarity": float(similarity),
                "content": self.documents[doc_id],
                "metadata": self.metadata[doc_id]
            })
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get a specific document by ID"""
        if doc_id not in self.documents:
            return None
        
        return {
            "doc_id": doc_id,
            "content": self.documents[doc_id],
            "metadata": self.metadata[doc_id],
            "embedding": self.embeddings[doc_id].tolist()
        }
    
    def update_document(self, doc_id: str, content: str, metadata: Dict = None):
        """Update an existing document"""
        if doc_id not in self.documents:
            raise ValueError(f"Document {doc_id} not found")
        
        self.add_document(doc_id, content, metadata)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the knowledge base"""
        if doc_id not in self.documents:
            return False
        
        del self.documents[doc_id]
        del self.embeddings[doc_id]
        del self.metadata[doc_id]
        
        self._save_knowledge()
        return True
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        categories = set()
        for metadata in self.metadata.values():
            if "category" in metadata:
                categories.add(metadata["category"])
        return sorted(list(categories))
    
    def get_documents_by_category(self, category: str) -> List[Dict]:
        """Get all documents in a specific category"""
        results = []
        for doc_id, metadata in self.metadata.items():
            if metadata.get("category") == category:
                results.append({
                    "doc_id": doc_id,
                    "content": self.documents[doc_id],
                    "metadata": metadata
                })
        return results
    
    def add_crop_specific_knowledge(self, crop: str, knowledge: Dict):
        """Add crop-specific knowledge"""
        doc_id = f"crop_{crop.lower()}"
        content = f"Crop-specific information for {crop}:\n\n" + knowledge.get("content", "")
        
        metadata = {
            "category": "crop_specific",
            "crop": crop,
            "priority": knowledge.get("priority", "medium")
        }
        metadata.update(knowledge.get("metadata", {}))
        
        self.add_document(doc_id, content, metadata)
    
    def get_contextual_knowledge(self, query: str, sensor_data: List[Dict] = None, 
                               weather_data: Dict = None) -> List[Dict]:
        """Get knowledge relevant to current conditions"""
        # Base search
        results = self.search(query, top_k=3)
        
        # Add context-specific knowledge
        if sensor_data:
            # Check for specific conditions
            latest_sensor = sensor_data[0] if sensor_data else {}
            
            # pH-related knowledge
            if "ph" in latest_sensor:
                ph_value = latest_sensor["ph"]
                if ph_value < 6.0:
                    ph_results = self.search("acidic soil pH lime", top_k=2)
                    results.extend(ph_results)
                elif ph_value > 7.5:
                    ph_results = self.search("alkaline soil pH sulfur", top_k=2)
                    results.extend(ph_results)
            
            # Moisture-related knowledge
            if "soil_moisture" in latest_sensor:
                moisture = latest_sensor["soil_moisture"]
                if moisture < 30:
                    irrigation_results = self.search("drought irrigation water stress", top_k=2)
                    results.extend(irrigation_results)
                elif moisture > 80:
                    drainage_results = self.search("waterlogged drainage", top_k=2)
                    results.extend(drainage_results)
        
        if weather_data:
            # Weather-specific knowledge
            if "current" in weather_data:
                temp = weather_data["current"].get("temperature", 25)
                humidity = weather_data["current"].get("humidity", 60)
                
                if temp > 35:
                    heat_results = self.search("heat stress temperature", top_k=2)
                    results.extend(heat_results)
                
                if humidity > 80:
                    disease_results = self.search("high humidity fungal disease", top_k=2)
                    results.extend(disease_results)
        
        # Remove duplicates and limit results
        seen_ids = set()
        unique_results = []
        for result in results:
            if result["doc_id"] not in seen_ids:
                seen_ids.add(result["doc_id"])
                unique_results.append(result)
        
        return unique_results[:8]  # Limit to 8 most relevant pieces
    
    def _save_knowledge(self):
        """Save knowledge base to disk"""
        try:
            # Save documents and metadata as JSON
            with open(self.knowledge_dir / "documents.json", "w") as f:
                json.dump({
                    "documents": self.documents,
                    "metadata": self.metadata
                }, f, indent=2)
            
            # Save embeddings as pickle (binary format for numpy arrays)
            with open(self.knowledge_dir / "embeddings.pkl", "wb") as f:
                pickle.dump(self.embeddings, f)
                
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
    
    def load_knowledge(self):
        """Load knowledge base from disk"""
        try:
            # Load documents and metadata
            docs_file = self.knowledge_dir / "documents.json"
            if docs_file.exists():
                with open(docs_file, "r") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", {})
                    self.metadata = data.get("metadata", {})
            
            # Load embeddings
            embeddings_file = self.knowledge_dir / "embeddings.pkl"
            if embeddings_file.exists():
                with open(embeddings_file, "rb") as f:
                    self.embeddings = pickle.load(f)
            
            print(f"Loaded {len(self.documents)} documents from knowledge base")
            
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            # Initialize with default knowledge if loading fails
            self._initialize_default_knowledge()
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        categories = self.get_categories()
        category_counts = {}
        
        for category in categories:
            category_counts[category] = len(self.get_documents_by_category(category))
        
        return {
            "total_documents": len(self.documents),
            "categories": categories,
            "category_counts": category_counts,
            "last_updated": max(
                [meta.get("added_date", "") for meta in self.metadata.values()],
                default=""
            )
        }
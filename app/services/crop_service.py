# app/services/crop_service.py
import pickle
import numpy as np
import os
import hashlib
import logging
from typing import Dict, List

# Configure logging
logger = logging.getLogger(__name__)

class CropPredictionService:
    """Integrates Smart Harvest ML model into FarmBot"""
    
    # Hashes should ideally be loaded from environment variables or a secure config
    # For now, these must be set to the actual SHA256 hashes of your valid .pkl files
    EXPECTED_MODEL_HASH = os.getenv("MODEL_HASH", "replace_with_actual_sha256_hash_of_model_pkl")
    EXPECTED_LABELS_HASH = os.getenv("LABELS_HASH", "replace_with_actual_sha256_hash_of_labels_pkl")
    
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), "../ml_models/Model.pkl")
        self.labels_path = os.path.join(os.path.dirname(__file__), "../ml_models/labels.pkl")
        
        self.model = self._safe_load_pickle(self.model_path, self.EXPECTED_MODEL_HASH)
        self.labels = self._safe_load_pickle(self.labels_path, self.EXPECTED_LABELS_HASH)
    
    def _safe_load_pickle(self, file_path: str, expected_hash: str):
        """
        Safely load a pickle file by verifying its SHA256 hash first.
        Prevents insecure deserialization attacks.
        """
        if not os.path.exists(file_path):
            logger.error(f"ML model file not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        # specific check to allow development without hashes, strictly warn in production
        if expected_hash.startswith("replace_with"):
             if os.getenv("ENVIRONMENT") == "production":
                 logger.critical(f"SECURITY ALERT: Model hash verification skipped for {file_path} in PRODUCTION. Set MODEL_HASH/LABELS_HASH.")
                 # In strict mode, you might want to raise an error here
             else:
                 logger.warning(f"Dev Mode: Skipping hash verification for {file_path}")
                 with open(file_path, 'rb') as f:
                     return pickle.load(f)

        # Verify hash
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        calculated_hash = sha256_hash.hexdigest()
        
        if calculated_hash != expected_hash:
            logger.critical(f"SECURITY ALERT: Hash mismatch for {file_path}. Expected {expected_hash}, got {calculated_hash}")
            raise ValueError(f"Integrity check failed for {file_path}. The file may have been tampered with.")
            
        # If hash matches, load safely
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def predict_crops(
        self,
        temperature: float,
        humidity: float,
        ph: float,
        rainfall: float,
        season: str,
        state: str,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        top_k: int = 5
    ) -> Dict:
        """
        Predict top crops using the Smart Harvest ML model
        """
        if not self.model or not self.labels:
             return {"error": "ML Model not initialized correctly due to security or missing files."}

        # State/Season encoding (same as Smart Harvest)
        state_season_map = {
            "uttarpradesh": 0, "maharashtra": 1, "punjab": 2,
            # ... add all mappings from StatesSeason.json
            "kharif": 20, "rabi": 21, "perennial": 22
        }
        
        season_encoded = state_season_map.get(season.lower(), 20)
        state_encoded = state_season_map.get(state.lower(), 0)
        
        # Prepare features (same order as training)
        features = np.array([[
            temperature, humidity, ph, rainfall,
            season_encoded, state_encoded,
            nitrogen, phosphorus, potassium
        ]])
        
        # Get predictions
        probabilities = self.model.predict_proba(features)[0]
        
        # Get top K crops
        top_indices = np.argsort(-probabilities)[:top_k]
        top_crops = self.labels.inverse_transform(top_indices)
        top_probs = probabilities[top_indices]
        
        results = []
        for crop, prob in zip(top_crops, top_probs):
            results.append({
                "crop": crop,
                "probability": float(prob),
                "confidence": f"{prob*100:.1f}%"
            })
        
        return {
            "recommended_crops": results,
            "optimal_crop": results[0]["crop"] if results else None,
            "model": "RandomForestClassifier",
            "features_used": {
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
                "season": season,
                "state": state,
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium
            }
        }
    
    def get_fertilizer_recommendation(
        self,
        crop: str,
        nitrogen: float,
        phosphorus: float,
        potassium: float
    ) -> Dict:
        """
        Fertilizer recommendations (from Smart Harvest)
        """
        # Load fertilizer CSV data (you'll need to add this)
        fertilizer_data = {
            "rice": {"N": 80, "P": 40, "K": 40},
            "wheat": {"N": 100, "P": 50, "K": 50},
            # ... add all from fertilizer.csv
        }
        
        crop_lower = crop.lower()
        if crop_lower not in fertilizer_data:
            return {"error": f"No data for crop: {crop}"}
        
        required = fertilizer_data[crop_lower]
        
        n_diff = required["N"] - nitrogen
        p_diff = required["P"] - phosphorus
        k_diff = required["K"] - potassium
        
        # Determine primary deficiency
        deficiencies = {
            "nitrogen": n_diff,
            "phosphorus": p_diff,
            "potassium": k_diff
        }
        
        primary_def = max(deficiencies, key=lambda k: abs(deficiencies[k]))
        
        recommendations = {
            "nitrogen": {
                "low": "Add manure, compost, or nitrogen-fixing crops",
                "high": "Reduce nitrogen fertilizers, plant heavy feeders"
            },
            "phosphorus": {
                "low": "Add bone meal or rock phosphate",
                "high": "Use phosphorus-free fertilizers"
            },
            "potassium": {
                "low": "Add wood ash, kelp meal, or potash fertilizers",
                "high": "Reduce potassium-rich amendments"
            }
        }
        
        status = "low" if deficiencies[primary_def] > 0 else "high"
        
        return {
            "crop": crop,
            "soil_status": {
                "nitrogen": {"current": nitrogen, "required": required["N"], "difference": n_diff},
                "phosphorus": {"current": phosphorus, "required": required["P"], "difference": p_diff},
                "potassium": {"current": potassium, "required": required["K"], "difference": k_diff}
            },
            "primary_concern": primary_def,
            "recommendation": recommendations[primary_def][status]
        }
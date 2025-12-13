# app/services/crop_service.py
import pickle
import numpy as np
import os
from typing import Dict, List

class CropPredictionService:
    """Integrates Smart Harvest ML model into FarmBot"""
    
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "../ml_models/Model.pkl")
        labels_path = os.path.join(os.path.dirname(__file__), "../ml_models/labels.pkl")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)
    
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
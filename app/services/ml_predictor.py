# app/services/ml_predictor.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from pathlib import Path

class AgriculturalMLPredictor:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.models = {
            "soil_moisture_predictor": RandomForestRegressor(n_estimators=100, random_state=42),
            "yield_predictor": RandomForestRegressor(n_estimators=150, random_state=42),
            "pest_risk_classifier": RandomForestRegressor(n_estimators=100, random_state=42),
            "anomaly_detector": IsolationForest(contamination=0.1, random_state=42)
        }
        
        # Scalers for feature normalization
        self.scalers = {
            "soil_features": StandardScaler(),
            "weather_features": StandardScaler(),
            "combined_features": StandardScaler()
        }
        
        # Model metadata
        self.model_metadata = {}
        
        # Load existing models if available
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            for model_name in self.models.keys():
                model_path = self.model_dir / f"{model_name}.joblib"
                scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
                metadata_path = self.model_dir / f"{model_name}_metadata.joblib"
                
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    print(f"Loaded {model_name} from disk")
                
                if scaler_path.exists():
                    if model_name.endswith("_predictor"):
                        scaler_key = "combined_features"
                    else:
                        scaler_key = "soil_features"
                    self.scalers[scaler_key] = joblib.load(scaler_path)
                
                if metadata_path.exists():
                    self.model_metadata[model_name] = joblib.load(metadata_path)
                    
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def _save_model(self, model_name: str):
        """Save model, scaler, and metadata to disk"""
        try:
            model_path = self.model_dir / f"{model_name}.joblib"
            scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
            metadata_path = self.model_dir / f"{model_name}_metadata.joblib"
            
            joblib.dump(self.models[model_name], model_path)
            
            # Save appropriate scaler
            if model_name.endswith("_predictor"):
                joblib.dump(self.scalers["combined_features"], scaler_path)
            else:
                joblib.dump(self.scalers["soil_features"], scaler_path)
            
            if model_name in self.model_metadata:
                joblib.dump(self.model_metadata[model_name], metadata_path)
                
            print(f"Saved {model_name} to disk")
            
        except Exception as e:
            print(f"Error saving model {model_name}: {e}")
    
    def prepare_features(self, sensor_data: List[Dict], weather_data: List[Dict] = None) -> pd.DataFrame:
        """Prepare features from sensor and weather data"""
        try:
            # Convert sensor data to DataFrame
            sensor_df = pd.DataFrame(sensor_data)
            
            if sensor_df.empty:
                return pd.DataFrame()
            
            # Ensure timestamp is datetime
            if 'timestamp' in sensor_df.columns:
                sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
                sensor_df = sensor_df.sort_values('timestamp')
            
            # Feature engineering for sensor data
            features = {}
            
            # Basic sensor features
            numeric_columns = ['soil_moisture', 'soil_temperature', 'ph', 'ec', 'n', 'p', 'k']
            for col in numeric_columns:
                if col in sensor_df.columns:
                    # Current value
                    features[f'{col}_current'] = sensor_df[col].iloc[-1] if not sensor_df[col].empty else 0
                    
                    # Statistical features
                    features[f'{col}_mean'] = sensor_df[col].mean()
                    features[f'{col}_std'] = sensor_df[col].std()
                    features[f'{col}_min'] = sensor_df[col].min()
                    features[f'{col}_max'] = sensor_df[col].max()
                    
                    # Trend features (if we have enough data)
                    if len(sensor_df) >= 3:
                        recent_values = sensor_df[col].tail(3).values
                        if len(recent_values) >= 2:
                            features[f'{col}_trend'] = (recent_values[-1] - recent_values[0]) / len(recent_values)
            
            # Time-based features
            if 'timestamp' in sensor_df.columns and not sensor_df['timestamp'].empty:
                latest_time = sensor_df['timestamp'].iloc[-1]
                features['hour_of_day'] = latest_time.hour
                features['day_of_week'] = latest_time.weekday()
                features['day_of_year'] = latest_time.dayofyear
                features['month'] = latest_time.month
            
            # Weather features (if available)
            if weather_data:
                weather_df = pd.DataFrame(weather_data)
                if not weather_df.empty:
                    weather_features = ['temperature', 'humidity', 'pressure', 'wind_speed']
                    for col in weather_features:
                        if col in weather_df.columns:
                            features[f'weather_{col}_mean'] = weather_df[col].mean()
                            features[f'weather_{col}_current'] = weather_df[col].iloc[-1] if not weather_df[col].empty else 0
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Fill NaN values
            feature_df = feature_df.fillna(0)
            
            return feature_df
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    async def predict_soil_moisture(self, sensor_data: List[Dict], weather_data: List[Dict] = None, 
                                  hours_ahead: int = 24) -> Dict[str, Any]:
        """Predict future soil moisture levels"""
        try:
            features = self.prepare_features(sensor_data, weather_data)
            
            if features.empty:
                return {"error": "Insufficient data for prediction"}
            
            model = self.models["soil_moisture_predictor"]
            
            # Check if model is trained
            if not hasattr(model, 'feature_importances_'):
                # Train model with available data
                await self._train_soil_moisture_model(sensor_data, weather_data)
            
            # Scale features
            scaled_features = self.scalers["combined_features"].transform(features)
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            
            # Get feature importance
            feature_names = features.columns.tolist()
            importances = model.feature_importances_
            
            # Generate recommendations
            recommendations = self._generate_moisture_recommendations(prediction, features.iloc[0])
            
            return {
                "predicted_moisture": round(prediction, 2),
                "prediction_horizon_hours": hours_ahead,
                "confidence": self._calculate_prediction_confidence(model, scaled_features),
                "feature_importance": dict(zip(feature_names, importances)),
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Moisture prediction failed: {str(e)}"}
    
    async def predict_yield(self, sensor_data: List[Dict], weather_data: List[Dict] = None,
                          crop_type: str = "general", growth_stage: str = "vegetative") -> Dict[str, Any]:
        """Predict crop yield based on current conditions"""
        try:
            features = self.prepare_features(sensor_data, weather_data)
            
            if features.empty:
                return {"error": "Insufficient data for yield prediction"}
            
            # Add crop-specific features
            crop_encoding = {"corn": 1, "wheat": 2, "soybean": 3, "rice": 4, "general": 0}
            stage_encoding = {"seedling": 1, "vegetative": 2, "flowering": 3, "fruiting": 4, "harvest": 5}
            
            features['crop_type'] = crop_encoding.get(crop_type.lower(), 0)
            features['growth_stage'] = stage_encoding.get(growth_stage.lower(), 2)
            
            model = self.models["yield_predictor"]
            
            # Check if model is trained
            if not hasattr(model, 'feature_importances_'):
                await self._train_yield_model(sensor_data, weather_data)
            
            # Scale features
            scaled_features = self.scalers["combined_features"].transform(features)
            
            # Make prediction
            yield_prediction = model.predict(scaled_features)[0]
            
            # Calculate yield factors
            yield_factors = self._analyze_yield_factors(features.iloc[0])
            
            return {
                "predicted_yield_index": round(yield_prediction, 2),
                "crop_type": crop_type,
                "growth_stage": growth_stage,
                "yield_factors": yield_factors,
                "optimization_suggestions": self._get_yield_optimization_suggestions(features.iloc[0]),
                "confidence": self._calculate_prediction_confidence(model, scaled_features),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Yield prediction failed: {str(e)}"}
    
    async def assess_pest_disease_risk(self, sensor_data: List[Dict], weather_data: List[Dict] = None) -> Dict[str, Any]:
        """Assess pest and disease risk based on environmental conditions"""
        try:
            features = self.prepare_features(sensor_data, weather_data)
            
            if features.empty:
                return {"error": "Insufficient data for risk assessment"}
            
            # Calculate environmental risk factors
            risk_factors = self._calculate_environmental_risk_factors(features.iloc[0])
            
            # Use pest risk model if trained, otherwise use rule-based assessment
            model = self.models["pest_risk_classifier"]
            
            if hasattr(model, 'feature_importances_'):
                scaled_features = self.scalers["combined_features"].transform(features)
                pest_risk_score = model.predict(scaled_features)[0]
            else:
                pest_risk_score = self._rule_based_pest_risk(features.iloc[0])
            
            # Categorize risk level
            if pest_risk_score >= 0.7:
                risk_level = "high"
            elif pest_risk_score >= 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            # Generate specific pest/disease alerts
            alerts = self._generate_pest_disease_alerts(features.iloc[0])
            
            return {
                "overall_risk_score": round(pest_risk_score, 3),
                "risk_level": risk_level,
                "environmental_factors": risk_factors,
                "specific_alerts": alerts,
                "prevention_recommendations": self._get_prevention_recommendations(risk_level, alerts),
                "monitoring_schedule": self._suggest_monitoring_schedule(risk_level),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Risk assessment failed: {str(e)}"}
    
    async def detect_anomalies(self, sensor_data: List[Dict]) -> Dict[str, Any]:
        """Detect anomalies in sensor readings"""
        try:
            features = self.prepare_features(sensor_data)
            
            if features.empty or len(sensor_data) < 10:
                return {"error": "Insufficient data for anomaly detection"}
            
            model = self.models["anomaly_detector"]
            
            # Prepare recent data points for anomaly detection
            recent_sensor_df = pd.DataFrame(sensor_data[-20:])  # Last 20 readings
            recent_features = []
            
            for _, row in recent_sensor_df.iterrows():
                row_features = []
                for col in ['soil_moisture', 'soil_temperature', 'ph', 'ec', 'n', 'p', 'k']:
                    if col in row and pd.notna(row[col]):
                        row_features.append(float(row[col]))
                    else:
                        row_features.append(0.0)
                
                if row_features:
                    recent_features.append(row_features)
            
            if not recent_features:
                return {"error": "No valid sensor readings found"}
            
            # Detect anomalies
            recent_features_array = np.array(recent_features)
            anomaly_scores = model.decision_function(recent_features_array)
            anomaly_labels = model.predict(recent_features_array)
            
            # Identify anomalous readings
            anomalies = []
            for i, (score, label) in enumerate(zip(anomaly_scores, anomaly_labels)):
                if label == -1:  # Anomaly detected
                    reading_index = len(sensor_data) - len(recent_features) + i
                    if reading_index >= 0 and reading_index < len(sensor_data):
                        anomalies.append({
                            "reading_index": reading_index,
                            "timestamp": sensor_data[reading_index].get("timestamp"),
                            "anomaly_score": float(score),
                            "affected_parameters": self._identify_anomalous_parameters(
                                sensor_data[reading_index], recent_sensor_df
                            )
                        })
            
            return {
                "anomalies_detected": len(anomalies),
                "anomaly_details": anomalies,
                "overall_data_quality": "good" if len(anomalies) <= 2 else "concerning",
                "recommendations": self._get_anomaly_recommendations(anomalies),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Anomaly detection failed: {str(e)}"}
    
    async def _train_soil_moisture_model(self, sensor_data: List[Dict], weather_data: List[Dict] = None):
        """Train soil moisture prediction model"""
        try:
            if len(sensor_data) < 50:  # Need sufficient data for training
                print("Insufficient data for training soil moisture model")
                return
            
            df = pd.DataFrame(sensor_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Create features and targets
            X_list = []
            y_list = []
            
            # Create training samples with sliding window
            window_size = 5
            for i in range(window_size, len(df)):
                # Features: previous readings
                window_data = df.iloc[i-window_size:i]
                features = self.prepare_features(window_data.to_dict('records'), weather_data)
                
                if not features.empty:
                    # Target: next soil moisture reading
                    target_moisture = df.iloc[i]['soil_moisture']
                    if pd.notna(target_moisture):
                        X_list.append(features.iloc[0].values)
                        y_list.append(target_moisture)
            
            if len(X_list) < 20:
                print("Insufficient valid training samples")
                return
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scalers["combined_features"].fit_transform(X_train)
            X_test_scaled = self.scalers["combined_features"].transform(X_test)
            
            # Train model
            model = self.models["soil_moisture_predictor"]
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store metadata
            self.model_metadata["soil_moisture_predictor"] = {
                "training_date": datetime.now().isoformat(),
                "training_samples": len(X_train),
                "test_mae": mae,
                "test_r2": r2,
                "feature_count": X.shape[1]
            }
            
            # Save model
            self._save_model("soil_moisture_predictor")
            
            print(f"Soil moisture model trained. MAE: {mae:.3f}, R²: {r2:.3f}")
            
        except Exception as e:
            print(f"Error training soil moisture model: {e}")
    
    async def _train_yield_model(self, sensor_data: List[Dict], weather_data: List[Dict] = None):
        """Train yield prediction model with synthetic data (placeholder)"""
        try:
            # This is a simplified training process
            # In practice, you'd need historical yield data
            
            features = self.prepare_features(sensor_data, weather_data)
            if features.empty:
                return
            
            # Generate synthetic training data for demonstration
            np.random.seed(42)
            n_samples = 200
            
            # Create synthetic features
            synthetic_features = []
            synthetic_yields = []
            
            for _ in range(n_samples):
                # Random but realistic sensor values
                moisture = np.random.normal(50, 15)
                temp = np.random.normal(20, 5)
                ph = np.random.normal(6.5, 0.5)
                n_content = np.random.normal(30, 10)
                
                # Synthetic yield based on conditions (simplified relationship)
                yield_index = (
                    0.3 * min(max((moisture - 30) / 40, 0), 1) +  # Moisture factor
                    0.2 * min(max((25 - abs(temp - 20)) / 25, 0), 1) +  # Temperature factor
                    0.3 * min(max((1 - abs(ph - 6.5)) / 1.5, 0), 1) +  # pH factor
                    0.2 * min(max(n_content / 50, 0), 1)  # Nitrogen factor
                ) + np.random.normal(0, 0.1)
                
                synthetic_features.append([moisture, temp, ph, n_content, 2, 2])  # Include crop_type, growth_stage
                synthetic_yields.append(max(0, yield_index))
            
            X = np.array(synthetic_features)
            y = np.array(synthetic_yields)
            
            # Train model
            X_scaled = self.scalers["combined_features"].fit_transform(X)
            model = self.models["yield_predictor"]
            model.fit(X_scaled, y)
            
            # Store metadata
            self.model_metadata["yield_predictor"] = {
                "training_date": datetime.now().isoformat(),
                "training_samples": len(X),
                "model_type": "synthetic_trained"
            }
            
            self._save_model("yield_predictor")
            print("Yield prediction model trained with synthetic data")
            
        except Exception as e:
            print(f"Error training yield model: {e}")
    
    def _calculate_prediction_confidence(self, model, features) -> float:
        """Calculate confidence score for predictions"""
        try:
            if hasattr(model, 'predict'):
                # For tree-based models, use prediction variance as confidence indicator
                if hasattr(model, 'estimators_'):
                    predictions = [tree.predict(features)[0] for tree in model.estimators_]
                    variance = np.var(predictions)
                    confidence = max(0, min(1, 1 - variance / 10))  # Normalize variance
                else:
                    confidence = 0.7  # Default confidence
                
                return round(confidence, 3)
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _generate_moisture_recommendations(self, predicted_moisture: float, features: pd.Series) -> List[str]:
        """Generate irrigation recommendations based on predicted moisture"""
        recommendations = []
        
        if predicted_moisture < 30:
            recommendations.append("Immediate irrigation recommended - drought stress predicted")
            recommendations.append("Check irrigation system functionality")
        elif predicted_moisture < 40:
            recommendations.append("Plan irrigation within 12-24 hours")
            recommendations.append("Monitor soil moisture closely")
        elif predicted_moisture > 80:
            recommendations.append("Reduce irrigation - waterlogging risk")
            recommendations.append("Ensure proper drainage")
        else:
            recommendations.append("Moisture levels appear optimal")
        
        # Weather-based recommendations
        current_temp = features.get('soil_temperature_current', 20)
        if current_temp > 30:
            recommendations.append("High temperature detected - increase irrigation frequency")
        
        return recommendations
    
    def _analyze_yield_factors(self, features: pd.Series) -> Dict[str, Any]:
        """Analyze factors affecting yield"""
        factors = {}
        
        # Soil health factors
        ph = features.get('ph_current', 6.5)
        if 6.0 <= ph <= 7.0:
            factors['ph_impact'] = 'positive'
        elif 5.5 <= ph < 6.0 or 7.0 < ph <= 7.5:
            factors['ph_impact'] = 'neutral'
        else:
            factors['ph_impact'] = 'negative'
        
        # Moisture factor
        moisture = features.get('soil_moisture_current', 50)
        if 40 <= moisture <= 70:
            factors['moisture_impact'] = 'positive'
        else:
            factors['moisture_impact'] = 'negative'
        
        # Nutrient factors
        n_content = features.get('n_current', 0)
        if n_content > 25:
            factors['nitrogen_impact'] = 'positive'
        elif n_content > 15:
            factors['nitrogen_impact'] = 'neutral'
        else:
            factors['nitrogen_impact'] = 'negative'
        
        return factors
    
    def _get_yield_optimization_suggestions(self, features: pd.Series) -> List[str]:
        """Get suggestions for yield optimization"""
        suggestions = []
        
        ph = features.get('ph_current', 6.5)
        if ph < 6.0:
            suggestions.append("Apply lime to increase soil pH")
        elif ph > 7.5:
            suggestions.append("Apply sulfur or organic matter to reduce pH")
        
        moisture = features.get('soil_moisture_current', 50)
        if moisture < 40:
            suggestions.append("Improve irrigation to maintain optimal moisture")
        elif moisture > 70:
            suggestions.append("Improve drainage to prevent waterlogging")
        
        n_content = features.get('n_current', 0)
        if n_content < 20:
            suggestions.append("Consider nitrogen fertilizer application")
        
        return suggestions
    
    def _calculate_environmental_risk_factors(self, features: pd.Series) -> Dict[str, str]:
        """Calculate environmental risk factors for pests and diseases"""
        factors = {}
        
        # Temperature risk
        temp = features.get('soil_temperature_current', 20)
        if 20 <= temp <= 30:
            factors['temperature_risk'] = 'high'  # Optimal for many pests
        elif temp > 35 or temp < 10:
            factors['temperature_risk'] = 'low'
        else:
            factors['temperature_risk'] = 'medium'
        
        # Moisture risk
        moisture = features.get('soil_moisture_current', 50)
        if moisture > 70:
            factors['moisture_risk'] = 'high'  # High humidity favors diseases
        elif moisture < 30:
            factors['moisture_risk'] = 'low'
        else:
            factors['moisture_risk'] = 'medium'
        
        # Weather humidity (if available)
        humidity = features.get('weather_humidity_current', 60)
        if humidity > 80:
            factors['humidity_risk'] = 'high'
        elif humidity < 40:
            factors['humidity_risk'] = 'low'
        else:
            factors['humidity_risk'] = 'medium'
        
        return factors
    
    def _rule_based_pest_risk(self, features: pd.Series) -> float:
        """Rule-based pest risk assessment"""
        risk_score = 0.0
        
        # Temperature contribution
        temp = features.get('soil_temperature_current', 20)
        if 20 <= temp <= 30:
            risk_score += 0.3
        elif 15 <= temp < 20 or 30 < temp <= 35:
            risk_score += 0.2
        else:
            risk_score += 0.1
        
        # Moisture contribution
        moisture = features.get('soil_moisture_current', 50)
        if moisture > 70:
            risk_score += 0.3
        elif 50 <= moisture <= 70:
            risk_score += 0.2
        else:
            risk_score += 0.1
        
        # Humidity contribution
        humidity = features.get('weather_humidity_current', 60)
        if humidity > 80:
            risk_score += 0.3
        elif 60 <= humidity <= 80:
            risk_score += 0.2
        else:
            risk_score += 0.1
        
        # Season factor (based on day of year)
        day_of_year = features.get('day_of_year', 180)
        if 120 <= day_of_year <= 240:  # Growing season
            risk_score += 0.1
        
        return min(1.0, risk_score)
    
    def _generate_pest_disease_alerts(self, features: pd.Series) -> List[Dict[str, str]]:
        """Generate specific pest and disease alerts"""
        alerts = []
        
        temp = features.get('soil_temperature_current', 20)
        moisture = features.get('soil_moisture_current', 50)
        humidity = features.get('weather_humidity_current', 60)
        
        # Fungal disease alerts
        if humidity > 80 and temp > 15 and temp < 30:
            alerts.append({
                "type": "fungal_disease",
                "severity": "high",
                "description": "Conditions favorable for fungal diseases (powdery mildew, blight)"
            })
        
        # Pest alerts
        if temp > 25 and moisture > 40:
            alerts.append({
                "type": "insect_pest",
                "severity": "medium",
                "description": "Conditions favorable for aphids and other sap-sucking insects"
            })
        
        # Root rot alert
        if moisture > 80:
            alerts.append({
                "type": "root_disease",
                "severity": "high",
                "description": "High soil moisture increases root rot risk"
            })
        
        return alerts
    
    def _get_prevention_recommendations(self, risk_level: str, alerts: List[Dict]) -> List[str]:
        """Get prevention recommendations based on risk assessment"""
        recommendations = []
        
        if risk_level == "high":
            recommendations.append("Increase field monitoring frequency to daily")
            recommendations.append("Prepare preventive treatments")
        
        for alert in alerts:
            if alert["type"] == "fungal_disease":
                recommendations.append("Improve air circulation around plants")
                recommendations.append("Consider fungicide application if symptoms appear")
            elif alert["type"] == "insect_pest":
                recommendations.append("Deploy sticky traps for monitoring")
                recommendations.append("Encourage beneficial insects")
            elif alert["type"] == "root_disease":
                recommendations.append("Improve soil drainage")
                recommendations.append("Reduce irrigation frequency")
        
        return recommendations
    
    def _suggest_monitoring_schedule(self, risk_level: str) -> Dict[str, str]:
        """Suggest monitoring schedule based on risk level"""
        if risk_level == "high":
            return {
                "frequency": "daily",
                "focus_areas": "pest hotspots, disease symptoms, plant stress indicators",
                "duration": "continue until risk decreases"
            }
        elif risk_level == "medium":
            return {
                "frequency": "every 2-3 days",
                "focus_areas": "general plant health, early symptoms",
                "duration": "2 weeks or until conditions change"
            }
        else:
            return {
                "frequency": "weekly",
                "focus_areas": "routine health check",
                "duration": "standard monitoring"
            }
    
    def _identify_anomalous_parameters(self, anomalous_reading: Dict, recent_df: pd.DataFrame) -> List[str]:
        """Identify which parameters are anomalous"""
        anomalous_params = []
        
        for param in ['soil_moisture', 'soil_temperature', 'ph', 'ec', 'n', 'p', 'k']:
            if param in anomalous_reading and param in recent_df.columns:
                current_value = anomalous_reading[param]
                if pd.notna(current_value):
                    param_mean = recent_df[param].mean()
                    param_std = recent_df[param].std()
                    
                    if param_std > 0:
                        z_score = abs((current_value - param_mean) / param_std)
                        if z_score > 2:  # More than 2 standard deviations
                            anomalous_params.append(param)
        
        return anomalous_params
    
    def _get_anomaly_recommendations(self, anomalies: List[Dict]) -> List[str]:
        """Get recommendations for handling anomalies"""
        if not anomalies:
            return ["No anomalies detected - data quality is good"]
        
        recommendations = []
        recommendations.append(f"Investigate {len(anomalies)} anomalous readings")
        
        # Check if multiple recent anomalies
        if len(anomalies) > 3:
            recommendations.append("Multiple anomalies detected - check sensor calibration")
            recommendations.append("Consider sensor maintenance or replacement")
        
        # Parameter-specific recommendations
        affected_params = set()
        for anomaly in anomalies:
            affected_params.update(anomaly.get("affected_parameters", []))
        
        if "soil_moisture" in affected_params:
            recommendations.append("Check soil moisture sensor for accuracy")
        if "ph" in affected_params:
            recommendations.append("Calibrate pH sensor with buffer solutions")
        if
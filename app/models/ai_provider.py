# app/models/ai_provider.py
from groq import Groq
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

class GroqAIProvider:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be set in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        
        # Available models - you can adjust based on your needs
        self.models = {
            "fast": "llama3-8b-8192",      # Fast responses, good for quick questions
            "balanced": "llama3-70b-8192", # Balanced speed and intelligence  
            "smart": "mixtral-8x7b-32768"  # Most intelligent, for complex analysis
        }
        
        # Model selection based on query complexity
        self.default_model = "balanced"
        
    def _select_model(self, query: str, complexity_hint: str = None) -> str:
        """Select appropriate model based on query complexity"""
        if complexity_hint:
            return self.models.get(complexity_hint, self.default_model)
        
        # Simple heuristics for model selection
        if len(query) < 50 and "?" in query:
            return self.models["fast"]
        elif any(word in query.lower() for word in ["analyze", "predict", "recommend", "complex", "detailed"]):
            return self.models["smart"]
        else:
            return self.models["balanced"]
    
    async def generate_response(self, 
                              messages: List[Dict[str, str]], 
                              model_hint: str = None,
                              max_tokens: int = 1024,
                              temperature: float = 0.3) -> Dict[str, Any]:
        """Generate AI response using Groq"""
        try:
            model = self._select_model(messages[-1]["content"], model_hint) if messages else self.default_model
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model_used": model,
                "tokens_used": response.usage.total_tokens,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I'm having trouble processing your request right now. Please try again.",
                "timestamp": datetime.now().isoformat()
            }
    
    async def generate_agricultural_advice(self,
                                         user_query: str,
                                         sensor_data: List[Dict] = None,
                                         weather_data: Dict = None,
                                         historical_data: List[Dict] = None,
                                         farm_context: Dict = None) -> Dict[str, Any]:
        """Generate specialized agricultural advice with context"""
        
        # Build comprehensive prompt with all available data
        system_prompt = self._build_agricultural_system_prompt()
        context_prompt = self._build_context_prompt(sensor_data, weather_data, historical_data, farm_context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context_prompt}\n\nFarmer's Question: {user_query}"}
        ]
        
        # Use smart model for agricultural advice
        result = await self.generate_response(messages, model_hint="smart", max_tokens=1500, temperature=0.2)
        
        if result["success"]:
            # Parse and structure the advice
            advice = self._structure_advice(result["response"], user_query)
            result["structured_advice"] = advice
        
        return result
    
    def _build_agricultural_system_prompt(self) -> str:
        """Build comprehensive system prompt for agricultural AI"""
        return """You are an expert agricultural AI advisor with deep knowledge of:
- Soil science, plant nutrition, and crop management
- Weather patterns and their impact on farming
- Integrated pest management and sustainable practices
- Precision agriculture and sensor data interpretation
- Local farming conditions and best practices

GUIDELINES:
1. Always provide practical, actionable advice
2. Consider safety and sustainability in all recommendations
3. Use clear, farmer-friendly language (avoid overly technical jargon)
4. When data is insufficient, ask specific clarifying questions
5. Consider economic factors and resource constraints
6. Provide immediate actions and long-term strategies
7. Include confidence levels in your recommendations
8. Reference specific data points when making suggestions

RESPONSE FORMAT:
- Start with a brief assessment of the situation
- Provide immediate actionable recommendations
- Include supporting reasoning based on data
- Suggest monitoring points and follow-up actions
- End with additional questions if more information is needed

Always prioritize farmer safety and crop health in your advice."""

    def _build_context_prompt(self, 
                            sensor_data: List[Dict] = None,
                            weather_data: Dict = None,
                            historical_data: List[Dict] = None,
                            farm_context: Dict = None) -> str:
        """Build context prompt with all available data"""
        
        context_parts = ["AVAILABLE DATA FOR ANALYSIS:"]
        
        # Sensor data section
        if sensor_data:
            context_parts.append("\n📊 RECENT SENSOR READINGS:")
            for reading in sensor_data[:5]:  # Latest 5 readings
                timestamp = reading.get('timestamp', 'Unknown time')
                context_parts.append(f"Time: {timestamp}")
                if reading.get('soil_moisture'): context_parts.append(f"  Soil Moisture: {reading['soil_moisture']}%")
                if reading.get('soil_temperature'): context_parts.append(f"  Soil Temperature: {reading['soil_temperature']}°C")
                if reading.get('ph'): context_parts.append(f"  pH: {reading['ph']}")
                if reading.get('ec'): context_parts.append(f"  Electrical Conductivity: {reading['ec']}")
                if reading.get('n'): context_parts.append(f"  Nitrogen: {reading['n']} ppm")
                if reading.get('p'): context_parts.append(f"  Phosphorus: {reading['p']} ppm")
                if reading.get('k'): context_parts.append(f"  Potassium: {reading['k']} ppm")
                context_parts.append("")
        
        # Weather data section
        if weather_data:
            context_parts.append("\n🌤️ WEATHER CONDITIONS:")
            context_parts.append(f"Current Weather: {json.dumps(weather_data, indent=2)}")
        
        # Historical data section
        if historical_data:
            context_parts.append(f"\n📈 HISTORICAL CONTEXT ({len(historical_data)} data points):")
            context_parts.append("Previous interactions and outcomes available for trend analysis.")
        
        # Farm context section
        if farm_context:
            context_parts.append("\n🚜 FARM CONTEXT:")
            context_parts.append(f"Farm Summary: {json.dumps(farm_context, indent=2)}")
        
        if len(context_parts) == 1:  # Only the header
            context_parts.append("\n⚠️ Limited data available. Will ask clarifying questions as needed.")
        
        return "\n".join(context_parts)
    
    def _structure_advice(self, response: str, original_query: str) -> Dict[str, Any]:
        """Structure the AI response into organized advice"""
        return {
            "original_query": original_query,
            "full_response": response,
            "timestamp": datetime.now().isoformat(),
            "confidence_level": self._extract_confidence(response),
            "action_items": self._extract_actions(response),
            "follow_up_questions": self._extract_questions(response),
            "data_based": "sensor data" in response.lower() or "weather" in response.lower()
        }
    
    def _extract_confidence(self, response: str) -> str:
        """Extract confidence level from response"""
        response_lower = response.lower()
        if "highly confident" in response_lower or "certain" in response_lower:
            return "high"
        elif "moderately confident" in response_lower or "likely" in response_lower:
            return "medium"  
        elif "uncertain" in response_lower or "need more" in response_lower:
            return "low"
        else:
            return "medium"  # default
    
    def _extract_actions(self, response: str) -> List[str]:
        """Extract action items from response"""
        actions = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for action-oriented lines
            if any(word in line.lower() for word in ["should", "recommend", "apply", "water", "fertilize", "test", "monitor"]):
                if len(line) > 10 and not line.startswith("#"):  # Avoid headers
                    actions.append(line)
        
        return actions[:5]  # Return top 5 actions
    
    def _extract_questions(self, response: str) -> List[str]:
        """Extract follow-up questions from response"""
        questions = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.endswith('?') and len(line) > 10:
                questions.append(line)
        
        return questions[:3]  # Return up to 3 questions
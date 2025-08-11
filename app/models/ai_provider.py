# app/models/enhanced_ai_provider.py
from groq import Groq
import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import asyncio
from dataclasses import dataclass

@dataclass
class InteractionContext:
    session_id: str
    user_query: str
    sensor_data: List[Dict]
    weather_data: Dict
    farm_context: Dict
    historical_patterns: List[Dict]
    knowledge_base_results: List[Dict]

class EnhancedAgriculturalAI:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be set in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Enhanced model selection
        self.models = {
            "fast": "llama3-8b-8192",
            "balanced": "llama3-70b-8192", 
            "smart": "mixtral-8x7b-32768",
            "reasoning": "llama3-70b-8192"  # For complex reasoning tasks
        }
        
        # Conversation patterns for learning
        self.conversation_patterns = {}
        self.user_feedback_history = {}
        
        # Agricultural expertise domains
        self.expertise_domains = {
            "soil_management": ["ph", "nutrients", "moisture", "temperature", "ec"],
            "pest_control": ["insects", "diseases", "prevention", "treatment"],
            "crop_management": ["planting", "growth", "harvest", "varieties"],
            "irrigation": ["water", "scheduling", "efficiency", "drainage"],
            "fertilization": ["npk", "organic", "timing", "application"],
            "weather_response": ["adaptation", "protection", "timing"]
        }
        
    async def generate_contextual_advice(self, context: InteractionContext) -> Dict[str, Any]:
        """Generate comprehensive agricultural advice with full context awareness"""
        try:
            # Analyze context complexity to select appropriate model
            complexity_score = self._assess_context_complexity(context)
            model = self._select_optimal_model(context.user_query, complexity_score)
            
            # Build comprehensive system prompt
            system_prompt = await self._build_enhanced_system_prompt(context)
            
            # Create structured input with all available data
            user_prompt = await self._build_contextual_user_prompt(context)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Add conversation history for context
            if context.session_id:
                historical_messages = await self._get_relevant_history(
                    context.session_id, context.user_query
                )
                messages.extend(historical_messages)
            
            # Generate response with enhanced parameters
            response = await self._generate_enhanced_response(
                messages, model, complexity_score
            )
            
            if response["success"]:
                # Post-process and structure the advice
                structured_advice = await self._structure_enhanced_advice(
                    response["response"], context
                )
                response["structured_advice"] = structured_advice
                
                # Learn from interaction patterns
                await self._learn_from_interaction(context, response)
            
            return response
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I'm experiencing technical difficulties. Let me try to help with the information I can process.",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _build_enhanced_system_prompt(self, context: InteractionContext) -> str:
        """Build an enhanced system prompt with domain expertise"""
        
        # Identify relevant expertise domains
        relevant_domains = self._identify_relevant_domains(context.user_query)
        
        base_prompt = """You are an advanced agricultural AI advisor with comprehensive expertise in modern farming practices. Your role is to provide precise, actionable, and farmer-friendly advice based on real-time data and scientific principles.

CORE EXPERTISE AREAS:
- Precision agriculture and sensor data interpretation
- Soil science and nutrient management
- Integrated pest and disease management
- Climate-smart agriculture and weather adaptation
- Sustainable farming practices
- Crop physiology and growth optimization
- Water management and irrigation efficiency

RESPONSE METHODOLOGY:
1. Data-Driven Analysis: Always base recommendations on available sensor data, weather conditions, and historical patterns
2. Risk Assessment: Evaluate potential risks and provide preventive measures
3. Economic Consideration: Consider cost-effectiveness and resource optimization
4. Sustainability Focus: Prioritize long-term soil health and environmental impact
5. Practical Implementation: Provide step-by-step actionable advice
6. Continuous Monitoring: Suggest monitoring points and follow-up actions

COMMUNICATION STYLE:
- Use clear, practical language that farmers can easily understand
- Provide specific measurements, timings, and quantities when possible
- Explain the reasoning behind recommendations
- Acknowledge uncertainty when data is insufficient
- Ask clarifying questions to improve advice quality"""

        # Add domain-specific expertise
        if relevant_domains:
            base_prompt += f"\n\nCURRENT FOCUS AREAS: {', '.join(relevant_domains)}"
            
            for domain in relevant_domains:
                if domain == "soil_management":
                    base_prompt += "\n\nSOIL MANAGEMENT EXPERTISE:\n- Optimal pH ranges for different crops\n- Nutrient deficiency identification and correction\n- Soil structure and organic matter management\n- Salinity and sodicity management"
                elif domain == "pest_control":
                    base_prompt += "\n\nPEST MANAGEMENT EXPERTISE:\n- Integrated Pest Management (IPM) principles\n- Beneficial insect identification and conservation\n- Disease cycle understanding and intervention points\n- Resistance management strategies"
                elif domain == "irrigation":
                    base_prompt += "\n\nIRRIGATION EXPERTISE:\n- Soil moisture monitoring and interpretation\n- Crop water requirements by growth stage\n- Irrigation scheduling optimization\n- Water use efficiency techniques"
        
        # Add data context awareness
        if context.sensor_data:
            base_prompt += "\n\nYou have access to real-time sensor data including soil conditions, nutrients, and environmental parameters."
        
        if context.weather_data:
            base_prompt += "\n\nCurrent and forecast weather data is available for informed decision-making."
        
        if context.historical_patterns:
            base_prompt += "\n\nHistorical data patterns are available to inform trend analysis and predictions."
        
        base_prompt += "\n\nALWAYS provide confidence levels for your recommendations and suggest when additional data or expert consultation might be needed."
        
        return base_prompt
    
    async def _build_contextual_user_prompt(self, context: InteractionContext) -> str:
        """Build comprehensive user prompt with all available context"""
        
        prompt_parts = ["FARMER'S QUESTION:", context.user_query, "\n"]
        
        # Add sensor data analysis
        if context.sensor_data:
            prompt_parts.extend([
                "CURRENT SENSOR READINGS:",
                self._format_sensor_data(context.sensor_data),
                ""
            ])
        
        # Add weather context
        if context.weather_data:
            prompt_parts.extend([
                "WEATHER CONDITIONS:",
                json.dumps(context.weather_data, indent=2),
                ""
            ])
        
        # Add farm context
        if context.farm_context:
            prompt_parts.extend([
                "FARM CONTEXT:",
                json.dumps(context.farm_context, indent=2),
                ""
            ])
        
        # Add relevant knowledge base information
        if context.knowledge_base_results:
            prompt_parts.extend([
                "RELEVANT AGRICULTURAL KNOWLEDGE:",
                self._format_knowledge_results(context.knowledge_base_results),
                ""
            ])
        
        # Add historical patterns if available
        if context.historical_patterns:
            prompt_parts.extend([
                "HISTORICAL PATTERNS:",
                self._format_historical_patterns(context.historical_patterns),
                ""
            ])
        
        # Add specific analysis requests
        prompt_parts.extend([
            "PLEASE PROVIDE:",
            "1. Immediate assessment of current conditions",
            "2. Specific actionable recommendations with timing",
            "3. Risk factors and preventive measures",
            "4. Monitoring suggestions and follow-up actions",
            "5. Confidence level for each recommendation",
            ""
        ])
        
        return "\n".join(prompt_parts)
    
    def _format_sensor_data(self, sensor_data: List[Dict]) -> str:
        """Format sensor data for prompt inclusion"""
        if not sensor_data:
            return "No recent sensor data available"
        
        formatted = []
        for i, reading in enumerate(sensor_data[:5]):  # Last 5 readings
            timestamp = reading.get('timestamp', 'Unknown')
            formatted.append(f"Reading {i+1} ({timestamp}):")
            
            for param in ['soil_moisture', 'soil_temperature', 'ph', 'ec', 'n', 'p', 'k']:
                if param in reading and reading[param] is not None:
                    unit = self._get_parameter_unit(param)
                    formatted.append(f"  {param.replace('_', ' ').title()}: {reading[param]}{unit}")
            
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _get_parameter_unit(self, param: str) -> str:
        """Get appropriate unit for sensor parameters"""
        units = {
            'soil_moisture': '%',
            'soil_temperature': '°C',
            'ph': '',
            'ec': ' dS/m',
            'n': ' ppm',
            'p': ' ppm',
            'k': ' ppm'
        }
        return units.get(param, '')
    
    def _format_knowledge_results(self, knowledge_results: List[Dict]) -> str:
        """Format knowledge base results for prompt inclusion"""
        formatted = []
        for i, result in enumerate(knowledge_results[:3]):  # Top 3 most relevant
            formatted.append(f"Knowledge {i+1} (Relevance: {result.get('similarity', 0):.2f}):")
            formatted.append(result.get('content', 'No content available'))
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _format_historical_patterns(self, patterns: List[Dict]) -> str:
        """Format historical patterns for prompt inclusion"""
        if not patterns:
            return "No significant historical patterns identified"
        
        formatted = ["Recent trends and patterns:"]
        for pattern in patterns[:3]:  # Top 3 patterns
            formatted.append(f"- {pattern.get('description', 'Pattern identified')}")
            if 'confidence' in pattern:
                formatted.append(f"  (Confidence: {pattern['confidence']})")
        
        return "\n".join(formatted)
    
    def _assess_context_complexity(self, context: InteractionContext) -> float:
        """Assess the complexity of the context to determine processing needs"""
        complexity_score = 0.0
        
        # Query complexity
        query_length = len(context.user_query.split())
        if query_length > 20:
            complexity_score += 0.2
        elif query_length > 10:
            complexity_score += 0.1
        
        # Data richness
        if context.sensor_data:
            complexity_score += 0.2
        if context.weather_data:
            complexity_score += 0.1
        if context.historical_patterns:
            complexity_score += 0.2
        if context.knowledge_base_results:
            complexity_score += 0.1
        
        # Domain complexity
        domains = self._identify_relevant_domains(context.user_query)
        if len(domains) > 2:
            complexity_score += 0.2
        elif len(domains) > 1:
            complexity_score += 0.1
        
        return min(1.0, complexity_score)
    
    def _identify_relevant_domains(self, query: str) -> List[str]:
        """Identify relevant agricultural domains from the query"""
        query_lower = query.lower()
        relevant_domains = []
        
        for domain, keywords in self.expertise_domains.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_domains.append(domain)
        
        return relevant_domains
    
    def _select_optimal_model(self, query: str, complexity_score: float) -> str:
        """Select the optimal model based on query and complexity"""
        query_lower = query.lower()
        
        # Use reasoning model for complex analysis
        if complexity_score > 0.7:
            return self.models["reasoning"]
        
        # Use smart model for technical questions
        if any(word in query_lower for word in ["analyze", "predict", "calculate", "optimize", "recommend"]):
            return self.models["smart"]
        
        # Use fast model for simple questions
        if len(query.split()) < 10 and "?" in query:
            return self.models["fast"]
        
        # Default to balanced
        return self.models["balanced"]
    
    async def _generate_enhanced_response(self, messages: List[Dict], model: str, complexity_score: float) -> Dict:
        """Generate response with enhanced parameters based on complexity"""
        try:
            # Adjust parameters based on complexity
            max_tokens = int(1024 + (complexity_score * 1024))  # 1024-2048 tokens
            temperature = 0.1 + (complexity_score * 0.2)  # 0.1-0.3 temperature
            
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
                "complexity_score": complexity_score,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I'm having trouble processing your request. Please try rephrasing your question.",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _structure_enhanced_advice(self, response: str, context: InteractionContext) -> Dict[str, Any]:
        """Structure the AI response into enhanced advice format"""
        
        # Extract different types of information from response
        advice_structure = {
            "original_query": context.user_query,
            "full_response": response,
            "timestamp": datetime.now().isoformat(),
            "confidence_level": self._extract_confidence_advanced(response),
            "immediate_actions": self._extract_immediate_actions(response),
            "short_term_actions": self._extract_short_term_actions(response),
            "long_term_strategies": self._extract_long_term_strategies(response),
            "monitoring_points": self._extract_monitoring_points(response),
            "risk_factors": self._extract_risk_factors(response),
            "follow_up_questions": self._extract_questions(response),
            "data_requirements": self._identify_data_gaps(response, context),
            "economic_considerations": self._extract_economic_factors(response),
            "sustainability_notes": self._extract_sustainability_factors(response)
        }
        
        return advice_structure
    
    def _extract_confidence_advanced(self, response: str) -> Dict[str, Any]:
        """Extract advanced confidence information"""
        response_lower = response.lower()
        
        # Overall confidence
        if any(phrase in response_lower for phrase in ["highly confident", "certain", "definitely"]):
            overall = "high"
        elif any(phrase in response_lower for phrase in ["moderately confident", "likely", "probably"]):
            overall = "medium"
        elif any(phrase in response_lower for phrase in ["uncertain", "unclear", "need more", "insufficient"]):
            overall = "low"
        else:
            overall = "medium"
        
        # Confidence factors
        factors = []
        if "based on sensor data" in response_lower:
            factors.append("sensor_data_available")
        if "weather forecast" in response_lower:
            factors.append("weather_data_available")
        if "historical" in response_lower:
            factors.append("historical_data_available")
        if "need more information" in response_lower:
            factors.append("additional_data_needed")
        
        return {
            "level": overall,
            "factors": factors,
            "recommendations_with_data": len([f for f in factors if f.endswith("_available")]) > 0
        }
    
    def _extract_immediate_actions(self, response: str) -> List[str]:
        """Extract immediate action items"""
        actions = []
        lines = response.split('\n')
        
        immediate_keywords = ["immediately", "right now", "urgent", "today", "asap"]
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in immediate_keywords):
                if len(line) > 15 and not line.startswith('#'):
                    actions.append(line)
        
        return actions[:3]
    
    def _extract_short_term_actions(self, response: str) -> List[str]:
        """Extract short-term action items (1-7 days)"""
        actions = []
        lines = response.split('\n')
        
        short_term_keywords = ["within", "next few days", "this week", "in 2-3 days", "tomorrow"]
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in short_term_keywords):
                if len(line) > 15 and not line.startswith('#'):
                    actions.append(line)
        
        return actions[:5]
    
    def _extract_long_term_strategies(self, response: str) -> List[str]:
        """Extract long-term strategic recommendations"""
        strategies = []
        lines = response.split('\n')
        
        long_term_keywords = ["long-term", "future", "next season", "ongoing", "establish"]
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in long_term_keywords):
                if len(line) > 15 and not line.startswith('#'):
                    strategies.append(line)
        
        return strategies[:3]
    
    def _extract_monitoring_points(self, response: str) -> List[str]:
        """Extract monitoring and observation points"""
        monitoring = []
        lines = response.split('\n')
        
        monitoring_keywords = ["monitor", "check", "observe", "measure", "track", "watch"]
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in monitoring_keywords):
                if len(line) > 15 and not line.startswith('#'):
                    monitoring.append(line)
        
        return monitoring[:4]
    
    def _extract_risk_factors(self, response: str) -> List[str]:
        """Extract identified risk factors"""
        risks = []
        lines = response.split('\n')
        
        risk_keywords = ["risk", "danger", "threat", "problem", "concern", "warning"]
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in risk_keywords):
                if len(line) > 15 and not line.startswith('#'):
                    risks.append(line)
        
        return risks[:3]
    
    def _identify_data_gaps(self, response: str, context: InteractionContext) -> List[str]:
        """Identify what additional data would improve the advice"""
        gaps = []
        response_lower = response.lower()
        
        if "more information" in response_lower or "additional data" in response_lower:
            if not context.sensor_data:
                gaps.append("soil_sensor_data")
            if not context.weather_data:
                gaps.append("weather_forecast")
            if not context.farm_context:
                gaps.append("farm_details")
            if "crop type" in response_lower:
                gaps.append("crop_information")
            if "growth stage" in response_lower:
                gaps.append("growth_stage_details")
        
        return gaps
    
    def _extract_economic_factors(self, response: str) -> List[str]:
        """Extract economic considerations"""
        economic = []
        lines = response.split('\n')
        
        economic_keywords = ["cost", "price", "budget", "economic", "profit", "expense", "investment"]
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in economic_keywords):
                if len(line) > 15 and not line.startswith('#'):
                    economic.append(line)
        
        return economic[:3]
    
    def _extract_sustainability_factors(self, response: str) -> List[str]:
        """Extract sustainability considerations"""
        sustainability = []
        lines = response.split('\n')
        
        sustainability_keywords = ["sustainable", "environment", "organic", "natural", "ecosystem", "soil health"]
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in sustainability_keywords):
                if len(line) > 15 and not line.startswith('#'):
                    sustainability.append(line)
        
        return sustainability[:3]
    
    def _extract_questions(self, response: str) -> List[str]:
        """Extract follow-up questions"""
        questions = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.endswith('?') and len(line) > 10:
                questions.append(line)
        
        return questions[:3]
    
    async def _get_relevant_history(self, session_id: str, current_query: str) -> List[Dict]:
        """Get relevant conversation history for context"""
        # This would integrate with your memory system
        # Placeholder for now - implement based on your memory.py
        return []
    
    async def _learn_from_interaction(self, context: InteractionContext, response: Dict):
        """Learn from user interactions for continuous improvement"""
        try:
            # Store interaction patterns
            pattern_key = f"interaction_{context.session_id}_{datetime.now().isoformat()}"
            
            interaction_data = {
                "query_type": self._classify_query_type(context.user_query),
                "domains": self._identify_relevant_domains(context.user_query),
                "data_available": {
                    "sensor": bool(context.sensor_data),
                    "weather": bool(context.weather_data),
                    "historical": bool(context.historical_patterns)
                },
                "response_complexity": response.get("complexity_score", 0),
                "model_used": response.get("model_used"),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store for pattern analysis
            self.conversation_patterns[pattern_key] = interaction_data
            
            # Cleanup old patterns (keep last 1000)
            if len(self.conversation_patterns) > 1000:
                oldest_keys = sorted(self.conversation_patterns.keys())[:100]
                for key in oldest_keys:
                    del self.conversation_patterns[key]
                    
        except Exception as e:
            print(f"Error in learning from interaction: {e}")
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of agricultural query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["disease", "pest", "insect", "fungus"]):
            return "pest_disease"
        elif any(word in query_lower for word in ["water", "irrigation", "moisture", "drought"]):
            return "irrigation"
        elif any(word in query_lower for word in ["fertilizer", "nutrient", "nitrogen", "phosphorus"]):
            return "fertilization"
        elif any(word in query_lower for word in ["soil", "ph", "ec", "temperature"]):
            return "soil_management"
        elif any(word in query_lower for word in ["weather", "rain", "temperature", "climate"]):
            return "weather_related"
        elif any(word in query_lower for word in ["plant", "grow", "harvest", "crop"]):
            return "crop_management"
        else:
            return "general"
    
    async def process_user_feedback(self, session_id: str, interaction_id: str, 
                                  feedback_type: str, rating: Optional[int] = None,
                                  comments: Optional[str] = None) -> Dict[str, Any]:
        """Process user feedback for continuous improvement"""
        try:
            feedback_data = {
                "session_id": session_id,
                "interaction_id": interaction_id,
                "feedback_type": feedback_type,
                "rating": rating,
                "comments": comments,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store feedback for analysis
            feedback_key = f"feedback_{session_id}_{interaction_id}"
            self.user_feedback_history[feedback_key] = feedback_data
            
            # Analyze feedback patterns
            improvement_insights = await self._analyze_feedback_patterns()
            
            return {
                "feedback_stored": True,
                "insights": improvement_insights,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "feedback_stored": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze feedback patterns for improvement insights"""
        if not self.user_feedback_history:
            return {"status": "insufficient_data"}
        
        # Analyze recent feedback
        recent_feedback = list(self.user_feedback_history.values())[-50:]  # Last 50 pieces of feedback
        
        # Calculate metrics
        total_feedback = len(recent_feedback)
        positive_feedback = len([f for f in recent_feedback if f.get("rating", 0) >= 4])
        negative_feedback = len([f for f in recent_feedback if f.get("rating", 0) <= 2])
        
        # Identify common issues
        common_issues = {}
        for feedback in recent_feedback:
            if feedback.get("feedback_type") in ["not_helpful", "incorrect"]:
                comments = feedback.get("comments", "").lower()
                # Simple keyword analysis for common issues
                if "data" in comments:
                    common_issues["data_quality"] = common_issues.get("data_quality", 0) + 1
                if "understand" in comments:
                    common_issues["clarity"] = common_issues.get("clarity", 0) + 1
                if "wrong" in comments or "incorrect" in comments:
                    common_issues["accuracy"] = common_issues.get("accuracy", 0) + 1
        
        return {
            "total_feedback": total_feedback,
            "satisfaction_rate": (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0,
            "improvement_areas": common_issues,
            "needs_attention": negative_feedback > (total_feedback * 0.3)  # More than 30% negative
        }
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning and improvement system"""
        return {
            "conversation_patterns_stored": len(self.conversation_patterns),
            "feedback_entries": len(self.user_feedback_history),
            "domains_encountered": len(set(
                pattern.get("query_type", "unknown") 
                for pattern in self.conversation_patterns.values()
            )),
            "last_learning_update": datetime.now().isoformat()
        }
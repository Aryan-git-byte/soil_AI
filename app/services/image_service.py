# app/services/image_service.py
import base64
import os
from typing import Optional, Dict
from datetime import datetime
import httpx
from app.core.config import OPENROUTER_KEYS

class ImageAnalysisService:
    """
    Handles image upload and AI-powered analysis for agricultural use cases.
    Supports: crop disease detection, plant health assessment, soil analysis, etc.
    """

    SUPPORTED_FORMATS = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    MAX_SIZE_MB = 10

    @staticmethod
    def encode_image_to_base64(image_bytes: bytes) -> str:
        """
        Encode image bytes to base64 string.
        """
        return base64.b64encode(image_bytes).decode('utf-8')

    @staticmethod
    def validate_image(content_type: str, size: int) -> tuple[bool, Optional[str]]:
        """
        Validate image format and size.
        Returns (is_valid, error_message)
        """
        if content_type not in ImageAnalysisService.SUPPORTED_FORMATS:
            return False, f"Unsupported format. Supported: {', '.join(ImageAnalysisService.SUPPORTED_FORMATS)}"
        
        max_size_bytes = ImageAnalysisService.MAX_SIZE_MB * 1024 * 1024
        if size > max_size_bytes:
            return False, f"Image too large. Max size: {ImageAnalysisService.MAX_SIZE_MB}MB"
        
        return True, None

    @staticmethod
    async def analyze_image(
        image_base64: str,
        media_type: str,
        query: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Send image to AI model for analysis.
        
        Args:
            image_base64: Base64 encoded image
            media_type: MIME type (e.g., "image/jpeg")
            query: User's question about the image
            context: Optional additional context (location, soil data, etc.)
        """
        
        # Build the prompt with context
        prompt = f"""You are FarmBot Nova, an expert agricultural assistant.

Analyze this image and answer the user's question.

User Question: {query}

"""
        
        if context:
            prompt += f"\nAdditional Context:\n{context}\n"
        
        prompt += """
Provide detailed, actionable insights. If you detect:
- **Crop Disease**: Identify the disease, severity, and treatment recommendations
- **Plant Health**: Assess overall health, nutrient deficiencies, growth stage
- **Soil Quality**: Analyze texture, color, moisture, and visible characteristics
- **Pest Infestation**: Identify pests and suggest organic/chemical control methods
- **Weed Identification**: Name the weed species and removal strategies

Be specific, practical, and farmer-friendly.
"""

        # Prepare messages for vision model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        # Try with each API key
        for i, key in enumerate(OPENROUTER_KEYS):
            try:
                print(f"[ImageAnalysis] Trying key {i+1}/{len(OPENROUTER_KEYS)}...")
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://farmbot.com",
                            "X-Title": "FarmBot Nova"
                        },
                        json={
                            "model": "amazon/nova-2-lite-v1:free",  # Vision-capable model
                            "messages": messages
                        }
                    )

                    if response.status_code == 200:
                        print(f"[ImageAnalysis] ✅ Key {i+1} succeeded")
                        result = response.json()
                        return {
                            "success": True,
                            "analysis": result["choices"][0]["message"]["content"],
                            "model": "google/gemini-2.0-flash-exp:free"
                        }
                    else:
                        error_msg = response.text[:200] if response.text else "No error body"
                        print(f"[ImageAnalysis] Key {i+1} failed: {response.status_code} - {error_msg}")

            except Exception as e:
                print(f"[ImageAnalysis] Key {i+1} exception: {str(e)[:150]}")

        return {
            "success": False,
            "error": "All API keys failed. Please try again later."
        }

    @staticmethod
    def get_analysis_suggestions(analysis_type: str = "general") -> list:
        """
        Provide suggested questions based on analysis type.
        """
        suggestions = {
            "disease": [
                "What disease does this plant have?",
                "How severe is this infection?",
                "What treatment should I use?",
                "Is this disease spreading?"
            ],
            "health": [
                "Is my crop healthy?",
                "What nutrients is my plant lacking?",
                "What stage of growth is this?",
                "Should I be concerned about anything?"
            ],
            "soil": [
                "What type of soil is this?",
                "Is the soil moisture adequate?",
                "What's the soil texture?",
                "Is this soil suitable for planting?"
            ],
            "pest": [
                "What pest is this?",
                "How do I control this pest?",
                "Is this infestation serious?",
                "Organic pest control options?"
            ],
            "weed": [
                "What weed species is this?",
                "How should I remove these weeds?",
                "Are these weeds harmful to my crops?",
                "Pre or post-emergence control?"
            ],
            "general": [
                "Analyze this image",
                "What do you see in this picture?",
                "Any concerns about this crop?",
                "What recommendations do you have?"
            ]
        }
        
        return suggestions.get(analysis_type, suggestions["general"])
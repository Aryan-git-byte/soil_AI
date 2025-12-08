# Configuration and API key management
import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API keys
OPENROUTER_KEYS = [
    os.getenv("OPENROUTER_API_KEY_1"),
    os.getenv("OPENROUTER_API_KEY_2"),
    os.getenv("OPENROUTER_API_KEY_3"),
]

OPENROUTER_KEYS = [k for k in OPENROUTER_KEYS if k]

# Other API keys
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
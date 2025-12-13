#!/bin/bash

# ======================================
# FARMBOT NOVA - PRODUCTION START SCRIPT (RENDER)
# ======================================

echo "🚀 Starting FarmBot Nova Backend (Production)"
echo "Platform: Render"
echo "=" 

# Set environment
export ENVIRONMENT=production

# Validate critical environment variables
echo "🔍 Validating environment variables..."

REQUIRED_VARS=(
    "OPENROUTER_API_KEY_1"
    "QDRANT_URL"
    "QDRANT_API_KEY"
    "SUPABASE_URL"
    "SUPABASE_KEY"
    "ALLOWED_ORIGINS"
    "ADMIN_SECRET"
    "API_KEYS"
)

MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
        echo "❌ Missing: $var"
    else
        echo "✓ Found: $var"
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo ""
    echo "❌ CRITICAL: Missing required environment variables:"
    printf '   - %s\n' "${MISSING_VARS[@]}"
    echo ""
    echo "Set these in Render Dashboard → Environment Variables"
    exit 1
fi

echo ""
echo "✅ All required environment variables present"
echo ""

# Optional variables check
echo "⚙️  Checking optional variables..."

if [ -z "$OPENWEATHER_API_KEY" ]; then
    echo "⚠️  OPENWEATHER_API_KEY not set (weather features disabled)"
fi

if [ -z "$TAVILY_API_KEY" ]; then
    echo "⚠️  TAVILY_API_KEY not set (web search disabled)"
fi

if [ -z "$SOIL_SAND_FILE" ]; then
    echo "⚠️  Soil data files not configured (soil analysis disabled)"
fi

echo ""
echo "=" 
echo "🌾 FarmBot Nova Configuration:"
echo "  Environment: $ENVIRONMENT"
echo "  CORS Origins: $ALLOWED_ORIGINS"
echo "  Qdrant: $QDRANT_URL"
echo "  Supabase: $SUPABASE_URL"
echo "=" 
echo ""

# Start server with production settings
echo "🚀 Starting Uvicorn server..."
echo ""

# Render uses PORT environment variable
PORT=${PORT:-8000}

# Use Uvicorn with production settings
# - No --reload (causes memory leaks)
# - Multiple workers for concurrency
# - Proper logging
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 2 \
    --log-level info \
    --no-access-log \
    --proxy-headers \
    --forwarded-allow-ips='*'
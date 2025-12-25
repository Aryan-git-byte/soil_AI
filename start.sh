#!/bin/bash

# =============

echo "🚀 Starting FarmBot Nova Backend v1.0.0 (Production)"
echo "Platform: Render"
echo "=" 

# Set environment
export ENVIRONMENT=production

# Validate critical environment variables
echo "🔍 Validating environment variables..."

REQUIRED_VARS=(
    "GROQ_API_KEY"
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
echo "🌾 FarmBot Nova v1.0.0 Configuration:"
echo "  Environment: $ENVIRONMENT"
echo "  CORS Origins: $ALLOWED_ORIGINS"
echo "  Qdrant: $QDRANT_URL"
echo "  Supabase: $SUPABASE_URL"
echo "  Security Headers: ENABLED"
echo "=" 
echo ""

# ✅ FIX #1: Use Gunicorn with Uvicorn workers (PRODUCTION BEST PRACTICE)
echo "🚀 Starting Gunicorn with Uvicorn workers..."
echo ""

# Render uses PORT environment variable
PORT=${PORT:-8000}

# Calculate optimal worker count
# Render starter plan has 0.5 CPU, so use 2 workers
# For higher plans: workers = 2 * CPU_cores + 1
WORKERS=${WORKERS:-1}

# ✅ MEMORY-OPTIMIZED CONFIGURATION FOR FREE TIER
exec gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 1 \
    --worker-tmp-dir /dev/shm \
    --bind 0.0.0.0:$PORT \
    --max-requests 500 \
    --timeout 60 \
    --graceful-timeout 30 \
    --preload \
    --log-level warning \
    --access-logfile - \
    --error-logfile -

# Explanation of flags:
# --worker-class: Use Uvicorn workers for async support
# --workers: Number of worker processes (2 for starter plan)
# --max-requests: Restart workers after N requests (prevents memory leaks)
# --max-requests-jitter: Add randomness to prevent simultaneous restarts
# --timeout: Worker timeout (30s for AI API calls)
# --keepalive: Keep connections alive for 5s
# --graceful-timeout: Time to finish requests during shutdown
# --preload: Load app before forking (saves memory)
# --log-level: Info level logging
# --access-logfile/-error-logfile: Log to stdout/stderr
# --capture-output: Capture worker output
# --enable-stdio-inheritance: Inherit stdio from parent
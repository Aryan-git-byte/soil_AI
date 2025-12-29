#!/bin/bash

# ======================================
# FARMBOT NOVA - 512MB RAM OPTIMIZED
# ======================================

echo "🚀 Starting FarmBot Nova Backend (512MB Optimized)"
echo "Platform: Render Free Tier"
echo "========================================="

# Set environment
export ENVIRONMENT=production

# ✅ CRITICAL: Memory optimizations
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MALLOC_ARENA_MAX=2  # Reduce glibc memory fragmentation

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

echo ""
echo "========================================="
echo "🌾 FarmBot Nova Configuration:"
echo "  Environment: $ENVIRONMENT"
echo "  RAM Limit: 512MB"
echo "  Workers: 1 (memory optimized)"
echo "  Model: FastEmbed ONNX (~50-80MB)"
echo "  Raster Files: 82MB"
echo "========================================="
echo ""

# ✅ MEMORY-CRITICAL CONFIGURATION
PORT=${PORT:-8000}

echo "🚀 Starting Gunicorn (512MB optimized)..."
echo ""

exec gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 1 \
    --threads 2 \
    --worker-tmp-dir /dev/shm \
    --bind 0.0.0.0:$PORT \
    --max-requests 300 \
    --max-requests-jitter 50 \
    --timeout 60 \
    --graceful-timeout 30 \
    --preload \
    --log-level warning \
    --access-logfile - \
    --error-logfile -

# ========================================
# OPTIMIZATION NOTES (512MB RAM):
# ========================================
# 
# --workers 1             → Single worker (critical for 512MB)
# --threads 2             → Handle 2 concurrent requests per worker
# --worker-tmp-dir /dev/shm → Use RAM disk for worker files
# --max-requests 300      → Restart worker after 300 requests (prevent memory leaks)
# --max-requests-jitter 50 → Add randomness to prevent simultaneous restarts
# --timeout 60            → 60s timeout for slow AI API calls
# --preload               → Load app before forking (saves memory)
# --log-level warning     → Minimal logging (saves CPU/IO)
#
# ========================================
# EXPECTED MEMORY USAGE:
# ========================================
# 
# FastEmbed model:         50-80MB   ✅
# Raster files (mmap):     ~10MB     ✅ (memory-mapped, not fully loaded)
# FastAPI + deps:          100MB     ✅
# Qdrant client:           30MB      ✅
# Supabase client:         20MB      ✅
# OS + Python runtime:     150MB     ✅
# Request buffer:          80MB      ✅
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOTAL:                   440-500MB ✅ (Safe margin!)
# ========================================
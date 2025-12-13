# ======================================
# FARMBOT NOVA - PRODUCTION DOCKERFILE
# ======================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for rasterio
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY start.sh .

# Make start script executable
RUN chmod +x start.sh

# Expose port (Render sets this via PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["./start.sh"]
```

---

## 12. **.dockerignore**
```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/

# Virtual environments
venv/
env/
.venv/

# Environment files
.env
.env.*

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data files (too large for container)
*.tif
*.pdf
knowledge_pipeline/sources/*.pdf

# Git
.git/
.gitignore

# Documentation
README.md
*.md

# Testing
.pytest_cache/
.coverage
htmlcov/

# Notebooks
*.ipynb
knowledge_pipeline/

# Temporary
temp/
tmp/
*.log
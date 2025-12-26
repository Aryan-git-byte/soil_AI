#!/usr/bin/env python3
"""
Run the FastAPI app locally using Uvicorn with safe defaults for local development.

Usage:
  python run_server.py --host 127.0.0.1 --port 8000 --reload

This script will set minimal environment variables if they are missing to avoid
startup crashes (e.g. ALLOWED_ORIGINS and ADMIN_SECRET). It is intended for
local development only. Do NOT use these defaults in production.
"""
import os
import argparse
import sys

def ensure_env_defaults():
    # Provide safe defaults for local development so app doesn't raise on startup
    os.environ.setdefault("ENVIRONMENT", os.environ.get("ENVIRONMENT", "development"))
    os.environ.setdefault("ALLOWED_ORIGINS", os.environ.get("ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000"))
    os.environ.setdefault("ADMIN_SECRET", os.environ.get("ADMIN_SECRET", "dev-admin-secret"))
    os.environ.setdefault("API_KEYS", os.environ.get("API_KEYS", "dev-api-key"))

def parse_args():
    p = argparse.ArgumentParser(description="Run the FastAPI app locally with Uvicorn")
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    p.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    p.add_argument("--workers", type=int, default=1, help="Number of workers (uvicorn only supports multiple workers via the command line)")
    return p.parse_args()

def main():
    ensure_env_defaults()
    args = parse_args()

    # Import here so env defaults are set before app imports (if they matter)
    try:
        import uvicorn
    except Exception as e:
        print("uvicorn is required. Install with: pip install uvicorn", file=sys.stderr)
        raise

    app_location = "app.main:app"
    print(f"Starting server -> {app_location} on http://{args.host}:{args.port} (reload={args.reload})")

    # Use uvicorn.run for single-process dev server
    uvicorn.run(app_location, host=args.host, port=args.port, reload=args.reload)

if __name__ == '__main__':
    main()

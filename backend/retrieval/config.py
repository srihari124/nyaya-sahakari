import os

# Keep workers=1 to avoid model memory duplication across worker processes.
UVICORN_WORKERS = int(os.getenv("UVICORN_WORKERS", "1"))
UVICORN_HOST = os.getenv("UVICORN_HOST", "0.0.0.0")
UVICORN_PORT = int(os.getenv("UVICORN_PORT", "8000"))


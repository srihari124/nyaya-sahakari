import uvicorn

from retrieval.src.api import app
from retrieval.config import UVICORN_WORKERS, UVICORN_HOST, UVICORN_PORT


if __name__ == "__main__":
    # Keep workers=1 for Apple Silicon local runs:
    # multiple workers duplicate transformer model memory and can freeze the system.
    uvicorn.run(app, host=UVICORN_HOST, port=UVICORN_PORT, workers=UVICORN_WORKERS)

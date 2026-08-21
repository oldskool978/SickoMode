from fastapi import FastAPI
from api.endpoints import router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Psychoacoustic Multi-Band Limiter API",
        version="1.0.0",
        description="Mel/CELT-Scale Transient-Locked Dynamic Limiting Service"
    )
    app.include_router(router, prefix="/api/v1")
    return app

app = create_app()
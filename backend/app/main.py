from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.matches import router as matches_router
from app.api.routes.predictions import router as predictions_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="IPL Local Predictor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, tags=["health"])
app.include_router(matches_router, prefix="/matches", tags=["matches"])
app.include_router(predictions_router, prefix="/predictions", tags=["predictions"])
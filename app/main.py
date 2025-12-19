from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.voice import router as voice_router

app = FastAPI(title="Voice Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(voice_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Voice Chatbot API"}
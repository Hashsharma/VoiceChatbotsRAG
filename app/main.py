from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.voice import router as voice_router
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI(title="Voice Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(voice_router, prefix="/api/v1")

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=BASE_DIR / "templates")
@app.get("/")
async def root():
    return {"message": "Voice Chatbot API"}

@app.get("/voicebot", response_class=HTMLResponse)
async def voice_chat_page(request: Request):
    return templates.TemplateResponse(
        "test_client.html",
        {"request": request}
    )
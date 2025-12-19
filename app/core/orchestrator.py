import asyncio
import whisper
import pyttsx3
import io
from typing import Optional
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path


class VoiceOrchestrator:
    def __init__(self):
        # Path to your local Whisper model
        base_dir = Path(__file__).resolve().parent
        model_path = base_dir / "../models/stt_models/whisper"

        # Initialize STT (Hugging Face Whisper)
        self.stt_processor = WhisperProcessor.from_pretrained(model_path)
        self.stt_model = WhisperForConditionalGeneration.from_pretrained(model_path)

        # Initialize TTS (pyttsx3)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed percent

        # Simple conversation memory
        self.conversation_history = []
        
    async def speech_to_text(self, audio_bytes: bytes) -> str:
        """Convert audio to text using Whisper."""
        try:
            # Load audio
            import numpy as np
            import wave
            
            # Convert bytes to numpy array
            audio_stream = io.BytesIO(audio_bytes)
            
            # Use Whisper
            result = self.stt_model.transcribe(audio_stream.name)
            return result["text"]
            
        except Exception as e:
            print(f"STT Error: {e}")
            return ""
    
    async def get_llm_response(self, user_input: str) -> str:
        """Get response from LLM (simplified for now)."""
        # Add to conversation history
        self.conversation_history.append(f"User: {user_input}")
        
        # Simple echo bot for testing
        response = f"I heard you say: '{user_input}'. This is a test response."
        
        # Alternative: Use a rule-based response
        user_lower = user_input.lower()
        
        if "hello" in user_lower or "hi" in user_lower:
            response = "Hello! How can I help you today?"
        elif "how are you" in user_lower:
            response = "I'm doing well, thank you for asking! How about you?"
        elif "bye" in user_lower or "goodbye" in user_lower:
            response = "Goodbye! Have a great day!"
        elif "name" in user_lower:
            response = "I'm your voice assistant. You can call me VoiceBot."
        else:
            response = f"You said: '{user_input}'. I'm still learning to have full conversations. Try saying hello!"
        
        self.conversation_history.append(f"Assistant: {response}")
        
        # Keep history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
            
        return response
    
    async def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech using pyttsx3."""
        try:
            # Create in-memory audio file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                tmp_path = tmpfile.name
            
            # Save speech to file
            self.tts_engine.save_to_file(text, tmp_path)
            self.tts_engine.runAndWait()
            
            # Read the file
            with open(tmp_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Clean up
            os.unlink(tmp_path)
            
            return audio_bytes
            
        except Exception as e:
            print(f"TTS Error: {e}")
            # Return empty bytes on error
            return b""
    
    async def reset_conversation(self, client_id: str):
        """Reset conversation history."""
        self.conversation_history = []
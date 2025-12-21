import asyncio
import whisper
import pyttsx3
import numpy as np
from pathlib import Path


class VoiceOrchestrator:
    def __init__(self, model_size="base"):
        """
        Use locally stored Whisper models.
        
        Args:
            model_size: "tiny", "base", "small", "medium", "large"
        """
        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / "models" / "stt_models" / "whisper"
        
        print(f"Loading Whisper {model_size} model from {model_path}...")
        
        # Load model from local directory
        try:
            self.stt_model = whisper.load_model(
                model_size, 
                download_root=str(model_path)
            )
            print(f"✓ Whisper model loaded from local cache")
        except Exception as e:
            print(f"Error loading local model: {e}")
            print("Falling back to online download...")
            self.stt_model = whisper.load_model(model_size)
        
        # Initialize TTS
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            print("✓ TTS engine initialized")
        except Exception as e:
            print(f"TTS Error: {e}")
            self.tts_engine = None
        
        self.conversation_history = []

    async def speech_to_text(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Convert audio to text using Whisper."""
        try:
            # Method 1: Convert bytes directly to numpy array (Recommended)
            # Assuming audio is 16-bit PCM, mono
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe using Whisper
            result = self.stt_model.transcribe(
                audio_array,
                fp16=False,  # Use float32 for CPU
                language="en",
                task="transcribe"
            )
            
            text = result["text"].strip()
            print(f"STT Result: '{text}'")
            return text
            
        except Exception as e:
            print(f"STT Error: {e}")
            return ""

    # Alternative method if you need to handle WAV files
    async def speech_to_text_from_wav(self, audio_bytes: bytes) -> str:
        """Alternative method for WAV formatted audio."""
        try:
            # Save bytes to temporary file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                tmp_path = tmpfile.name
                tmpfile.write(audio_bytes)
            
            # Transcribe from file
            result = self.stt_model.transcribe(tmp_path)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            text = result["text"].strip()
            print(f"STT Result: '{text}'")
            return text
            
        except Exception as e:
            print(f"STT Error (WAV method): {e}")
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
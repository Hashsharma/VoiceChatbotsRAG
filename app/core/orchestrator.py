# Updated VoiceOrchestrator with fast voice engine
import asyncio
import whisper
import numpy as np
from pathlib import Path
import io
from pydub import AudioSegment
import torch
from scipy.io.wavfile import write
import traceback
from ..speech.utils.voice_engine import FastVoiceEngine  # Import our new engine

base_dir = Path(__file__).resolve().parent.parent

class VoiceOrchestrator:
    def __init__(self, model_size="base"):
        """
        Use locally stored Whisper models.
        
        Args:
            model_size: "tiny", "base", "small", "medium", "large"
        """
        # Initialize fast voice engine
        self.voice_engine = FastVoiceEngine(use_gpu=True, gpu_id=0)
        
        # Load Whisper
        model_path = base_dir / "models" / "stt_models" / "whisper"
        self.conversation_histories = {}  # client_id -> conversation_history
        
        print(f"Loading Whisper {model_size} model from {model_path}...")
        
        try:
            self.stt_model = whisper.load_model(
                model_size, 
                download_root=str(model_path)
            )
            print(f"✅ Whisper model loaded")
        except Exception as e:
            print(f"Error loading local model: {e}")
            print("Falling back to online download...")
            self.stt_model = whisper.load_model(model_size)
        
        # Keep your existing initialization
        self.conversation_history = []
    
    async def initialize_voice(self, voice_file="my_voice.wav"):
        """Initialize voice engine (call this once at startup)"""
        print("🚀 Initializing voice system...")
        await self.voice_engine.initialize(voice_file)
        
        # Show GPU info
        gpu_info = self.voice_engine.get_gpu_info()
        if gpu_info.get("gpu_available", False):
            print(f"🎮 GPU: {gpu_info['gpu_name']}")
            print(f"🧠 Memory: {gpu_info['memory_allocated']} / {gpu_info['memory_cached']}")
    
    async def speech_to_text(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        Convert audio bytes to text using Whisper.
        Supports raw PCM, WAV, WebM, or Opus audio.
        """
        try:
            if not audio_bytes:
                print("STT Error: Empty audio bytes")
                return ""

            print(f"🎤 STT: Received {len(audio_bytes)} bytes of audio")

            audio_array = None

            # Try raw PCM first
            try:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                if len(audio_array) > sample_rate * 0.5:
                    print(f"📊 PCM conversion successful, duration {len(audio_array)/sample_rate:.2f}s")
                else:
                    audio_array = None
            except:
                audio_array = None

            # Try pydub if PCM failed
            if audio_array is None:
                try:
                    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
                    audio_segment = audio_segment.set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)
                    pcm_bytes = audio_segment.raw_data
                    audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    print(f"🔄 Conversion via pydub successful")
                except Exception as e:
                    print(f"STT Error: Audio conversion failed: {e}")
                    return ""

            # Check audio volume
            if np.max(np.abs(audio_array)) < 0.01:
                print("🔇 Audio is too quiet/silent")
                return ""

            # Transcribe with Whisper
            print("✍️ Starting transcription...")
            result = self.stt_model.transcribe(
                audio_array,
                fp16=False,
                language="en",
                task="transcribe"
            )

            text = result.get("text", "").strip()
            print(f"📝 STT Result: '{text}'")
            return text

        except Exception as e:
            print(f"❌ STT Error: {e}")
            traceback.print_exc()
            return ""
    
    async def text_to_speech(self, text: str, sample_rate: int = 22050, 
                           language: str = "en", speed: float = 1.0) -> bytes:
        """
        Fast async text-to-speech with GPU acceleration.
        """
        try:
            # if not text or not text.strip():
            #     return b""

            print(f"🔊 TTS: '{text[:50]}...'")
            text = "I m better feeling"
            # Use the fast voice engine
            audio_bytes = await self.voice_engine.synthesize(
                text=text,
                language=language,
                speed=speed
            )
            
            print(f"✅ Generated {len(audio_bytes)} bytes")
            return audio_bytes
            
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            traceback.print_exc()
            return b""
    
    # Keep all your existing methods unchanged...
    async def get_llm_response(self, user_input: str, client_id: str) -> str:
        """Get LLM response with user-specific conversation history."""
        if client_id not in self.conversation_histories:
            self.conversation_histories[client_id] = []
        
        user_history = self.conversation_histories[client_id]
        user_history.append(f"User: {user_input}")
        
        response = self._generate_response(user_input, user_history)
        user_history.append(f"Assistant: {response}")
        
        if len(user_history) > 20:
            self.conversation_histories[client_id] = user_history[-20:]
        
        return response
    
    def _generate_response(self, user_input: str, history: list) -> str:
        """Generate response using rules or actual LLM."""
        user_lower = user_input.lower()
        
        context_aware = self._check_conversation_context(history, user_input)
        if context_aware:
            return context_aware
        
        if "hello" in user_lower or "hi" in user_lower:
            return "Hello! How can I help you today?"
        elif "how are you" in user_lower:
            return "I'm doing well, thank you for asking! How about you?"
        elif "bye" in user_lower or "goodbye" in user_lower:
            return "Goodbye! Have a great day!"
        elif "name" in user_lower:
            return "I'm your voice assistant. You can call me VoiceBot."
        elif "thank" in user_lower:
            return "You're welcome!"
        else:
            if len(history) > 2:
                return f"You mentioned: '{user_input}'. Could you tell me more about that?"
            return f"I understand you said: '{user_input}'. How can I assist you further?"
    
    def _check_conversation_context(self, history: list, current_input: str) -> str:
        """Check if current input relates to previous conversation."""
        if len(history) < 2:
            return ""
        
        last_assistant_msg = history[-1] if "Assistant:" in history[-1] else ""
        
        if "name" in last_assistant_msg.lower() and "my name is" in current_input.lower():
            name = current_input.lower().replace("my name is", "").strip()
            return f"Nice to meet you, {name}! How can I help you today?"
        
        if "how are you" in last_assistant_msg.lower():
            if any(word in current_input.lower() for word in ["good", "fine", "well"]):
                return "That's great to hear! What would you like to talk about?"
            elif any(word in current_input.lower() for word in ["bad", "tired", "sad"]):
                return "I'm sorry to hear that. Is there anything I can do to help?"
        
        return ""
    
    async def reset_conversation(self, client_id: str):
        """Reset conversation history for specific user."""
        if client_id in self.conversation_histories:
            self.conversation_histories[client_id] = []
            return True
        return False
    
    async def get_conversation_summary(self, client_id: str) -> str:
        """Get summary of user's conversation history."""
        if client_id in self.conversation_histories:
            history = self.conversation_histories[client_id]
            return "\n".join(history[-6:]) if history else "No conversation history"
        return "User not found"
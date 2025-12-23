# minimal_voice_engine.py
import asyncio
import numpy as np
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
import traceback

class MinimalVoiceEngine:
    """Minimal voice engine that works without complex TTS dependencies"""
    
    def __init__(self, use_gpu=False):  # GPU not used in minimal version
        self.initialized = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock = asyncio.Lock()
        
    async def initialize(self, voice_file="my_voice.wav"):
        """Minimal initialization"""
        if self.initialized:
            return
        
        async with self.lock:
            if self.initialized:
                return
            
            print("🔄 Initializing minimal voice engine...")
            self.initialized = True
            print("✅ Minimal voice engine ready (beep-only mode)")
    
    async def synthesize(self, text: str, language: str = "en") -> bytes:
        """Generate simple beep audio based on text"""
        if not self.initialized:
            await self.initialize()
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._generate_beep,
            text
        )
    
    def _generate_beep(self, text: str) -> bytes:
        """Generate a simple beep audio"""
        try:
            sample_rate = 24000
            duration = max(0.5, min(len(text) * 0.1, 3.0))
            
            # Different tones for different text
            if any(word in text.lower() for word in ["hello", "hi", "hey"]):
                freq = 440  # Greeting tone
            elif "?" in text:
                freq = 550  # Question tone
            elif any(word in text.lower() for word in ["thank", "thanks"]):
                freq = 330  # Appreciation tone
            else:
                freq = 220  # Normal tone
            
            # Generate sine wave
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
            
            # Add fade in/out
            fade_samples = int(0.1 * sample_rate)
            if fade_samples * 2 < len(audio):
                # Fade in
                fade_in = np.linspace(0, 1, fade_samples)
                audio[:fade_samples] *= fade_in
                # Fade out
                fade_out = np.linspace(1, 0, fade_samples)
                audio[-fade_samples:] *= fade_out
            
            # Convert to WAV bytes
            return self._audio_to_wav_bytes(audio, sample_rate)
            
        except Exception as e:
            print(f"❌ Beep generation failed: {e}")
            return self._create_silence()
    
    def _audio_to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio array to WAV bytes"""
        import wave
        import struct
        
        # Ensure audio is in correct range
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        buffer = io.BytesIO()
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            
            # Convert to bytes
            frames = b''.join(struct.pack('<h', sample) for sample in audio_int16)
            wav_file.writeframes(frames)
        
        return buffer.getvalue()
    
    def _create_silence(self, duration=0.5) -> bytes:
        """Create silent audio"""
        sample_rate = 24000
        samples = int(sample_rate * duration)
        silent = np.zeros(samples)
        return self._audio_to_wav_bytes(silent, sample_rate)
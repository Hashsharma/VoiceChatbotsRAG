import numpy as np
import asyncio

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
    
    async def process_chunk(self, audio_chunk: bytes) -> bytes:
        """Simple audio processing (can be enhanced later)."""
        # For now, just return as-is
        return audio_chunk
    
    async def detect_voice(self, audio_chunk: bytes) -> bool:
        """Simple voice activity detection based on energy."""
        if len(audio_chunk) == 0:
            return False
        
        # Convert to numpy array
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        
        if len(audio_data) == 0:
            return False
        
        # Calculate energy
        energy = np.sum(audio_data.astype(np.float32) ** 2) / len(audio_data)
        
        # Simple threshold (adjust based on your audio levels)
        threshold = 1000
        
        return energy > threshold
    
    async def bytes_to_wav(self, audio_bytes: bytes) -> bytes:
        """Convert raw bytes to WAV format."""
        import wave
        import io
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_bytes)
        
        return wav_buffer.getvalue()
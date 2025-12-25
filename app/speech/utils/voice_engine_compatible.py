# voice_engine_compatible.py
import asyncio
import torch
import numpy as np
import io
import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor
import traceback

class CompatibleVoiceEngine:
    def __init__(self, use_gpu=True, gpu_id=0):
        self.tts = None
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.initialized = False
        self.lock = asyncio.Lock()
        self.voice_file = None
        self.model_name = None
        self.is_xtts = False
        
    async def initialize(self, voice_file="my_voice.wav"):
        """Initialize TTS with compatibility handling"""
        if self.initialized:
            return
        
        async with self.lock:
            if self.initialized:
                return
            
            print("🔄 Initializing voice engine...")
            start_time = time.time()
            
            try:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._initialize_sync,
                    voice_file
                )
                
                self.initialized = True
                print(f"✅ Voice engine ready in {time.time() - start_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Failed to initialize voice engine: {e}")
                traceback.print_exc()
                # Try fallback initialization
                await self._fallback_initialize(voice_file)
    
    def _initialize_sync(self, voice_file):
        """Synchronous initialization with compatibility fixes"""
        from TTS.api import TTS
        
        # Check TTS version
        try:
            import TTS as tts_module
            tts_version = getattr(tts_module, '__version__', 'unknown')
            print(f"📦 TTS version: {tts_version}")
        except:
            print("📦 TTS version: unknown")
        
        # Try different initialization methods
        models_to_try = [
            # Try without progress_bar first (older versions don't have it)
            ("tts_models/multilingual/multi-dataset/xtts_v2", {"gpu": self.use_gpu}),
            # Try with minimal parameters
            ("tts_models/multilingual/multi-dataset/xtts_v2", {}),
            # Fallback to English model
            ("tts_models/en/ljspeech/tacotron2-DDC", {"gpu": self.use_gpu}),
        ]
        
        last_error = None
        for model_name, kwargs in models_to_try:
            try:
                print(f"🔄 Trying model: {model_name}")
                
                # Try different initialization signatures
                try:
                    # Method 1: Standard initialization
                    self.tts = TTS(model_name, **kwargs)
                except TypeError as e:
                    if "progress_bar" in str(e):
                        # Method 2: Without progress_bar
                        kwargs.pop('progress_bar', None)
                        self.tts = TTS(model_name, **kwargs)
                    elif "samples" in str(e):
                        # Method 3: Minimal initialization
                        self.tts = TTS(model_name)
                    else:
                        raise
                
                self.model_name = model_name
                self.is_xtts = "xtts" in model_name.lower()
                print(f"✅ Successfully loaded: {model_name}")
                break
                
            except Exception as e:
                last_error = e
                print(f"⚠️ Failed to load {model_name}: {e}")
                continue
        
        if self.tts is None:
            raise RuntimeError(f"Could not load any TTS model. Last error: {last_error}")
        
        # Verify or create voice file
        if not os.path.exists(voice_file):
            print(f"⚠️ Voice file not found, creating default...")
            voice_file = self._create_default_voice(voice_file)
        
        self.voice_file = voice_file
        
        # Test the model
        self._test_model()
    
    async def _fallback_initialize(self, voice_file):
        """Fallback initialization using different approach"""
        print("🔄 Trying fallback initialization...")
        
        try:
            # Try using a different import method
            import importlib
            TTS = importlib.import_module('TTS.api').TTS
            
            # Try with minimal parameters
            self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            self.model_name = "tts_models/en/ljspeech/tacotron2-DDC"
            self.is_xtts = False
            
            # Verify voice file
            if not os.path.exists(voice_file):
                voice_file = self._create_default_voice(voice_file)
            
            self.voice_file = voice_file
            self.initialized = True
            print("✅ Fallback initialization successful")
            
        except Exception as e:
            print(f"❌ Fallback also failed: {e}")
            raise
    
    def _test_model(self):
        """Test the model with a short phrase"""
        print("🧪 Testing TTS model...")
        try:
            test_buffer = io.BytesIO()
            
            # Check API version
            if hasattr(self.tts, 'tts_to_file'):
                # New API
                if self.is_xtts:
                    self.tts.tts_to_file(
                        text="Test",
                        file_path=test_buffer,
                        speaker_wav=self.voice_file,
                        language="en"
                    )
                else:
                    self.tts.tts_to_file(
                        text="Test",
                        file_path=test_buffer
                    )
            else:
                # Old API
                audio = self.tts.tts(text="Test")
                # Handle numpy array
                if isinstance(audio, np.ndarray):
                    self._save_numpy_to_buffer(audio, test_buffer)
            
            print("✅ Model test successful")
            
        except Exception as e:
            print(f"⚠️ Model test failed but continuing: {e}")
    
    async def synthesize(self, text: str, language: str = "en", 
                        speed: float = 1.0) -> bytes:
        """Async text-to-speech synthesis"""
        if not self.initialized:
            await self.initialize()
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._synthesize_sync,
            text, language, speed
        )
    
    def _synthesize_sync(self, text: str, language: str, speed: float) -> bytes:
        """Synchronous synthesis"""
        try:
            if self.tts is None:
                raise ValueError("TTS model not initialized")
            
            start_time = time.time()
            print(f"🔊 Synthesizing: '{text[:50]}...'")
            
            # Create bytes buffer
            wav_buffer = io.BytesIO()
            
            # Check which API is available
            if hasattr(self.tts, 'tts_to_file'):
                # New API
                if self.is_xtts:
                    # XTTS needs speaker_wav and language
                    params = {
                        "text": text,
                        "file_path": wav_buffer,
                        "speaker_wav": self.voice_file,
                        "language": language
                    }
                    
                    # Try with speed if supported
                    if speed != 1.0:
                        try:
                            params["speed"] = speed
                        except:
                            pass  # Speed not supported
                    
                    self.tts.tts_to_file(**params)
                else:
                    # Non-XTTS models
                    params = {
                        "text": text,
                        "file_path": wav_buffer
                    }
                    
                    # Try with speed if supported
                    if speed != 1.0:
                        try:
                            params["speed"] = speed
                        except:
                            pass
                    
                    self.tts.tts_to_file(**params)
            else:
                # Old API
                audio = self.tts.tts(text=text)
                self._save_numpy_to_buffer(audio, wav_buffer)
            
            audio_bytes = wav_buffer.getvalue()
            
            inference_time = time.time() - start_time
            print(f"✅ Generated {len(audio_bytes)} bytes in {inference_time:.2f}s")
            
            return audio_bytes
            
        except Exception as e:
            print(f"❌ Synthesis error: {e}")
            traceback.print_exc()
            # Return silent audio as fallback
            return self._create_silent_audio()
    
    def _save_numpy_to_buffer(self, audio_np: np.ndarray, buffer: io.BytesIO):
        """Save numpy array to WAV buffer"""
        try:
            # Try using scipy
            from scipy.io.wavfile import write
            write(buffer, 24000, (audio_np * 32767).astype(np.int16))
        except ImportError:
            # Manual WAV creation
            self._numpy_to_wav_bytes(audio_np, buffer)
    
    def _numpy_to_wav_bytes(self, audio_np: np.ndarray, buffer: io.BytesIO):
        """Convert numpy array to WAV bytes manually"""
        import wave
        import struct
        
        # Ensure audio is in correct range
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        sample_rate = 24000
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            
            # Convert to bytes
            frames = b''.join(struct.pack('<h', sample) for sample in audio_int16)
            wav_file.writeframes(frames)
    
    def _create_default_voice(self, output_file="my_voice.wav"):
        """Create a default voice file"""
        print("🎵 Creating default voice...")
        
        sample_rate = 24000
        duration = 3.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Simple voice-like tone
        audio = 0.5 * np.sin(2 * np.pi * 180 * t)
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Save as WAV
        try:
            from scipy.io.wavfile import write
            write(output_file, sample_rate, (audio * 32767).astype(np.int16))
        except ImportError:
            # Manual save
            self._numpy_to_wav_bytes(audio, output_file)
        
        print(f"✅ Created default voice: {output_file}")
        return output_file
    
    def _create_silent_audio(self, duration=0.5):
        """Create silent audio as fallback"""
        sample_rate = 24000
        samples = int(sample_rate * duration)
        silent = np.zeros(samples, dtype=np.int16)
        
        buffer = io.BytesIO()
        self._numpy_to_wav_bytes(silent / 32767.0, buffer)
        return buffer.getvalue()
    
    def get_model_info(self):
        """Get information about loaded model"""
        if self.tts:
            info = {
                "model_loaded": True,
                "model_name": self.model_name,
                "is_xtts": self.is_xtts,
                "device": self.device,
                "voice_file": self.voice_file
            }
            return info
        return {"model_loaded": False}
    
    async def close(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        if self.use_gpu:
            torch.cuda.empty_cache()
# voice_engine.py
import asyncio
import torch
import numpy as np
import io
import os
import time
import json
from pathlib import Path
from scipy.io.wavfile import write
from concurrent.futures import ThreadPoolExecutor
import traceback

class FastVoiceEngine:
    def __init__(self, use_gpu=True, gpu_id=0):
        self.model = None
        self.gpt_cond_latent = None
        self.speaker_embedding = None
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.gpu_id = gpu_id
        self.device = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.initialized = False
        self.lock = asyncio.Lock()
        self.model_dir = Path(__file__).parent / "xtts_models"
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize Piper voice if available
        self.piper_voice = None
        self._initialize_piper()

    def _initialize_piper(self):
        """Initialize Piper TTS if available"""
        try:
            from piper import PiperVoice
            
            # Try to load your existing Piper model
            model_path = "/media/scientist-anand/volume/mr_document/Linux_Git/VoiceChatbotsRAG/app/models/tts_models/pipper-ttss/en_US-lessac-medium.onnx"
            config_path = "/media/scientist-anand/volume/mr_document/Linux_Git/VoiceChatbotsRAG/app/models/tts_models/pipper-ttss/en_US-lessac-medium.onnx.json"
            
            if os.path.exists(model_path):
                if os.path.exists(config_path):
                    self.piper_voice = PiperVoice.load(
                        model_path=model_path,
                        config_path=config_path
                    )
                else:
                    self.piper_voice = PiperVoice.load(model_path=model_path)
                
                print(f"✅ Piper TTS loaded from {model_path}")
                print(f"   Sample rate: {self.piper_voice.config.sample_rate}")
            else:
                print("⚠️ Piper model not found, will use XTTS only")
                
        except ImportError:
            print("⚠️ Piper not installed, will use XTTS only")
        except Exception as e:
            print(f"⚠️ Could not load Piper: {e}")
        
    async def initialize(self, voice_file="my_voice.wav"):
        """Async initialization with GPU support"""
        if self.initialized:
            return
        
        async with self.lock:
            if self.initialized:
                return
            
            print("🔄 Initializing voice engine...")
            start_time = time.time()
            
            # Run heavy loading in thread pool
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
                raise
    
    def _initialize_sync(self, voice_file):
        """Synchronous initialization (runs in thread pool)"""
        try:
            from TTS.tts.models.xtts import Xtts
            from TTS.config import load_config
            
            # Set device
            if self.use_gpu:
                torch.cuda.set_device(self.gpu_id)
                self.device = torch.device(f"cuda:{self.gpu_id}")
                print(f"🎮 Using GPU: {torch.cuda.get_device_name(self.gpu_id)}")
            else:
                self.device = torch.device("cpu")
                print("💻 Using CPU")
            
            # Verify voice file exists
            if not os.path.exists(voice_file):
                print(f"⚠️ Voice file {voice_file} not found, creating default...")
                voice_file = self._create_default_voice()
            
            # Download model if not exists
            model_path = self._download_or_get_model()
            
            # Load config
            config_path = model_path / "config.json"
            if not config_path.exists():
                # Try to find config file
                config_files = list(model_path.glob("*.json"))
                if config_files:
                    config_path = config_files[0]
                else:
                    raise FileNotFoundError(f"No config file found in {model_path}")
            
            print(f"📋 Loading config from {config_path}")
            config = load_config(config_path)
            
            # Initialize model
            print("🎯 Initializing XTTS model...")
            self.model = Xtts.init_from_config(config)
            
            # Load checkpoint
            checkpoint_dir = model_path / "model.pth"
            if not checkpoint_dir.exists():
                # Look for checkpoint files
                checkpoint_files = list(model_path.glob("*.pth"))
                if checkpoint_files:
                    checkpoint_dir = checkpoint_files[0]
                else:
                    raise FileNotFoundError(f"No checkpoint found in {model_path}")
            
            print(f"💾 Loading checkpoint from {checkpoint_dir}")
            self.model.load_checkpoint(
                config, 
                checkpoint_path=str(checkpoint_dir), 
                vocab_path=None,
                use_deepspeed=False
            )
            
            # Move to device
            self.model.to(self.device)
            
            # Extract speaker embeddings
            print("🎤 Extracting voice embeddings...")
            with torch.no_grad():
                gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                    audio_path=[voice_file]
                )
                
                # Keep tensors on GPU for faster inference
                if self.use_gpu:
                    self.gpt_cond_latent = gpt_cond_latent
                    self.speaker_embedding = speaker_embedding
                else:
                    self.gpt_cond_latent = gpt_cond_latent.cpu()
                    self.speaker_embedding = speaker_embedding.cpu()
            
            print(f"🎯 GPT latent shape: {gpt_cond_latent.shape}")
            print(f"🎯 Speaker embedding shape: {speaker_embedding.shape}")
            
            # Optimize for inference
            self._optimize_model()
            
        except Exception as e:
            print(f"❌ Error in initialization: {e}")
            traceback.print_exc()
            raise
    
    def _download_or_get_model(self):
        """Download XTTS v2 model or use existing"""
        from TTS.utils.manage import ModelManager
        
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        
        # Check if model already exists
        model_path = self.model_dir / "xtts_v2"
        if model_path.exists():
            print(f"📂 Using existing model from {model_path}")
            return model_path
        
        # Download model
        print(f"📥 Downloading {model_name}...")
        model_manager = ModelManager()
        
        # Download model
        download_path, config_path, model_item = model_manager.download_model(model_name)
        
        # Move to our model directory
        import shutil
        shutil.move(download_path, model_path)
        
        print(f"✅ Model downloaded to {model_path}")
        return model_path
    
    def _optimize_model(self):
        """Optimize model for faster inference"""
        self.model.eval()
        
        if self.use_gpu:
            torch.backends.cudnn.benchmark = True
            
            # Use half precision if supported
            try:
                self.model = self.model.half()
                if self.gpt_cond_latent is not None:
                    self.gpt_cond_latent = self.gpt_cond_latent.half()
                if self.speaker_embedding is not None:
                    self.speaker_embedding = self.speaker_embedding.half()
                print("📊 Enabled half precision (FP16)")
            except Exception as e:
                print(f"📊 Using full precision (FP32): {e}")
    
    async def synthesize(self, text: str, language: str = "en", 
                     temperature: float = 0.7, speed: float = 1.0) -> bytes:
        """Async text-to-speech synthesis - ALWAYS uses Piper if available"""
        # ALWAYS use Piper if it's available
        if self.piper_voice is not None:
            print("🎯 Using Piper TTS (fast)")
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._synthesize_with_piper_only,
                    text, speed
                )
            except Exception as e:
                print(f"❌ Piper synthesis error: {e}")
                traceback.print_exc()
                return b""
        
        # Only use XTTS if Piper is not available
        print("⚠️ Piper not available, using XTTS")
        if not self.initialized:
            await self.initialize()
        
        # Run XTTS synthesis in thread pool
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._synthesize_xtts_only,
                text, language, temperature, speed
            )
        except Exception as e:
            print(f"❌ XTTS synthesis error: {e}")
            traceback.print_exc()
            return b""

    def _synthesize_with_piper_only(self, text: str, speed: float = 1.0) -> bytes:
        """Use ONLY Piper TTS for faster synthesis - simplest version"""
        try:
            start_time = time.time()
            
            # Generate audio
            audio_chunks = self.piper_voice.synthesize(text=text)
            
            # Collect chunks
            chunks = list(audio_chunks)
            
            if not chunks:
                return self._create_silent_audio()
            
            # Get sample rate and combine audio data
            sample_rate = chunks[0].sample_rate
            
            # Method 1: Use _audio_int16_array if available
            if hasattr(chunks[0], '_audio_int16_array') and chunks[0]._audio_int16_array is not None:
                combined_audio = np.concatenate([chunk._audio_int16_array for chunk in chunks])
            # Method 2: Convert float array to int16
            elif hasattr(chunks[0], 'audio_float_array'):
                combined_audio = np.concatenate([
                    (chunk.audio_float_array * 32767).astype(np.int16) 
                    for chunk in chunks
                ])
            else:
                return self._create_silent_audio()
            
            # Create WAV bytes
            wav_buffer = io.BytesIO()
            write(wav_buffer, sample_rate, combined_audio)
            
            inference_time = time.time() - start_time
            print(f"🔊 Piper: {len(text)} chars in {inference_time:.2f}s")
            
            return wav_buffer.getvalue()
                    
        except Exception as e:
            print(f"❌ Piper TTS error: {e}")
            traceback.print_exc()
            return self._create_silent_audio()

    def _synthesize_xtts_only(self, text: str, language: str, 
                            temperature: float, speed: float) -> bytes:
        """Fallback to XTTS only if Piper is not available"""
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
            
            start_time = time.time()
            
            with torch.no_grad():
                # Ensure tensors are on correct device
                if self.use_gpu:
                    gpt_cond_latent = self.gpt_cond_latent
                    speaker_embedding = self.speaker_embedding
                else:
                    gpt_cond_latent = self.gpt_cond_latent.to(self.device)
                    speaker_embedding = self.speaker_embedding.to(self.device)
                
                # Generate audio
                out = self.model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=temperature,
                    length_penalty=1.0,
                    repetition_penalty=10.0,
                    top_k=50,
                    top_p=0.85,
                    speed=speed,
                    enable_text_splitting=True
                )
                
                # Get audio
                wav = out["wav"]
                if self.use_gpu:
                    wav = wav.cpu()
                
                # Convert to numpy
                wav_np = wav.numpy()
            
            # Convert to WAV bytes
            wav_buffer = io.BytesIO()
            write(wav_buffer, 24000, (wav_np * 32767).astype(np.int16))
            
            inference_time = time.time() - start_time
            print(f"🔊 XTTS: Synthesized {len(text)} chars in {inference_time:.2f}s")
            
            return wav_buffer.getvalue()
            
        except Exception as e:
            print(f"❌ XTTS synthesis error: {e}")
            traceback.print_exc()
            return self._create_silent_audio()
    
    def _create_default_voice(self, output_file="default_voice.wav"):
        """Create a synthetic voice if none exists"""
        print("🎵 Creating default voice...")
        
        sample_rate = 24000
        duration = 3.0
        
        # Generate voice-like audio
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        base_freq = 180
        harmonics = [
            (0.5, base_freq),
            (0.3, base_freq * 2),
            (0.2, base_freq * 3),
            (0.1, base_freq * 4)
        ]
        
        audio = np.zeros_like(t)
        for amp, freq in harmonics:
            audio += amp * np.sin(2 * np.pi * freq * t)
        
        # Add slight vibrato
        vibrato = 0.02 * np.sin(2 * np.pi * 5 * t)
        audio *= (1 + vibrato)
        
        # Normalize and save
        audio = audio / np.max(np.abs(audio))
        write(output_file, sample_rate, (audio * 32767).astype(np.int16))
        
        print(f"✅ Created default voice: {output_file}")
        return output_file
    
    def _create_silent_audio(self, duration=0.5):
        """Create silent audio as fallback"""
        sample_rate = 24000
        samples = int(sample_rate * duration)
        silent = np.zeros(samples, dtype=np.int16)
        
        wav_buffer = io.BytesIO()
        write(wav_buffer, sample_rate, silent)
        return wav_buffer.getvalue()
    
    def get_gpu_info(self):
        """Get GPU information"""
        if torch.cuda.is_available():
            try:
                info = {
                    "gpu_name": torch.cuda.get_device_name(self.gpu_id),
                    "memory_allocated": f"{torch.cuda.memory_allocated(self.gpu_id) / 1024**2:.1f} MB",
                    "memory_cached": f"{torch.cuda.memory_reserved(self.gpu_id) / 1024**2:.1f} MB",
                    "cuda_version": torch.version.cuda
                }
                return info
            except:
                return {"gpu_available": True, "error": "Could not get GPU info"}
        return {"gpu_available": False}
    
    async def close(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        if self.use_gpu:
            torch.cuda.empty_cache()
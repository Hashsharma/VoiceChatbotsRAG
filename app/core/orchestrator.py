import asyncio
import whisper
import pyttsx3
import numpy as np
from pathlib import Path
import io
from pydub import AudioSegment


class VoiceOrchestrator:
    def __init__(self, model_size="base"):
        """
        Use locally stored Whisper models.
        
        Args:
            model_size: "tiny", "base", "small", "medium", "large"
        """
        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / "models" / "stt_models" / "whisper"
        self.conversation_histories = {}  # client_id -> conversation_history
        
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
            if len(audio_bytes) == 0:
                print("STT Error: Empty audio bytes")
                return ""
            
            print(f"STT: Received {len(audio_bytes)} bytes of audio")
            
            # Method 1: Try direct conversion if it's already PCM
            try:
                # If audio is already 16-bit PCM
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Check if audio has reasonable length (at least 0.5 seconds)
                if len(audio_array) < sample_rate * 0.5:
                    print(f"STT Error: Audio too short: {len(audio_array)/sample_rate:.2f} seconds")
                    return ""
                
                print(f"STT: Direct PCM conversion, duration: {len(audio_array)/sample_rate:.2f}s")
                
            except Exception as e:
                print(f"STT: Not raw PCM, trying WebM/Opus conversion: {e}")
                # Method 2: Convert WebM/Opus to PCM
                try:
                    # Create AudioSegment from bytes
                    audio_segment = AudioSegment.from_file(
                        io.BytesIO(audio_bytes),
                        format="webm"  # or "opus" depending on format
                    )
                    
                    # Convert to mono, 16kHz, 16-bit PCM
                    audio_segment = audio_segment.set_channels(1)
                    audio_segment = audio_segment.set_frame_rate(sample_rate)
                    audio_segment = audio_segment.set_sample_width(2)  # 16-bit
                    
                    # Export to raw PCM
                    pcm_bytes = audio_segment.raw_data
                    
                    # Convert to numpy array
                    audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    print(f"STT: WebM conversion successful, duration: {len(audio_array)/sample_rate:.2f}s")
                    
                except Exception as conv_error:
                    print(f"STT: WebM conversion failed, trying WAV: {conv_error}")
                    # Method 3: Try as WAV
                    try:
                        audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
                        audio_segment = audio_segment.set_channels(1).set_frame_rate(sample_rate)
                        pcm_bytes = audio_segment.raw_data
                        audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        print(f"STT: WAV conversion successful")
                    except:
                        print("STT: All conversion methods failed")
                        return ""
            
            # Check audio array
            print(f"STT: Audio array shape: {audio_array.shape}, max: {np.max(np.abs(audio_array)):.4f}")
            
            # Ensure audio isn't silent
            if np.max(np.abs(audio_array)) < 0.01:  # Very quiet
                print("STT Error: Audio is too quiet/silent")
                return ""
            
            # Transcribe using Whisper
            print("STT: Starting transcription...")
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
            print(f"STT Error: {str(e)}")
            import traceback
            traceback.print_exc()
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
    
    async def get_llm_response(self, user_input: str, client_id: str) -> str:
        """Get LLM response with user-specific conversation history."""
        
        # Initialize history for new users
        if client_id not in self.conversation_histories:
            self.conversation_histories[client_id] = []
        
        # Get user's conversation history
        user_history = self.conversation_histories[client_id]
        
        # Add user input to history
        user_history.append(f"User: {user_input}")
        
        # Generate response based on conversation history
        response = self._generate_response(user_input, user_history)
        
        # Add assistant response to history
        user_history.append(f"Assistant: {response}")
        
        # Keep history manageable (last 10 exchanges = 20 messages)
        if len(user_history) > 20:
            self.conversation_histories[client_id] = user_history[-20:]
        
        return response
    
    def _generate_response(self, user_input: str, history: list) -> str:
        """Generate response using rules or actual LLM."""
        user_lower = user_input.lower()
        
        # Check if this is a follow-up to previous conversation
        context_aware = self._check_conversation_context(history, user_input)
        
        if context_aware:
            return context_aware
        
        # Your existing rule-based responses
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
            # For unknown queries, check if we should ask for clarification
            if len(history) > 2:
                return f"You mentioned: '{user_input}'. Could you tell me more about that?"
            return f"I understand you said: '{user_input}'. How can I assist you further?"
    
    def _check_conversation_context(self, history: list, current_input: str) -> str:
        """Check if current input relates to previous conversation."""
        if len(history) < 2:  # Need at least one previous exchange
            return ""
        
        # Get last assistant response
        last_assistant_msg = history[-1] if "Assistant:" in history[-1] else ""
        
        # Simple context checking examples
        if "name" in last_assistant_msg.lower() and "my name is" in current_input.lower():
            name = current_input.lower().replace("my name is", "").strip()
            return f"Nice to meet you, {name}! How can I help you today?"
        
        if "how are you" in last_assistant_msg.lower() and any(word in current_input.lower() for word in ["good", "fine", "well", "bad", "tired"]):
            if any(word in current_input.lower() for word in ["good", "fine", "well"]):
                return "That's great to hear! What would you like to talk about?"
            else:
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
            # Return last few messages or create summary
            return "\n".join(history[-6:]) if history else "No conversation history"
        return "User not found"
    
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

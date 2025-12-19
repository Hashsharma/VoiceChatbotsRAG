from fastapi import WebSocket, WebSocketDisconnect, APIRouter
import json
import asyncio
from app.schemas.websocket import AudioMessage, TextMessage, ControlMessage
from app.core.orchestrator import VoiceOrchestrator
from app.utils.audio_utils import AudioProcessor
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

'''
    Think of it as the centralized control room or switchboard operator for a multi-user voice application. It's responsible for keeping track of all active users/sessions and managing their voice processing pipelines.

    The Core Problem It Solves
    When you build a voice application that serves multiple users simultaneously (like a voice chat app, customer service bot, or multiplayer game with voice AI), you need to:
    Keep track of who's connected
    Maintain their individual state/conversation history
    Ensure their audio streams don't get mixed up

    Clean up resources when they disconnect
    ConnectionManager = The call center's main switchboard and supervisor
    active_connections = The board showing which operators are talking to which customers
    VoiceOrchestrator = The trained operator who knows how to handle calls
    AudioProcessor = The noise-cancelling headphones all operators use

    "user_123": {
        "websocket": ws_object,
        "conversation_history": [...],
        "voice_settings": {...},
        "last_active": timestamp
    },

'''

class ConnectionManager:
    def __init__(self):
        self.active_connections = {}
        self.orchestrator = VoiceOrchestrator()
        self.audio_processor = AudioProcessor()
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_audio(self, client_id: str, audio_data: bytes):
        if client_id in self.active_connections:
            message = AudioMessage(
                type="audio",
                data=audio_data.hex(),
                format="pcm_16000"
            )
            await self.active_connections[client_id].send_json(message.dict())
    
    async def send_text(self, client_id: str, text: str):
        if client_id in self.active_connections:
            message = TextMessage(
                type="text",
                content=text,
                timestamp=asyncio.get_event_loop().time()
            )
            await self.active_connections[client_id].send_json(message.dict())

manager = ConnectionManager()

@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    client_id = f"client_{id(websocket)}"
    
    # Buffer for accumulating audio chunks
    audio_buffer = bytearray()
    buffer_size = 0
    max_buffer_size = 5 * 16000 * 2  # 5 seconds of 16kHz, 16-bit audio
    
    # Conversation state
    is_speaking = False
    silence_counter = 0
    silence_threshold = 10  # Number of silent chunks before processing
    
    await manager.connect(websocket, client_id)
    
    try:
        # Send welcome message
        await manager.send_text(client_id, "Connected to voice chat. Start speaking...")
        
        while True:
            # Receive message from client
            try:
                data = await websocket.receive()
            except Exception as e:
                logger.error(f"Error receiving data: {e}")
                break
                
            # Handle different message types
            if "text" in data:
                # Text message (for testing or fallback)
                text_data = json.loads(data["text"])
                await handle_text_message(text_data, client_id)
                
            elif "bytes" in data:
                # Audio bytes (main voice communication)
                audio_chunk = data["bytes"]
                
                # Process audio chunk
                processed_chunk = await manager.audio_processor.process_chunk(audio_chunk)
                
                # Add VAD (Voice Activity Detection) - simple energy-based
                has_voice = await manager.audio_processor.detect_voice(processed_chunk)
                
                if has_voice:
                    audio_buffer.extend(processed_chunk)
                    silence_counter = 0
                    is_speaking = True
                    
                    # Send back listening indicator
                    await manager.send_text(client_id, "Listening...")
                    
                else:
                    silence_counter += 1
                    
                    # If we were speaking and now have silence, process the audio
                    if is_speaking and silence_counter >= silence_threshold:
                        if len(audio_buffer) > 0:
                            # Process the accumulated audio
                            await process_audio_buffer(audio_buffer, client_id)
                            
                            # Clear buffer
                            audio_buffer.clear()
                            buffer_size = 0
                            is_speaking = False
                        
                        silence_counter = 0
                
                # Prevent buffer from growing too large
                if len(audio_buffer) > max_buffer_size:
                    # Force process if buffer is too full
                    await process_audio_buffer(audio_buffer, client_id)
                    audio_buffer.clear()
                    buffer_size = 0
                    is_speaking = False
            
            elif data.get("type") == "websocket.disconnect":
                break
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
    finally:
        manager.disconnect(client_id)

async def process_audio_buffer(audio_buffer: bytearray, client_id: str):
    """Process accumulated audio buffer through STT → LLM → TTS pipeline."""
    if len(audio_buffer) == 0:
        return
    
    try:
        logger.info(f"Processing audio buffer of size {len(audio_buffer)} bytes")
        
        # Step 1: Convert audio to text (STT)
        await manager.send_text(client_id, "Processing speech...")
        text = await manager.orchestrator.speech_to_text(bytes(audio_buffer))
        logger.info(f"STT Result: {text}")
        
        if not text or len(text.strip()) < 2:
            await manager.send_text(client_id, "I didn't catch that. Could you repeat?")
            return
        
        await manager.send_text(client_id, f"You said: {text}")
        
        # Step 2: Get response from LLM
        await manager.send_text(client_id, "Thinking...")
        response_text = await manager.orchestrator.get_llm_response(text)
        
        # Step 3: Convert response to speech (TTS)
        await manager.send_text(client_id, "Generating audio response...")
        audio_response = await manager.orchestrator.text_to_speech(response_text)
        
        # Step 4: Send audio response back
        await manager.send_text(client_id, f"Assistant: {response_text}")
        await manager.send_audio(client_id, audio_response)
        
        logger.info("Audio processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        await manager.send_text(client_id, "Sorry, I encountered an error processing your request.")
        
        # Fallback: Send text response if TTS fails
        try:
            response_text = "I apologize, but I'm having trouble with voice synthesis. Here's my text response."
            await manager.send_text(client_id, response_text)
        except:
            pass

async def handle_text_message(data: dict, client_id: str):
    """Handle text messages (for testing or fallback mode)."""
    try:
        message_type = data.get("type", "text")
        
        if message_type == "control":
            # Handle control messages (start/stop recording, etc.)
            control_msg = ControlMessage(**data)
            
            if control_msg.command == "start_recording":
                await manager.send_text(client_id, "Recording started...")
            elif control_msg.command == "stop_recording":
                await manager.send_text(client_id, "Recording stopped.")
            elif control_msg.command == "reset":
                await manager.send_text(client_id, "Conversation reset.")
                await manager.orchestrator.reset_conversation(client_id)
                
        elif message_type == "text":
            # Direct text input (bypass STT)
            text_msg = TextMessage(**data)
            response_text = await manager.orchestrator.get_llm_response(text_msg.content)
            await manager.send_text(client_id, f"Assistant: {response_text}")
            
            # Optional: Also send TTS response
            try:
                audio_response = await manager.orchestrator.text_to_speech(response_text)
                await manager.send_audio(client_id, audio_response)
            except Exception as e:
                logger.warning(f"TTS failed for text message: {e}")
                
    except Exception as e:
        logger.error(f"Error handling text message: {str(e)}")
        await manager.send_text(client_id, f"Error: {str(e)}")
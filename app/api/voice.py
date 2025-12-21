from fastapi import WebSocket, WebSocketDisconnect, APIRouter
import json
import asyncio
from app.schemas.websocket import AudioMessage, TextMessage, ControlMessage
from app.core.orchestrator import VoiceOrchestrator
from app.utils.audio_utils import AudioProcessor
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections = {}
        self.orchestrator = VoiceOrchestrator()
        self.audio_processor = AudioProcessor()
        
    async def send_text(self, client_id: str, text: str):
        """Send text message to client"""
        websocket = self.active_connections.get(client_id)
        if websocket:
            try:
                # Send directly via websocket
                await websocket.send_text(text)
            except Exception as e:
                logger.error(f"Error sending text to {client_id}: {e}")
    
    async def send_audio(self, client_id: str, audio_data: bytes):
        """Send audio message to client"""
        websocket = self.active_connections.get(client_id)
        if websocket:
            try:
                audio_message = {
                    "type": "audio",
                    "data": audio_data.hex(),
                    "format": "wav"
                }
                await websocket.send_json(audio_message)
            except Exception as e:
                logger.error(f"Error sending audio to {client_id}: {e}")

manager = ConnectionManager()

@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    client_id = f"client_{id(websocket)}"
    
    # Accept the connection FIRST
    await websocket.accept()
    
    # Buffer for accumulating audio chunks
    audio_buffer = bytearray()
    max_buffer_size = 5 * 16000 * 2
    
    # Conversation state
    is_speaking = False
    silence_counter = 0
    silence_threshold = 10
    
    # Register with manager (AFTER accept)
    manager.active_connections[client_id] = websocket
    logger.info(f"Client {client_id} connected")
    
    try:
        # Send welcome message (AFTER connection is accepted)
        await manager.send_text(client_id, "Connected to voice chat. Start speaking...")
        
        while True:
            try:
                # Receive message
                message = await websocket.receive()
                
                # Handle connection message
                if message.get("type") == "websocket.connect":
                    # Already handled by websocket.accept()
                    continue
                
                # Handle disconnect
                if message.get("type") == "websocket.disconnect":
                    logger.info(f"Client {client_id} requested disconnect")
                    break
                
                # Handle text messages
                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                        await handle_text_message(data, client_id)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from {client_id}: {message['text']}")
                        # Use direct websocket.send_text() instead of manager method
                        await websocket.send_text("Invalid message format")
                
                # Handle binary audio data
                elif "bytes" in message:
                    audio_chunk = message["bytes"]
                    logger.debug(f"Received audio chunk of {len(audio_chunk)} bytes from {client_id}")
                    
                    # Process audio if processor is available
                    if hasattr(manager, 'audio_processor') and manager.audio_processor:
                        try:
                            processed_chunk = await manager.audio_processor.process_chunk(audio_chunk)
                            
                            # Voice activity detection
                            has_voice = await manager.audio_processor.detect_voice(processed_chunk)
                            
                            if has_voice:
                                audio_buffer.extend(processed_chunk)
                                silence_counter = 0
                                is_speaking = True
                                
                                # Send feedback
                                # await websocket.send_text("👂 Listening...")
                                
                            else:
                                silence_counter += 1
                                
                                # Process when speech ends
                                if is_speaking and silence_counter >= silence_threshold:
                                    if len(audio_buffer) > 0:
                                        await process_audio_buffer(audio_buffer, client_id)
                                        audio_buffer.clear()
                                        is_speaking = False
                                    silence_counter = 0
                            
                            # Prevent buffer overflow
                            if len(audio_buffer) > max_buffer_size:
                                await websocket.send_text("Processing long audio...")
                                await process_audio_buffer(audio_buffer, client_id)
                                audio_buffer.clear()
                                is_speaking = False
                                
                        except Exception as e:
                            logger.error(f"Error processing audio: {e}")
                            await websocket.send_text("Error processing audio")
                    else:
                        await websocket.send_text("Audio processor not available")
                
                else:
                    logger.warning(f"Unexpected message format: {message}")
                    
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
    finally:
        # Clean up
        if client_id in manager.active_connections:
            del manager.active_connections[client_id]
        logger.info(f"Client {client_id} disconnected")

async def process_audio_buffer(audio_buffer: bytearray, client_id: str):
    """Process accumulated audio buffer through STT → LLM → TTS pipeline."""
    if len(audio_buffer) == 0:
        return
    
    try:
        logger.info(f"Processing audio buffer of size {len(audio_buffer)} bytes for {client_id}")
        
        # Get the websocket directly
        websocket = manager.active_connections.get(client_id)
        if not websocket:
            logger.error(f"No active connection for {client_id}")
            return
        
        # Step 1: Convert audio to text (STT)
        await websocket.send_text("🔄 Processing speech...")
        
        # Call your STT service
        text = await manager.orchestrator.speech_to_text(bytes(audio_buffer))
        
        if not text or len(text.strip()) < 2:
            await websocket.send_text("❓ I didn't catch that. Could you repeat?")
            return
        
        await websocket.send_text(f"🗣️ You said: {text}")
        
        # Step 2: Get response from LLM
        await websocket.send_text("🤔 Thinking...")
        response_text = await manager.orchestrator.get_llm_response(text, client_id)
        
        # Step 3: Convert response to speech (TTS)
        await websocket.send_text("🎵 Generating audio response...")
        audio_response = await manager.orchestrator.text_to_speech(response_text)
        
        # Step 4: Send audio response back
        await websocket.send_text(f"🤖 Assistant: {response_text}")
        
        # Send audio as JSON
        audio_message = {
            "type": "audio",
            "data": audio_response.hex() if hasattr(audio_response, 'hex') else audio_response,
            "format": "wav"
        }
        await websocket.send_json(audio_message)
        
        logger.info(f"Audio processing completed for {client_id}")
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        websocket = manager.active_connections.get(client_id)
        if websocket:
            await websocket.send_text("⚠️ Sorry, I encountered an error processing your request.")

async def handle_text_message(data: dict, client_id: str):
    """Handle text messages (for testing or fallback mode)."""
    try:
        message_type = data.get("type", "text")
        
        if message_type == "control":
            # Handle control messages (start/stop recording, etc.)
            control_msg = ControlMessage(**data)
            
            if control_msg.command == "start_recording":
                await manager.send_text(client_id, "🎤 Recording started...")
            elif control_msg.command == "stop_recording":
                await manager.send_text(client_id, "⏹️ Recording stopped.")
            elif control_msg.command == "reset":
                await manager.send_text(client_id, "🔄 Conversation reset.")
                await manager.orchestrator.reset_conversation(client_id)
            elif control_msg.command == "ping":
                await manager.send_text(client_id, "🏓 pong")
                
        elif message_type == "text":
            # Direct text input (bypass STT)
            text_msg = TextMessage(**data)
            response_text = await manager.orchestrator.get_llm_response(text_msg.content, client_id)
            await manager.send_text(client_id, f"🤖 Assistant: {response_text}")
            
            # Optional: Also send TTS response
            try:
                audio_response = await manager.orchestrator.text_to_speech(response_text)
                await manager.send_audio(client_id, audio_response)
            except Exception as e:
                logger.warning(f"TTS failed for text message: {e}")
                
    except Exception as e:
        logger.error(f"Error handling text message: {str(e)}", exc_info=True)
        await manager.send_text(client_id, f"⚠️ Error: {str(e)}")
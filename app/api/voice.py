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
    
    # Accept the connection
    await websocket.accept()
    logger.info(f"✅ Client {client_id} connected")
    
    # Store connection
    manager.active_connections[client_id] = websocket
    
    # Buffer for audio chunks
    audio_buffer = bytearray()
    last_audio_time = asyncio.get_event_loop().time()
    
    # Send welcome message
    await websocket.send_text("✅ Connected to voice assistant. Click 'Start Listening' to begin.")
    
    try:
        while True:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                
                # Handle text messages
                if "text" in message:
                    text_data = message["text"]
                    logger.info(f"📝 Received text from {client_id}: {text_data}")
                    
                    try:
                        # Try to parse as JSON
                        data = json.loads(text_data)
                        
                        if data.get("type") == "connection":
                            await websocket.send_json({
                                "type": "text",
                                "message": f"Welcome {data.get('client', 'client')}!"
                            })
                        elif data.get("type") == "control" and data.get("command") == "reset":
                            audio_buffer.clear()
                            await websocket.send_text("🔄 Conversation reset")
                        else:
                            await websocket.send_text(f"📨 Received: {data}")
                            
                    except json.JSONDecodeError:
                        # Plain text message
                        await websocket.send_text(f"📨 You said: {text_data}")
                
                # Handle binary audio data
                elif "bytes" in message:
                    audio_chunk = message["bytes"]
                    logger.info(f"🎵 Received audio chunk from {client_id}: {len(audio_chunk)} bytes")
                    last_audio_time = asyncio.get_event_loop().time()
                    
                    # Add to buffer
                    audio_buffer.extend(audio_chunk)
                    
                    # Check if this is a complete recording (larger chunk)
                    if len(audio_chunk) > 10000:  # Larger chunks indicate final audio
                        logger.info(f"🎤 Large audio chunk received ({len(audio_chunk)} bytes), processing...")
                        await websocket.send_text("🔄 Processing your audio...")
                        
                        # Process the audio
                        if len(audio_buffer) > 0:
                            try:
                                await process_audio_buffer(audio_buffer, client_id)
                                audio_buffer.clear()
                            except Exception as e:
                                logger.error(f"Error processing audio: {e}")
                                await websocket.send_text("❌ Error processing audio")
                    else:
                        # Small chunk, just acknowledge
                        if len(audio_buffer) % 100000 < len(audio_chunk):  # Log every ~100KB
                            await websocket.send_text(f"👂 Listening... ({len(audio_buffer)//1000}KB)")
                
                else:
                    logger.warning(f"Unexpected message format from {client_id}")
                    
            except asyncio.TimeoutError:
                # Check if we have buffered audio to process
                current_time = asyncio.get_event_loop().time()
                if len(audio_buffer) > 0 and (current_time - last_audio_time) > 2.0:
                    # 2 seconds of silence, process buffered audio
                    logger.info(f"⏱️ Silence detected, processing buffered audio ({len(audio_buffer)} bytes)")
                    await websocket.send_text("🔄 Processing...")
                    
                    if len(audio_buffer) > 0:
                        try:
                            await process_audio_buffer(audio_buffer, client_id)
                            audio_buffer.clear()
                        except Exception as e:
                            logger.error(f"Error processing audio: {e}")
                            await websocket.send_text("❌ Error processing audio")
                
                # Send keep-alive
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": asyncio.get_event_loop().time()
                })
                
            except WebSocketDisconnect:
                logger.info(f"🔌 Client {client_id} disconnected")
                break
            except Exception as e:
                logger.error(f"❌ Error processing message from {client_id}: {e}")
                await websocket.send_text(f"❌ Error: {str(e)}")
                
    except Exception as e:
        logger.error(f"💥 WebSocket error for {client_id}: {str(e)}", exc_info=True)
    finally:
        # Clean up
        if client_id in manager.active_connections:
            del manager.active_connections[client_id]
        logger.info(f"👋 Client {client_id} disconnected completely")

async def process_audio_buffer(audio_buffer: bytearray, client_id: str):
    """Process accumulated audio buffer through STT → LLM → TTS pipeline."""
    if len(audio_buffer) == 0:
        return
    
    try:
        logger.info(f"🔄 Processing audio buffer of size {len(audio_buffer)} bytes for {client_id}")
        
        # Get the websocket
        websocket = manager.active_connections.get(client_id)
        if not websocket:
            logger.error(f"No active connection for {client_id}")
            return
        
        # Step 1: Convert audio to text (STT)
        await websocket.send_text("🗣️ Converting speech to text...")
        
        # Call your STT service - use bytes() instead of bytearray
        text = await manager.orchestrator.speech_to_text(bytes(audio_buffer))
        
        if not text or len(text.strip()) < 2:
            await websocket.send_text("❓ I didn't catch that. Could you repeat?")
            return
        
        await websocket.send_text(f"📝 You said: {text}")
        
        # Step 2: Get response from LLM
        await websocket.send_text("🤔 Thinking...")
        response_text = await manager.orchestrator.get_llm_response(text, client_id)
        
        # Step 3: Convert response to speech (TTS)
        await websocket.send_text("🎵 Generating audio response...")
        audio_response = await manager.orchestrator.text_to_speech(response_text)
        
        # Step 4: Send audio response back
        await websocket.send_text(f"🤖 Assistant: {response_text}")
        
        # Send audio as binary data (more efficient than hex encoding)
        await websocket.send_bytes(audio_response)
        
        logger.info(f"✅ Audio processing completed for {client_id}")
        
    except Exception as e:
        logger.error(f"❌ Error processing audio: {str(e)}", exc_info=True)
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
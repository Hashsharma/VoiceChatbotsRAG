from pydantic import BaseModel
from typing import Optional
from enum import Enum

class MessageType(str, Enum):
    AUDIO = "audio"
    TEXT = "text"
    CONTROL = "control"
    ERROR = "error"

class AudioMessage(BaseModel):
    type: MessageType = MessageType.AUDIO
    data: str  # hex encoded audio data
    format: str = "pcm_16000"
    sample_rate: int = 16000
    channels: int = 1

class TextMessage(BaseModel):
    type: MessageType = MessageType.TEXT
    content: str
    timestamp: Optional[float] = None
    is_final: bool = True

class ControlMessage(BaseModel):
    type: MessageType = MessageType.CONTROL
    command: str  # "start_recording", "stop_recording", "reset", "pause", "resume"
    value: Optional[str] = None
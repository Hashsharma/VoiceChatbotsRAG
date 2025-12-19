import os
import sys
from pathlib import Path

def create_voice_chatbot_structure(base_path="VoiceChatbotsRAG"):
    """Create the complete voice chatbot folder structure"""
    
    # Define the directory structure
    structure = {
        "app": {
            "__init__.py": "",
            "main.py": "",
            "config.py": "",
            "dependencies.py": "",
            "api": {
                "__init__.py": "",
                "chat.py": "",
                "voice.py": "",
                "health.py": "",
                "websocket_manager.py": "",
            },
            "core": {
                "__init__.py": "",
                "orchestrator.py": "",
                "llm": {
                    "__init__.py": "",
                    "manager.py": "",
                    "prompts.py": "",
                    "cache.py": "",
                },
                "memory": {
                    "__init__.py": "",
                    "short_term.py": "",
                    "long_term.py": "",
                    "summarizer.py": "",
                },
                "processing": {
                    "__init__.py": "",
                    "intents.py": "",
                    "entities.py": "",
                    "sentiment.py": "",
                },
            },
            "speech": {
                "__init__.py": "",
                "stt": {
                    "__init__.py": "",
                    "base.py": "",
                    "whisper.py": "",
                    "faster_whisper.py": "",
                    "streaming.py": "",
                },
                "tts": {
                    "__init__.py": "",
                    "base.py": "",
                    "elevenlabs.py": "",
                    "openai_tts.py": "",
                    "streaming.py": "",
                },
                "audio": {
                    "__init__.py": "",
                    "processor.py": "",
                    "vad.py": "",
                    "effects.py": "",
                    "codecs.py": "",
                },
            },
            "models": {
                "intents": "",
                "vad": "",
                "embeddings": "",
            },
            "services": {
                "__init__.py": "",
                "cache": {
                    "__init__.py": "",
                    "redis.py": "",
                    "in_memory.py": "",
                },
                "queue": {
                    "__init__.py": "",
                    "task_queue.py": "",
                    "events.py": "",
                },
                "monitoring": {
                    "__init__.py": "",
                    "metrics.py": "",
                    "tracing.py": "",
                },
            },
            "utils": {
                "__init__.py": "",
                "audio_utils.py": "",
                "streaming.py": "",
                "rate_limiter.py": "",
                "timing.py": "",
            },
            "schemas": {
                "__init__.py": "",
                "chat.py": "",
                "audio.py": "",
                "websocket.py": "",
            },
        },
        "tests": {
            "unit": {
                "test_stt.py": "",
                "test_orchestrator.py": "",
                "test_llm.py": "",
            },
            "integration": {
                "test_voice_flow.py": "",
                "test_concurrent.py": "",
            },
            "performance": {
                "test_latency.py": "",
            },
        },
        "data": {
            "audio": {
                "inputs": "",
                "outputs": "",
                "cache": "",
            },
            "conversations": {
                "sqlite": "",
                "backups": "",
            },
            "logs": {
                "voice_logs": "",
                "debug_audio": "",
            },
        },
        "scripts": {
            "train_intent.py": "",
            "benchmark_stt.py": "",
            "stress_test.py": "",
            "audio_cleanup.py": "",
        },
        "configs": {
            "dev.yaml": "",
            "prod.yaml": "",
            "models.yaml": "",
        },
    }

    # Additional root files
    root_files = [
        "README.md",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "Dockerfile",
        "docker-compose.yml",
        "prometheus.yml",
        "voice_flow_diagram.md",
    ]

    def create_nested_structure(base, structure_dict):
        """Recursively create directories and files"""
        for name, content in structure_dict.items():
            path = base / name
            
            if isinstance(content, dict):
                # It's a directory
                path.mkdir(parents=True, exist_ok=True)
                print(f"📁 Created directory: {path}")
                
                # Create __init__.py if it's a Python package
                if name not in ["models", "data", "tests", "scripts", "configs"]:
                    init_file = path / "__init__.py"
                    if not init_file.exists():
                        init_file.touch()
                        print(f"  📄 Created: {init_file}")
                
                create_nested_structure(path, content)
            elif isinstance(content, str):
                # It's a file
                path.touch()
                print(f"📄 Created file: {path}")
    
    # Create base directory
    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"🚀 Creating project structure at: {base_dir.absolute()}")
    
    # Create nested structure
    create_nested_structure(base_dir, structure)
    
    # Create root files
    for file_name in root_files:
        file_path = base_dir / file_name
        file_path.touch()
        print(f"📄 Created root file: {file_path}")
    
    print(f"\n✅ Project structure created successfully!")
    print(f"📊 Summary:")
    print(f"   Total directories: {sum(1 for _ in base_dir.rglob('*') if _.is_dir())}")
    print(f"   Total files: {sum(1 for _ in base_dir.rglob('*') if _.is_file())}")
    
    return base_dir


create_voice_chatbot_structure()
import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
FALLBACK_GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]

class GroqClientManager:
    """
    Manages Groq API client initialization, environment configuration,
    and connectivity checks with graceful fallback.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "").strip()
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
        self._client = None

    @property
    def is_available(self) -> bool:
        """Returns True if a non-empty API key is configured."""
        return bool(self.api_key and len(self.api_key) > 5)

    def get_client(self):
        """Initializes and returns the Groq client instance if available."""
        if not self.is_available:
            return None
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize Groq client: {e}")
                return None
        return self._client

    def get_available_models(self) -> List[str]:
        """Returns active chat models available on Groq account, or fallback defaults."""
        client = self.get_client()
        if client:
            try:
                models = client.models.list()
                chat_models = [
                    m.id for m in models.data
                    if "whisper" not in m.id and "guard" not in m.id
                ]
                if chat_models:
                    return sorted(chat_models)
            except Exception:
                pass
        return FALLBACK_GROQ_MODELS

    def update_credentials(self, api_key: str, model: Optional[str] = None):
        """Allows updating API credentials at runtime (e.g. from UI input)."""
        self.api_key = api_key.strip()
        if model:
            self.model = model.strip()
        self._client = None

"""
config.py — DefaultConfig
-------------------------
Centralizes all environment-variable-backed configuration for the chatbot.

Sensitive values (API keys, endpoints) are NEVER hardcoded here.
Set them as OS environment variables before launching the bot:

  export HF_API_TOKEN=hf_your_token_here        (macOS/Linux)
  SET    HF_API_TOKEN=hf_your_token_here         (Windows CMD)

See .env.example for the full list of supported variables.

Note: This class is intentionally modeled after the DefaultConfig pattern
described in the course assignment instructions for Azure-backed bots.
"""

import os


class DefaultConfig:
    """Reads runtime configuration from environment variables."""

    # HuggingFace Inference API token
    # Generate one at: https://huggingface.co/settings/tokens
    HF_API_TOKEN: str = os.environ.get("HF_API_TOKEN", "")

    # Optional: override the default model endpoint
    # Default points to cardiffnlp/twitter-roberta-base-sentiment-latest
    HF_MODEL_ENDPOINT: str = os.environ.get(
        "HF_MODEL_ENDPOINT",
        "https://api-inference.huggingface.co/models/"
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
    )

    # Request timeout in seconds (increase if you experience frequent timeouts)
    REQUEST_TIMEOUT_SECONDS: int = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "15"))

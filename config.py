"""
Configuration and environment variable loading.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration from environment variables."""

    notion_token: str
    notion_page_id: str  # The parent page containing company doc subpages
    openrouter_api_key: str
    mastodon_token: str  # For future use (manual posting reference)
    mastodon_instance: str

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        load_dotenv()

        required_vars = [
            "NOTION_TOKEN",
            "NOTION_PAGE_ID",
            "OPENROUTER_API_KEY",
        ]

        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

        return cls(
            notion_token=os.getenv("NOTION_TOKEN", ""),
            notion_page_id=os.getenv("NOTION_PAGE_ID", ""),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            mastodon_token=os.getenv("MASTODON_TOKEN", ""),  # Optional for now
            mastodon_instance=os.getenv("MASTODON_INSTANCE", "https://mastodon.social"),
        )


# LLM settings
LLM_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"  # Via OpenRouter (free tier)
LLM_MAX_TOKENS = 4096  # Increased for reasoning models
LLM_TEMPERATURE = 0.7

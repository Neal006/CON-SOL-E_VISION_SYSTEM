"""Configuration for PLC Dashboard System — loads from .env."""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class PLCConfig:
    """PLC connection settings (MC Protocol)."""
    HOST: str = os.getenv("PLC_HOST", "192.168.1.100")
    PORT: int = int(os.getenv("PLC_PORT", "1025"))
    TIMEOUT: int = int(os.getenv("PLC_TIMEOUT", "5"))
    MAX_RETRIES: int = int(os.getenv("PLC_MAX_RETRIES", "3"))
    RETRY_DELAY: int = int(os.getenv("PLC_RETRY_DELAY", "2"))


class APIConfig:
    """FastAPI server settings."""
    API_KEY: str = os.getenv("API_KEY", "your-secret-api-key-change-this")
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000").split(",")
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    ENABLE_DOCS: bool = os.getenv("ENABLE_DOCS", "True").lower() == "true"


class DashboardConfig:
    """Streamlit dashboard settings."""
    PAGE_TITLE: str = "PLC Control Dashboard"
    PAGE_ICON: str = "⚙️"
    LAYOUT: str = "wide"
    REFRESH_INTERVAL: int = int(os.getenv("REFRESH_INTERVAL", "2"))
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    API_KEY: str = os.getenv("API_KEY", "your-secret-api-key-change-this")


class ColorTheme:
    """Dashboard color palette — dark industrial theme."""
    PRIMARY: str = "#1E88E5"
    SECONDARY: str = "#455A64"
    SUCCESS: str = "#4CAF50"
    WARNING: str = "#FF9800"
    ERROR: str = "#F44336"
    INFO: str = "#2196F3"
    DARK_BG: str = "#0E1117"
    CARD_BG: str = "#1E2127"
    TEXT_PRIMARY: str = "#FAFAFA"
    TEXT_SECONDARY: str = "#B0BEC5"
    ACCENT: str = "#FF6F00"

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file from the current project directory
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)


class Settings(BaseSettings):
    # SAP Target System Configuration
    SAP_SYSTEM_ID: str = os.getenv("SAP_SYSTEM_ID", "DEV")
    SAP_CLIENT: str = os.getenv("SAP_CLIENT", "100")
    SAP_S4HANA_RELEASE: str = os.getenv("SAP_S4HANA_RELEASE", "2023")
    SAP_BASE_URL: str = os.getenv("SAP_BASE_URL", "https://YAWSS4HSBX.sapyash.com:44301").rstrip("/")
    SAP_USERNAME: str = os.getenv("SAP_USERNAME", "AIENAPARTHIP")
    SAP_PASSWORD: str = os.getenv("SAP_PASSWORD", "Training@2026")
    SAP_VERIFY_SSL: bool = os.getenv("SAP_VERIFY_SSL", "false").lower() in ("true", "1", "yes")
    SAP_REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("SAP_REQUEST_TIMEOUT_SECONDS", "30"))

    # Graph and Vector DB
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma_db"))

    # LLM Providers
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "claude-3-5-sonnet-20241022")

    class Config:
        env_file = ".env"
        extra = "allow"


# Global settings instance
settings = Settings()

import os
import yaml
from pathlib import Path
from typing import Any
from typing import Optional
from pydantic import BaseModel
from pydantic import PrivateAttr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.absolute()


class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float
    max_tokens: int


class EmbeddingConfig(BaseModel):
    model: str
    dimension: int


class VectorDBConfig(BaseModel):
    type: str
    location: str
    collection_name: str


class ChunkingConfig(BaseModel):
    strategy: str  # semantic or fixed
    chunk_size: int
    chunk_overlap: int
    semantic_threshold: float


class SSLConfig(BaseModel):
    enable_verification: bool


class DeduplicationConfig(BaseModel):
    enabled: bool
    similarity_threshold: float
    information_weight: float


class LanguageConfig(BaseModel):
    detection_enabled: bool
    translation_provider: str
    supported_languages: list
    primary_language: str


class APIConfig(BaseModel):
    host: str
    port: int
    debug: bool
    cors_enabled: bool
    max_request_size: int


class Settings(BaseModel):
    """
    Validated application settings loaded from config/config.yaml and environment
    variables. The locale property is lazily loaded on first access from the
    SETTLEBOT_LOCALE environment variable.
    """

    environment: str
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_db: VectorDBConfig
    chunking: ChunkingConfig
    deduplication: DeduplicationConfig
    language: LanguageConfig
    api: APIConfig
    ssl: SSLConfig

    _locale: Optional[Any] = PrivateAttr(default=None)

    @classmethod
    def from_yaml(cls, file_path: Path) -> "Settings":
        """
        Load Settings from a YAML configuration file.
        :param file_path: Path - Path to the YAML config file.
        :return: Settings - Validated settings instance.
        """
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return cls.parse_obj(config_dict)

    @property
    def locale(self) -> Optional[Any]:
        """
        Lazily loaded LocaleConfig for the active deployment.
        Reads SETTLEBOT_LOCALE and loads config/locale/{locale_name}_config.json
        on first access. Returns None when SETTLEBOT_LOCALE is not set; services
        operate without locale injection in that case.
        :return: Optional[LocaleConfig] - Active locale configuration, or None.
        """
        if self._locale is None:
            locale_name = os.getenv("SETTLEBOT_LOCALE")
            if not locale_name:
                return None
            from config.locale import LocaleConfig  # lazy import to avoid circular

            object.__setattr__(self, "_locale", LocaleConfig.from_file(locale_name))
        return self._locale


# Initialize settings from YAML config file
settings_file = ROOT_DIR / "config" / "config.yaml"
settings = Settings.from_yaml(settings_file)

# Override with environment variables if provided
if os.getenv("HOST"):
    settings.api.host = os.getenv("HOST")

if os.getenv("PORT"):
    settings.api.port = int(os.getenv("PORT"))

if os.getenv("CHUNKING_STRATEGY"):
    settings.chunking.strategy = os.getenv("CHUNKING_STRATEGY")

if os.getenv("DEDUPLICATION_ENABLED"):
    settings.deduplication.enabled = (
        os.getenv("DEDUPLICATION_ENABLED").lower() == "true"
    )

_PLACEHOLDER_KEY = "your_secure_random_key_here"
_settlebot_api_key = os.getenv("SETTLEBOT_API_KEY")
if not _settlebot_api_key or _settlebot_api_key == _PLACEHOLDER_KEY:
    raise ValueError(
        "SETTLEBOT_API_KEY environment variable is not set or uses the insecure "
        "placeholder value. Generate with: "
        'python -c \'import secrets; sys=__import__("sys"); '
        "sys.stdout.write(secrets.token_urlsafe(32))'"
    )

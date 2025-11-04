"""
Configuration management for LLM evaluation framework.

DECISION RATIONALE:
- Environment-based configuration for security (API keys in environment variables)
- Centralized configuration management for maintainability
- Type hints for configuration validation
- Default values for development/testing environments

References:
- Python best practices for configuration management (PEP 8)
- 12-factor app methodology for environment-based config
"""

import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """
    Configuration class for LLM evaluation framework.
    
    All sensitive values (API keys) must be provided via environment variables.
    Default values are provided for non-sensitive configuration.
    """
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    OUTPUT_DIR: Path = PROJECT_ROOT / "output"
    LOG_DIR: Path = PROJECT_ROOT / "logs"
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    
    # AWS Bedrock Configuration
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    BEDROCK_MODEL_ID: str = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
    
    # HuggingFace Configuration
    HUGGINGFACE_HUB_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_HUB_TOKEN")
    HF_CACHE_DIR: Path = Path(os.getenv("HF_CACHE_DIR", PROJECT_ROOT / ".cache" / "huggingface"))
    
    # Evaluation Configuration
    EVAL_BATCH_SIZE: int = int(os.getenv("EVAL_BATCH_SIZE", "8"))
    EVAL_MAX_SAMPLES: int = int(os.getenv("EVAL_MAX_SAMPLES", "100"))
    STATISTICAL_CONFIDENCE_LEVEL: float = float(os.getenv("STATISTICAL_CONFIDENCE_LEVEL", "0.95"))
    BOOTSTRAP_ITERATIONS: int = int(os.getenv("BOOTSTRAP_ITERATIONS", "1000"))
    
    # FAISS Configuration
    FAISS_INDEX_TYPE: str = os.getenv("FAISS_INDEX_TYPE", "L2")  # L2 or COSINE
    FAISS_N_PROBE: int = int(os.getenv("FAISS_N_PROBE", "64"))
    
    # RAG Configuration
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
    RAG_CONTEXT_WINDOW: int = int(os.getenv("RAG_CONTEXT_WINDOW", "4096"))
    
    # Toxicity Detection Configuration
    TOXICITY_THRESHOLD: float = float(os.getenv("TOXICITY_THRESHOLD", "0.7"))
    PERSPECTIVE_API_KEY: Optional[str] = os.getenv("PERSPECTIVE_API_KEY")
    
    # Code Quality Configuration
    MCCABE_COMPLEXITY_THRESHOLD: int = int(os.getenv("MCCABE_COMPLEXITY_THRESHOLD", "10"))
    COGNITIVE_COMPLEXITY_THRESHOLD: int = int(os.getenv("COGNITIVE_COMPLEXITY_THRESHOLD", "15"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Open-Source Model Configuration
    USE_OPENSOURCE_MODELS: bool = os.getenv("USE_OPENSOURCE_MODELS", "true").lower() == "true"
    HF_LLM_MODEL: str = os.getenv("HF_LLM_MODEL", "gpt2")  # Small model that works in free Colab
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration values.
        
        Returns:
            bool: True if configuration is valid, raises ValueError otherwise
        
        Raises:
            ValueError: If required configuration is missing
        """
        errors = []
        
        # Check for required API keys only if not using open-source models
        if not cls.USE_OPENSOURCE_MODELS:
            if not cls.OPENAI_API_KEY and not (cls.AWS_ACCESS_KEY_ID and cls.AWS_SECRET_ACCESS_KEY):
                errors.append("At least one LLM provider (OpenAI or AWS Bedrock) must be configured, or set USE_OPENSOURCE_MODELS=true")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [cls.DATA_DIR, cls.OUTPUT_DIR, cls.LOG_DIR, cls.HF_CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


# Validate configuration on import
try:
    Config.validate()
    Config.create_directories()
except ValueError as e:
    import warnings
    warnings.warn(f"Configuration validation warning: {e}", UserWarning)


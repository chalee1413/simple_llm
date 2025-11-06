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
    DATA_DIR: Path = PROJECT_ROOT / "data"  # Runtime data (ignored)
    EXAMPLE_INPUTS_DIR: Path = PROJECT_ROOT / "example_inputs"  # Example inputs (tracked)
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
    
    # RLHF Configuration
    RLHF_DATA_DIR: Path = PROJECT_ROOT / "example_inputs" / "rlhf"
    RLHF_OUTPUT_DIR: Path = PROJECT_ROOT / "output" / "rlhf"
    RLHF_PREFERENCES_DIR: Path = PROJECT_ROOT / "example_inputs" / "rlhf" / "preferences"
    RLHF_MODELS_DIR: Path = PROJECT_ROOT / "output" / "rlhf" / "models"
    
    # SFT Configuration
    SFT_LEARNING_RATE: float = float(os.getenv("SFT_LEARNING_RATE", "2e-5"))
    SFT_BATCH_SIZE: int = int(os.getenv("SFT_BATCH_SIZE", "4"))
    SFT_GRADIENT_ACCUMULATION_STEPS: int = int(os.getenv("SFT_GRADIENT_ACCUMULATION_STEPS", "4"))
    SFT_NUM_EPOCHS: int = int(os.getenv("SFT_NUM_EPOCHS", "3"))
    SFT_WARMUP_STEPS: int = int(os.getenv("SFT_WARMUP_STEPS", "100"))
    SFT_WEIGHT_DECAY: float = float(os.getenv("SFT_WEIGHT_DECAY", "0.01"))
    SFT_MAX_SEQ_LENGTH: int = int(os.getenv("SFT_MAX_SEQ_LENGTH", "512"))
    
    # Reward Model Configuration
    REWARD_MODEL_LEARNING_RATE: float = float(os.getenv("REWARD_MODEL_LEARNING_RATE", "1e-5"))
    REWARD_MODEL_BATCH_SIZE: int = int(os.getenv("REWARD_MODEL_BATCH_SIZE", "4"))
    REWARD_MODEL_GRADIENT_ACCUMULATION_STEPS: int = int(os.getenv("REWARD_MODEL_GRADIENT_ACCUMULATION_STEPS", "4"))
    REWARD_MODEL_NUM_EPOCHS: int = int(os.getenv("REWARD_MODEL_NUM_EPOCHS", "1"))
    REWARD_MODEL_WARMUP_STEPS: int = int(os.getenv("REWARD_MODEL_WARMUP_STEPS", "100"))
    
    # PPO Configuration
    PPO_LEARNING_RATE: float = float(os.getenv("PPO_LEARNING_RATE", "1e-5"))
    PPO_BATCH_SIZE: int = int(os.getenv("PPO_BATCH_SIZE", "8"))
    PPO_MINIBATCH_SIZE: int = int(os.getenv("PPO_MINIBATCH_SIZE", "2"))
    PPO_GRADIENT_ACCUMULATION_STEPS: int = int(os.getenv("PPO_GRADIENT_ACCUMULATION_STEPS", "4"))
    PPO_NUM_EPOCHS: int = int(os.getenv("PPO_NUM_EPOCHS", "4"))
    PPO_KL_PENALTY: float = float(os.getenv("PPO_KL_PENALTY", "0.1"))
    PPO_CLIP_RANGE: float = float(os.getenv("PPO_CLIP_RANGE", "0.2"))
    PPO_VALUE_COEF: float = float(os.getenv("PPO_VALUE_COEF", "0.1"))
    PPO_ENTROPY_COEF: float = float(os.getenv("PPO_ENTROPY_COEF", "0.01"))
    PPO_GAMMA: float = float(os.getenv("PPO_GAMMA", "1.0"))
    PPO_LAM: float = float(os.getenv("PPO_LAM", "0.95"))
    PPO_MAX_SEQ_LENGTH: int = int(os.getenv("PPO_MAX_SEQ_LENGTH", "512"))
    
    # DPO Configuration
    DPO_LEARNING_RATE: float = float(os.getenv("DPO_LEARNING_RATE", "1e-5"))
    DPO_BATCH_SIZE: int = int(os.getenv("DPO_BATCH_SIZE", "4"))
    DPO_GRADIENT_ACCUMULATION_STEPS: int = int(os.getenv("DPO_GRADIENT_ACCUMULATION_STEPS", "4"))
    DPO_NUM_EPOCHS: int = int(os.getenv("DPO_NUM_EPOCHS", "1"))
    DPO_BETA: float = float(os.getenv("DPO_BETA", "0.1"))  # Temperature parameter for DPO
    DPO_MAX_SEQ_LENGTH: int = int(os.getenv("DPO_MAX_SEQ_LENGTH", "512"))
    
    # KTO Configuration
    KTO_LEARNING_RATE: float = float(os.getenv("KTO_LEARNING_RATE", "1e-5"))
    KTO_BATCH_SIZE: int = int(os.getenv("KTO_BATCH_SIZE", "4"))
    KTO_GRADIENT_ACCUMULATION_STEPS: int = int(os.getenv("KTO_GRADIENT_ACCUMULATION_STEPS", "4"))
    KTO_NUM_EPOCHS: int = int(os.getenv("KTO_NUM_EPOCHS", "1"))
    KTO_BETA: float = float(os.getenv("KTO_BETA", "0.1"))  # Temperature parameter for KTO
    KTO_MAX_SEQ_LENGTH: int = int(os.getenv("KTO_MAX_SEQ_LENGTH", "512"))
    
    # RLHF Training Configuration
    RLHF_ALGORITHM: str = os.getenv("RLHF_ALGORITHM", "ppo")  # ppo, dpo, or kto
    RLHF_CHECKPOINT_STEPS: int = int(os.getenv("RLHF_CHECKPOINT_STEPS", "100"))
    RLHF_EVAL_STEPS: int = int(os.getenv("RLHF_EVAL_STEPS", "50"))
    RLHF_SAVE_STEPS: int = int(os.getenv("RLHF_SAVE_STEPS", "100"))
    RLHF_LOGGING_STEPS: int = int(os.getenv("RLHF_LOGGING_STEPS", "10"))
    
    # RLHF Data Configuration
    RLHF_PREFERENCE_BATCH_SIZE: int = int(os.getenv("RLHF_PREFERENCE_BATCH_SIZE", "10"))
    RLHF_MIN_PREFERENCES: int = int(os.getenv("RLHF_MIN_PREFERENCES", "100"))
    
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
        for directory in [
            cls.DATA_DIR, cls.OUTPUT_DIR, cls.LOG_DIR, cls.HF_CACHE_DIR,
            cls.EXAMPLE_INPUTS_DIR, cls.RLHF_DATA_DIR, cls.RLHF_OUTPUT_DIR, 
            cls.RLHF_PREFERENCES_DIR, cls.RLHF_MODELS_DIR
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Validate configuration on import
try:
    Config.validate()
    Config.create_directories()
except ValueError as e:
    import warnings
    warnings.warn(f"Configuration validation warning: {e}", UserWarning)


"""
RLHF (Reinforcement Learning from Human Feedback) module.

DECISION RATIONALE:
- Implements complete RLHF pipeline following 2025 SoTA practices
- Includes SFT, reward model training, PPO/DPO optimization
- Integrates with existing evaluation framework
- Supports both PPO and DPO algorithms

References:
- Schulman et al. (2017). Proximal Policy Optimization. https://arxiv.org/abs/1707.06347
- Rafailov et al. (2024). Direct Preference Optimization. https://arxiv.org/abs/2305.18290
"""

from .data_preprocessing import (
    prepare_sft_dataset,
    load_instruction_dataset,
    format_instruction_prompt,
    validate_instruction_data
)
from .supervised_finetuning import (
    SupervisedFineTuner,
    train_sft_model
)
from .preference_collection import (
    PreferenceCollector,
    save_preference_data,
    load_preference_data
)
from .reward_model import (
    RewardModel,
    train_reward_model
)
from .ppo_trainer import (
    PPOTrainerWrapper,
    train_with_ppo
)
from .dpo_trainer import (
    DPOTrainerWrapper,
    train_with_dpo
)
from .evaluation_metrics import (
    RewardModelEvaluator,
    TrainingMetricsTracker,
    evaluate_reward_model
)
from .kto_trainer import (
    KTOTrainerWrapper,
    train_with_kto,
    prepare_kto_dataset
)

__all__ = [
    "prepare_sft_dataset",
    "load_instruction_dataset",
    "format_instruction_prompt",
    "validate_instruction_data",
    "SupervisedFineTuner",
    "train_sft_model",
    "PreferenceCollector",
    "save_preference_data",
    "load_preference_data",
    "RewardModel",
    "train_reward_model",
    "PPOTrainerWrapper",
    "train_with_ppo",
    "DPOTrainerWrapper",
    "train_with_dpo",
    "KTOTrainerWrapper",
    "train_with_kto",
    "prepare_kto_dataset",
    "RewardModelEvaluator",
    "TrainingMetricsTracker",
    "evaluate_reward_model",
]

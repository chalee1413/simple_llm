"""
Human preference collection for RLHF pipeline.

DECISION RATIONALE:
- Pairwise comparison interface for human preferences
- Preference data storage and validation
- Batch collection workflow
- Statistics and quality metrics

References:
- Human preference collection methodologies (2024-2025)
- Pairwise comparison format for RLHF
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from config import Config

logger = logging.getLogger(__name__)


class PreferenceCollector:
    """
    Collect human preferences for RLHF training.
    
    DECISION RATIONALE:
    - Encapsulates preference collection logic
    - Supports pairwise comparison format
    - Validates preference data
    - Provides statistics and quality metrics
    """
    
    def __init__(self, preferences_dir: Optional[Path] = None):
        """
        Initialize preference collector.
        
        Args:
            preferences_dir: Directory to store preference data
        """
        self.preferences_dir = preferences_dir or Config.RLHF_PREFERENCES_DIR
        self.preferences_dir.mkdir(parents=True, exist_ok=True)
        
        self.preferences: List[Dict[str, Any]] = []
        logger.info(f"PreferenceCollector initialized. Preferences stored in: {self.preferences_dir}")
    
    def add_preference(
        self,
        prompt: str,
        response_chosen: str,
        response_rejected: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a preference comparison.
        
        Args:
            prompt: Input prompt
            response_chosen: Preferred response
            response_rejected: Rejected response
            metadata: Optional metadata (timestamp, annotator, etc.)
        """
        preference = {
            "prompt": prompt,
            "chosen": response_chosen,
            "rejected": response_rejected,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.preferences.append(preference)
        logger.debug(f"Added preference: {len(self.preferences)} total")
    
    def save_preferences(self, filename: Optional[str] = None):
        """
        Save collected preferences to file.
        
        Args:
            filename: Optional filename (default: preferences_<timestamp>.json)
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"preferences_{timestamp}.json"
        
        file_path = self.preferences_dir / filename
        
        data = {
            "total_preferences": len(self.preferences),
            "created_at": datetime.now().isoformat(),
            "preferences": self.preferences
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.preferences)} preferences to: {file_path}")
        return file_path
    
    def load_preferences(self, file_path: Path):
        """
        Load preferences from file.
        
        Args:
            file_path: Path to preferences JSON file
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Preferences file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.preferences = data.get("preferences", [])
        logger.info(f"Loaded {len(self.preferences)} preferences from: {file_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get preference collection statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.preferences:
            return {
                "total_preferences": 0,
                "avg_prompt_length": 0,
                "avg_chosen_length": 0,
                "avg_rejected_length": 0
            }
        
        prompt_lengths = [len(p["prompt"]) for p in self.preferences]
        chosen_lengths = [len(p["chosen"]) for p in self.preferences]
        rejected_lengths = [len(p["rejected"]) for p in self.preferences]
        
        return {
            "total_preferences": len(self.preferences),
            "avg_prompt_length": sum(prompt_lengths) / len(prompt_lengths),
            "avg_chosen_length": sum(chosen_lengths) / len(chosen_lengths),
            "avg_rejected_length": sum(rejected_lengths) / len(rejected_lengths),
            "min_prompt_length": min(prompt_lengths),
            "max_prompt_length": max(prompt_lengths),
            "min_chosen_length": min(chosen_lengths),
            "max_chosen_length": max(chosen_lengths),
            "min_rejected_length": min(rejected_lengths),
            "max_rejected_length": max(rejected_lengths)
        }
    
    def validate_preferences(self) -> bool:
        """
        Validate preference data.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        if len(self.preferences) == 0:
            raise ValueError("No preferences collected")
        
        required_fields = ["prompt", "chosen", "rejected"]
        
        for i, pref in enumerate(self.preferences):
            if not isinstance(pref, dict):
                raise ValueError(f"Preference {i} is not a dictionary")
            
            for field in required_fields:
                if field not in pref:
                    raise ValueError(f"Preference {i} missing required field: {field}")
                
                if not isinstance(pref[field], str) or len(pref[field].strip()) == 0:
                    raise ValueError(f"Preference {i} has invalid {field}")
        
        logger.info(f"Validated {len(self.preferences)} preferences")
        return True


def save_preference_data(
    preferences: List[Dict[str, Any]],
    file_path: Path
):
    """
    Save preference data to file.
    
    Args:
        preferences: List of preference dictionaries
        file_path: Path to save file
    """
    data = {
        "total_preferences": len(preferences),
        "created_at": datetime.now().isoformat(),
        "preferences": preferences
    }
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved {len(preferences)} preferences to: {file_path}")


def load_preference_data(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load preference data from file.
    
    Args:
        file_path: Path to preferences JSON file
    
    Returns:
        List of preference dictionaries
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Preferences file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    preferences = data.get("preferences", [])
    logger.info(f"Loaded {len(preferences)} preferences from: {file_path}")
    return preferences

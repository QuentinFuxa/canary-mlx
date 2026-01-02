"""
Canary MLX - NVIDIA Canary ASR model implementation for Apple Silicon.

A fast, efficient implementation of the Canary speech recognition model
optimized for Apple's MLX framework.
"""

import json
from pathlib import Path

import mlx.core as mx
from dacite import from_dict
from huggingface_hub import hf_hub_download, snapshot_download
from mlx.utils import tree_flatten, tree_unflatten

from canary_mlx.alignment import AlignedSentence, AlignedToken, TranscriptionResult
from canary_mlx.model import Canary, CanaryConfig, DecodingConfig

__version__ = "0.1.0"
__all__ = [
    "Canary",
    "CanaryConfig",
    "DecodingConfig",
    "TranscriptionResult",
    "AlignedToken",
    "AlignedSentence",
    "load_model",
]

# Default HuggingFace model ID
DEFAULT_MODEL_ID = "your-username/canary-mlx"


def load_model(
    path_or_hf_id: str | Path = DEFAULT_MODEL_ID,
    *,
    dtype: mx.Dtype = mx.bfloat16,
) -> Canary:
    """
    Load a Canary model from a local directory or HuggingFace Hub.
    
    Args:
        path_or_hf_id: Local path to model directory, or HuggingFace model ID.
                       If not specified, downloads the default model.
        dtype: Data type for model weights (default: bfloat16).
        
    Returns:
        Loaded Canary model ready for inference.
        
    Examples:
        # Load from HuggingFace (downloads automatically)
        >>> from canary_mlx import load_model
        >>> model = load_model("your-username/canary-mlx")
        >>> result = model.transcribe("audio.wav", language="en")
        
        # Load from local directory
        >>> model = load_model("./my_model")
    """
    path = Path(path_or_hf_id)
    
    # Check if it's a local path
    if path.exists() and path.is_dir():
        model_dir = path
    else:
        # Treat as HuggingFace model ID
        try:
            model_dir = Path(snapshot_download(
                str(path_or_hf_id),
                allow_patterns=["*.json", "*.safetensors", "*.model"],
            ))
        except Exception as e:
            raise ValueError(
                f"Could not load model from '{path_or_hf_id}'. "
                f"If it's a local path, make sure it exists. "
                f"If it's a HuggingFace model ID, check the ID is correct. "
                f"Error: {e}"
            )
    
    config_path = model_dir / "config.json"
    weight_path = model_dir / "model.safetensors"
    
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    if not weight_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {model_dir}")
    
    config = json.load(open(config_path, "r"))
    
    # Add model directory for tokenizer loading
    config["model_dir"] = model_dir
    
    # Create model from config
    cfg = from_dict(CanaryConfig, config)
    model = Canary(cfg)
    model.eval()
    
    # Load weights
    model.load_weights(str(weight_path))
    
    # Cast to desired dtype
    curr_weights = dict(tree_flatten(model.parameters()))
    curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
    model.update(tree_unflatten(curr_weights))
    
    return model

import json
import os
import shutil
from pathlib import Path

import torch 
import yaml
from safetensors.torch import save_file


INPUT_DIR = Path("canary_untared")
OUTPUT_DIR = Path("model")

INPUT_WEIGHTS = INPUT_DIR / "model_weights.ckpt"
INPUT_CONFIG = INPUT_DIR / "model_config.yaml"

OUTPUT_WEIGHTS = OUTPUT_DIR / "model.safetensors"
OUTPUT_CONFIG = OUTPUT_DIR / "config.json"


def remap_weight_key(key: str) -> str:
    """Remap NeMo weight keys to MLX expected keys."""
    
    if key.startswith("preprocessor"):
        return None    
    if "position_embedding.pos_enc" in key:
        return None    
    if "num_batches_tracked" in key:
        return None    
    if key.startswith("transf_decoder._embedding."):
        key = key.replace("transf_decoder._embedding.", "transf_decoder.")
        key = key.replace("transf_decoder.layer_norm.", "transf_decoder.embedding_layer_norm.")
    if key.startswith("transf_decoder._decoder.layers."):
        key = key.replace("transf_decoder._decoder.layers.", "transf_decoder.layers.")
    if key.startswith("transf_decoder._decoder.final_layer_norm"):
        key = key.replace("transf_decoder._decoder.final_layer_norm", "transf_decoder.final_layer_norm")
    if key.startswith("log_softmax.mlp.layer0"):
        key = key.replace("log_softmax.mlp.layer0", "head.classifier")
    key = key.replace(".query_net.", ".linear_q.")
    key = key.replace(".key_net.", ".linear_k.")
    key = key.replace(".value_net.", ".linear_v.")
    key = key.replace(".out_projection.", ".linear_out.")
    key = key.replace(".third_sub_layer.dense_in.", ".third_sub_layer.linear1.")
    key = key.replace(".third_sub_layer.dense_out.", ".third_sub_layer.linear2.")
    
    return key


def convert_weights(input_path: Path, output_path: Path) -> None:
    """Convert NeMo checkpoint to MLX safetensors."""
    state_dict = torch.load(input_path, map_location="cpu", weights_only=False)
    
    skipped_keys = []
    new_state = {}

    for key, value in state_dict.items():
        new_key = remap_weight_key(key)
        
        if new_key is None:
            skipped_keys.append(key)
            continue
        
        if isinstance(value, torch.Tensor):
            value = value.clone()
        
        if "conv" in new_key and "weight" in new_key:
            if len(value.shape) == 4:
                value = value.permute(0, 2, 3, 1).contiguous()
            elif len(value.shape) == 3:
                value = value.permute(0, 2, 1).contiguous()
        
        new_state[new_key] = value
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(new_state, str(output_path))
    print(f"Saved weights to {output_path}")


def convert_config(input_path: Path, output_path: Path) -> dict:
    """Convert NeMo YAML config to MLX JSON config."""
    
    with open(input_path, "r") as f:
        nemo_config = yaml.safe_load(f)
    
    preproc = nemo_config.get("preprocessor", {})
    preprocessor = {
        "sample_rate": preproc.get("sample_rate", 16000),
        "normalize": preproc.get("normalize", "per_feature"),
        "window_size": preproc.get("window_size", 0.025),
        "window_stride": preproc.get("window_stride", 0.01),
        "window": preproc.get("window", "hann"),
        "features": preproc.get("features", 128),
        "n_fft": preproc.get("n_fft", 512),
        "dither": preproc.get("dither", 1e-5),
        "pad_to": preproc.get("pad_to", 0),
        "pad_value": preproc.get("pad_value", 0),
        "preemph": preproc.get("preemph", 0.97),
        "mag_power": preproc.get("mag_power", 2.0),
    }
    
    enc = nemo_config.get("encoder", {})
    encoder = {
        "feat_in": enc.get("feat_in", 128),
        "n_layers": enc.get("n_layers", 24),
        "d_model": enc.get("d_model", 1024),
        "n_heads": enc.get("n_heads", 8),
        "ff_expansion_factor": enc.get("ff_expansion_factor", 4),
        "subsampling_factor": enc.get("subsampling_factor", 8),
        "self_attention_model": enc.get("self_attention_model", "rel_pos"),
        "subsampling": enc.get("subsampling", "dw_striding"),
        "conv_kernel_size": enc.get("conv_kernel_size", 9),
        "subsampling_conv_channels": enc.get("subsampling_conv_channels", 256),
        "pos_emb_max_len": enc.get("pos_emb_max_len", 5000),
        "causal_downsampling": enc.get("causal_downsampling", False),
        "use_bias": enc.get("use_bias", True),
        "xscaling": enc.get("xscaling", False),
        "subsampling_conv_chunking_factor": enc.get("subsampling_conv_chunking_factor", 1),
        "att_context_size": enc.get("att_context_size"),
    }
    
    dec_raw = nemo_config.get("transf_decoder", {})
    dec = dec_raw.get("config_dict", dec_raw)  # Use config_dict if present
    
    vocab_size = dec.get("vocab_size", "None")
    if vocab_size == "None" or vocab_size is None:
        head_cfg = nemo_config.get("head", {})
        vocab_size = head_cfg.get("num_classes", 4096)
    
    transf_decoder = {
        "vocab_size": vocab_size,
        "hidden_size": dec.get("hidden_size", 1024),
        "inner_size": dec.get("inner_size", 4096),
        "num_layers": dec.get("num_layers", 8),
        "num_attention_heads": dec.get("num_attention_heads", 8),
        "pre_ln": dec.get("pre_ln", True),
        "hidden_act": dec.get("hidden_act", "relu"),
        "pre_ln_final_layer_norm": dec_raw.get("pre_ln_final_layer_norm", dec.get("pre_ln_final_layer_norm", True)),
        "learn_positional_encodings": dec.get("learn_positional_encodings", False),
        "max_sequence_length": dec.get("max_sequence_length", 1024),
    }
    
    head_cfg = nemo_config.get("head", {})
    head = {
        "num_layers": head_cfg.get("num_layers", 1),
        "hidden_size": transf_decoder["hidden_size"],
        "num_classes": head_cfg.get("num_classes", transf_decoder["vocab_size"]),
    }
    
    mlx_config = {
        "preprocessor": preprocessor,
        "encoder": encoder,
        "transf_decoder": transf_decoder,
        "head": head,
        "prompt_format": nemo_config.get("prompt_format", "canary"),
        "tokenizer": {
            "type": "sentencepiece",
            "model_path": "tokenizer.model",
        },
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mlx_config, f, indent=2)
    print(f"Saved config to {output_path}")
    
    return mlx_config


def copy_tokenizer(input_dir: Path, output_dir: Path) -> None:
    """Copy tokenizer files to output directory."""
    tokenizer_files = list(input_dir.glob("*_tokenizer.model"))
    
    if not tokenizer_files:
        print("Warning: No tokenizer files found!")
        return
    
    src_path = tokenizer_files[0]
    dst_path = output_dir / "tokenizer.model"
    
    shutil.copy2(src_path, dst_path)
    print(f"Copied tokenizer from {src_path.name} to {dst_path}")


def main():
    
    if not INPUT_WEIGHTS.exists():
        print(f"Error: {INPUT_WEIGHTS} not found!")
        return
    
    if not INPUT_CONFIG.exists():
        print(f"Error: {INPUT_CONFIG} not found!")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    convert_config(INPUT_CONFIG, OUTPUT_CONFIG)
    convert_weights(INPUT_WEIGHTS, OUTPUT_WEIGHTS)
    copy_tokenizer(INPUT_DIR, OUTPUT_DIR)
    
    print(f"Conversion complete, model saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

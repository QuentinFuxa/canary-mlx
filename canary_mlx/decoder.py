"""Transformer decoder for Canary MLX."""

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from canary_mlx.attention import FixedPositionalEncoding, MultiHeadAttention, create_causal_mask
from canary_mlx.cache import DecoderCache
from canary_mlx.encoder import FeedForward


@dataclass
class DecoderConfig:
    """Configuration for the transformer decoder."""
    vocab_size: int
    hidden_size: int
    inner_size: int
    num_layers: int
    num_attention_heads: int
    pre_ln: bool
    hidden_act: Literal["relu"]
    pre_ln_final_layer_norm: bool
    learn_positional_encodings: bool
    max_sequence_length: int


@dataclass
class HeadConfig:
    """Configuration for the classification head."""
    num_layers: int
    hidden_size: int
    num_classes: int


class DecoderBlock(nn.Module):
    """Single transformer decoder block."""
    
    def __init__(self, config: DecoderConfig):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.first_sub_layer = MultiHeadAttention(
            config.num_attention_heads,
            config.hidden_size,
        )
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.second_sub_layer = MultiHeadAttention(
            config.num_attention_heads,
            config.hidden_size,
        )
        self.layer_norm_3 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.third_sub_layer = FeedForward(
            config.hidden_size, config.inner_size, activation=config.hidden_act
        )

    def __call__(
        self,
        x: mx.array,
        xa: mx.array,
        mask_x: mx.array | None = None,
        mask_xa: mx.array | None = None,
        cache: DecoderCache | None = None,
    ) -> mx.array:
        x_norm = self.layer_norm_1(x)
        x = x + self.first_sub_layer(x_norm, x_norm, x_norm, mask=mask_x, cache=cache)

        x_norm = self.layer_norm_2(x)
        x = x + self.second_sub_layer(x_norm, xa, xa, mask=mask_xa)

        x_norm = self.layer_norm_3(x)
        x = x + self.third_sub_layer(x_norm)

        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder for sequence generation."""
    
    def __init__(self, config: DecoderConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = (
            nn.Embedding(config.max_sequence_length, config.hidden_size)
            if config.learn_positional_encodings
            else FixedPositionalEncoding(
                config.hidden_size, max_len=config.max_sequence_length
            )
        )
        self.embedding_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)

        self.layers = [DecoderBlock(config) for _ in range(config.num_layers)]
        self.final_layer_norm = (
            nn.LayerNorm(config.hidden_size, eps=1e-5)
            if config.pre_ln and config.pre_ln_final_layer_norm
            else None
        )

    def __call__(
        self,
        x: mx.array,
        xa: mx.array,
        mask_x: mx.array | None = None,
        mask_xa: mx.array | None = None,
        cache: list[DecoderCache] | None = None,
    ) -> mx.array:
        offset = 0 if cache is None else cache[0].offset

        x = self.token_embedding(x)
        x = x + (
            self.position_embedding(x, offset=offset)
            if isinstance(self.position_embedding, FixedPositionalEncoding)
            else self.position_embedding(mx.arange(offset, offset + x.shape[1]))
        )
        x = self.embedding_layer_norm(x)

        mask_x = (
            mask_x & create_causal_mask(x.shape[1], offset)
            if mask_x is not None
            else create_causal_mask(x.shape[1], offset)
        )
        for i, layer in enumerate(self.layers):
            x = layer(
                x, xa, mask_x, mask_xa, cache=cache[i] if cache is not None else None
            )

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        return x


class ClassificationHead(nn.Module):
    """Linear classification head for token prediction."""
    
    def __init__(self, config: HeadConfig):
        super().__init__()

        if config.num_layers != 1:
            raise NotImplementedError(
                "Classification head with multiple layers not implemented, should be 1"
            )

        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        return self.classifier(x)


"""Conformer encoder for Canary MLX."""

import math
from dataclasses import dataclass
from typing import Literal, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import tree_flatten

from canary_mlx.attention import (
    LocalRelPositionalEncoding,
    MultiHeadAttention,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadLocalAttention,
)


@dataclass
class EncoderConfig:
    """Configuration for the Conformer encoder."""
    feat_in: int
    n_layers: int
    d_model: int
    n_heads: int
    ff_expansion_factor: int
    subsampling_factor: int
    self_attention_model: str
    subsampling: str
    conv_kernel_size: int
    subsampling_conv_channels: int
    pos_emb_max_len: int
    causal_downsampling: bool = False
    use_bias: bool = True
    xscaling: bool = False
    pos_bias_u: Optional[mx.array] = None
    pos_bias_v: Optional[mx.array] = None
    subsampling_conv_chunking_factor: int = 1
    att_context_size: Optional[list[int]] = None


class FeedForward(nn.Module):
    """Feed-forward module with configurable activation."""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        use_bias: bool = True,
        activation: Literal["relu", "silu"] = "silu",
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = nn.SiLU() if activation == "silu" else nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(self.activation(self.linear1(x)))


class Convolution(nn.Module):
    """Convolution module for Conformer."""
    
    def __init__(self, config: EncoderConfig):
        assert (config.conv_kernel_size - 1) % 2 == 0
        super().__init__()

        self.padding = (config.conv_kernel_size - 1) // 2

        self.pointwise_conv1 = nn.Conv1d(
            config.d_model,
            config.d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=config.use_bias,
        )
        self.depthwise_conv = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=config.conv_kernel_size,
            stride=1,
            padding=0,
            groups=config.d_model,
            bias=config.use_bias,
        )
        self.batch_norm = nn.BatchNorm(config.d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=config.use_bias,
        )

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        x = self.pointwise_conv1(x)
        x = nn.glu(x, axis=2)

        if cache is not None:
            x = cache.update_and_fetch_conv(x, padding=self.padding)
        else:
            x = mx.pad(x, ((0, 0), (self.padding, self.padding), (0, 0)))
        x = self.depthwise_conv(x)

        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        return x


class ConformerBlock(nn.Module):
    """Single Conformer block."""
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        ff_hidden_dim = config.d_model * config.ff_expansion_factor

        self.config = config

        self.norm_feed_forward1 = nn.LayerNorm(config.d_model)
        self.feed_forward1 = FeedForward(config.d_model, ff_hidden_dim, config.use_bias)

        self.norm_self_att = nn.LayerNorm(config.d_model)
        self.self_attn = (
            RelPositionMultiHeadAttention(
                config.n_heads,
                config.d_model,
                bias=config.use_bias,
                pos_bias_u=config.pos_bias_u,
                pos_bias_v=config.pos_bias_v,
            )
            if config.self_attention_model == "rel_pos"
            else RelPositionMultiHeadLocalAttention(
                config.n_heads,
                config.d_model,
                bias=config.use_bias,
                pos_bias_u=config.pos_bias_u,
                pos_bias_v=config.pos_bias_v,
                context_size=(config.att_context_size[0], config.att_context_size[1])
                if config.att_context_size is not None
                else (-1, -1),
            )
            if config.self_attention_model == "rel_pos_local_attn"
            else MultiHeadAttention(
                config.n_heads,
                config.d_model,
                bias=True,
            )
        )

        self.norm_conv = nn.LayerNorm(config.d_model)
        self.conv = Convolution(config)

        self.norm_feed_forward2 = nn.LayerNorm(config.d_model)
        self.feed_forward2 = FeedForward(config.d_model, ff_hidden_dim, config.use_bias)

        self.norm_out = nn.LayerNorm(config.d_model)

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))

        x_norm = self.norm_self_att(x)
        x = x + self.self_attn(
            x_norm, x_norm, x_norm, mask=mask, pos_emb=pos_emb, cache=cache
        )

        x = x + self.conv(self.norm_conv(x), cache=cache)
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))

        return self.norm_out(x)


class Subsampling(nn.Module):
    """Depthwise striding subsampling for audio features."""
    
    def __init__(self, config: EncoderConfig):
        super().__init__()

        assert (
            config.subsampling_factor > 0
            and (config.subsampling_factor & (config.subsampling_factor - 1)) == 0
        )
        self.subsampling_conv_chunking_factor = config.subsampling_conv_chunking_factor
        self._conv_channels = config.subsampling_conv_channels
        self._sampling_num = int(math.log(config.subsampling_factor, 2))
        self._stride = 2
        self._kernel_size = 3
        self._padding = (self._kernel_size - 1) // 2

        in_channels = 1
        final_freq_dim = config.feat_in
        for _ in range(self._sampling_num):
            final_freq_dim = (
                math.floor(
                    (final_freq_dim + 2 * self._padding - self._kernel_size)
                    / self._stride
                )
                + 1
            )
            if final_freq_dim < 1:
                raise ValueError("Non-positive final frequency dimension!")

        self.conv = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self._conv_channels,
                kernel_size=self._kernel_size,
                stride=self._stride,
                padding=self._padding,
            ),
            nn.ReLU(),
        ]
        in_channels = self._conv_channels

        for _ in range(self._sampling_num - 1):
            self.conv.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._padding,
                    groups=in_channels,
                )
            )
            self.conv.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self._conv_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                )
            )
            self.conv.append(nn.ReLU())

        self.out = nn.Linear(self._conv_channels * final_freq_dim, config.d_model)

    def conv_forward(self, x: mx.array) -> mx.array:
        x = x.transpose((0, 2, 3, 1))
        for layer in self.conv:
            x = layer(x)
        return x.transpose((0, 3, 1, 2))

    def conv_split_by_batch(self, x: mx.array) -> tuple[mx.array, bool]:
        b = x.shape[0]
        if b == 1:
            return x, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
        else:
            x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(x.size / x_ceil, 2))
            cf: int = 2**p

        new_batch_size = b // cf
        if new_batch_size == 0:
            return x, False

        return mx.concat(
            [self.conv_forward(chunk) for chunk in mx.split(x, new_batch_size, 0)]
        ), True

    def __call__(self, x: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
        for _ in range(self._sampling_num):
            lengths = (
                mx.floor(
                    (lengths + 2 * self._padding - self._kernel_size) / self._stride
                )
                + 1.0
            )
        lengths = lengths.astype(mx.int32)

        x = mx.expand_dims(x, axis=1)

        if self.subsampling_conv_chunking_factor != -1:
            if self.subsampling_conv_chunking_factor == 1:
                x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
                need_to_split = x.size > x_ceil
            else:
                need_to_split = True

            if need_to_split:
                x, success = self.conv_split_by_batch(x)
                if not success:
                    x = self.conv_forward(x)
            else:
                x = self.conv_forward(x)
        else:
            x = self.conv_forward(x)

        x = x.swapaxes(1, 2).reshape(x.shape[0], x.shape[2], -1)
        x = self.out(x)
        return x, lengths


class ConformerEncoder(nn.Module):
    """Conformer encoder for audio feature extraction."""
    
    def __init__(self, config: EncoderConfig):
        super().__init__()

        self.config = config

        if config.self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=config.d_model,
                max_len=config.pos_emb_max_len,
                scale_input=config.xscaling,
            )
        elif config.self_attention_model == "rel_pos_local_attn":
            self.pos_enc = LocalRelPositionalEncoding(
                d_model=config.d_model,
                max_len=config.pos_emb_max_len,
                scale_input=config.xscaling,
                context_size=(config.att_context_size[0], config.att_context_size[1])
                if config.att_context_size is not None
                else (-1, -1),
            )
        else:
            self.pos_enc = None

        if config.subsampling_factor > 1:
            if config.subsampling == "dw_striding" and config.causal_downsampling is False:
                self.pre_encode = Subsampling(config)
            else:
                self.pre_encode = nn.Identity()
                raise NotImplementedError(
                    "Other subsampling methods not implemented yet!"
                )
        else:
            self.pre_encode = nn.Linear(config.feat_in, config.d_model)

        self.layers = [ConformerBlock(config) for _ in range(config.n_layers)]

    def __call__(
        self, x: mx.array, lengths: mx.array | None = None, cache=None
    ) -> tuple[mx.array, mx.array]:
        if lengths is None:
            lengths = mx.full(
                (x.shape[0],),
                x.shape[-2],
                dtype=mx.int64,
            )

        if isinstance(self.pre_encode, Subsampling):
            x, out_lengths = self.pre_encode(x, lengths)
        elif isinstance(self.pre_encode, nn.Linear):
            x = self.pre_encode(x)
            out_lengths = lengths
        else:
            raise NotImplementedError("Non-implemented pre-encoding layer type!")

        if cache is None:
            cache = [None] * len(self.layers)

        pos_emb = None
        if self.pos_enc is not None:
            x, pos_emb = self.pos_enc(
                x,
                offset=cache[0].offset if cache[0] is not None else 0,
            )

        for layer, c in zip(self.layers, cache):
            x = layer(x, pos_emb=pos_emb, cache=c)

        return x, out_lengths


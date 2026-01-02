"""Caching utilities for efficient inference."""

import mlx.core as mx


class KVCache:
    """Key-value cache for transformer attention."""
    keys: mx.array | None
    values: mx.array | None

    offset: int
    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.conv = None
        self.offset = 0

    def update_and_fetch_kv(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        prev = self.offset
        if (
            self.keys is None
            or self.values is None
            or (prev + keys.shape[2]) > self.keys.shape[2]
        ):
            B, H, S, D_KEYS = keys.shape
            _, _, _, D_VALUES = values.shape
            S_CACHE = ((self.step + S - 1) // self.step) * self.step

            new_k = mx.zeros((B, H, S_CACHE, D_KEYS), keys.dtype)
            new_v = mx.zeros((B, H, S_CACHE, D_VALUES), keys.dtype)

            if self.keys is None or self.values is None:
                self.keys, self.values = new_k, new_v
            else:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values

        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


class EncoderCache:
    """Cache for conformer encoder with convolution state."""
    keys: mx.array | None
    values: mx.array | None
    conv: mx.array | None

    offset: int
    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.conv = None
        self.offset = 0

    def update_and_fetch_kv(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        prev = self.offset
        if (
            self.keys is None
            or self.values is None
            or (prev + keys.shape[2]) > self.keys.shape[2]
        ):
            B, H, S, D_KEYS = keys.shape
            _, _, _, D_VALUES = values.shape
            S_CACHE = ((self.step + S - 1) // self.step) * self.step

            new_k = mx.zeros((B, H, S_CACHE, D_KEYS), keys.dtype)
            new_v = mx.zeros((B, H, S_CACHE, D_VALUES), keys.dtype)

            if self.keys is None or self.values is None:
                self.keys, self.values = new_k, new_v
            else:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values

        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def update_and_fetch_conv(self, x: mx.array, padding: int = 0) -> mx.array:
        if padding == 0:
            return x

        B, S, D = x.shape

        if self.conv is None:
            self.conv = mx.zeros((B, padding, D), x.dtype)

        tokens_to_cache = min(padding, S)

        cache_update = x[:, S - tokens_to_cache : S, :]

        if tokens_to_cache < padding:
            self.conv = mx.concatenate(
                [self.conv[:, tokens_to_cache:, :], cache_update], axis=1
            )
        else:
            self.conv = cache_update

        result = mx.concatenate([self.conv, x], axis=1)
        result = mx.pad(result, ((0, 0), (0, padding)))

        return result


class DecoderCache:
    """Cache for transformer decoder."""
    keys: mx.array | None
    values: mx.array | None

    offset: int
    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.conv = None
        self.offset = 0

    def update_and_fetch_kv(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        prev = self.offset
        if (
            self.keys is None
            or self.values is None
            or (prev + keys.shape[2]) > self.keys.shape[2]
        ):
            B, H, S, D_KEYS = keys.shape
            _, _, _, D_VALUES = values.shape
            S_CACHE = ((self.step + S - 1) // self.step) * self.step

            new_k = mx.zeros((B, H, S_CACHE, D_KEYS), keys.dtype)
            new_v = mx.zeros((B, H, S_CACHE, D_VALUES), keys.dtype)

            if self.keys is None or self.values is None:
                self.keys, self.values = new_k, new_v
            else:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values

        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


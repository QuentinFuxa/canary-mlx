"""Canary MLX model implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, cast

import mlx.core as mx
import mlx.nn as nn
from typing_extensions import Literal

from canary_mlx.alignment import (
    AlignedToken,
    TranscriptionResult,
    merge_chunks,
    sentences_to_result,
    tokens_to_sentences,
)
from canary_mlx.audio import AudioConfig, compute_features, load_audio
from canary_mlx.cache import DecoderCache
from canary_mlx.decoder import ClassificationHead, DecoderConfig, HeadConfig, TransformerDecoder
from canary_mlx.encoder import ConformerEncoder, EncoderConfig
from canary_mlx.tokenizer import CanaryTokenizer


@dataclass
class CanaryConfig:
    """Configuration for the Canary model."""
    preprocessor: AudioConfig
    encoder: EncoderConfig
    transf_decoder: DecoderConfig
    head: HeadConfig
    prompt_format: Literal["canary", "canary2"]
    tokenizer: dict
    model_dir: Optional[Path] = None


@dataclass
class DecodingConfig:
    """Configuration for decoding/inference."""
    decoding: Literal["greedy", "beam"] = "beam"
    beam_size: int = 5
    temperature: float = 0.0
    max_length: int = 512


class Canary(nn.Module):
    """
    Canary speech recognition model for MLX.
    
    A encoder-decoder model that converts audio to text using
    a Conformer encoder and Transformer decoder.
    """

    def __init__(self, config: CanaryConfig):
        super().__init__()

        self.audio_config = config.preprocessor
        self.encoder_config = config.encoder
        self.prompt_format: Literal["canary", "canary2"] = config.prompt_format

        # Initialize tokenizer
        tokenizer_config = config.tokenizer
        if tokenizer_config.get("type") == "sentencepiece":
            model_path = tokenizer_config.get("model_path", "tokenizer.model")
            if config.model_dir is not None:
                model_path = config.model_dir / model_path
            self.tokenizer = CanaryTokenizer.from_file(model_path)
        else:
            raise ValueError(
                "Invalid tokenizer config. Expected 'type': 'sentencepiece' with 'model_path'."
            )

        self.encoder = ConformerEncoder(config.encoder)
        self.transf_decoder = TransformerDecoder(config.transf_decoder)
        self.head = ClassificationHead(config.head)

    def transcribe(
        self,
        path: Path | str,
        language: str = "en",
        timestamps: bool = False,
        punctuation: bool = True,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        chunk_duration: Optional[float] = None,
        overlap_duration: float = 15.0,
        chunk_callback: Optional[Callable] = None,
    ) -> TranscriptionResult | str:
        """
        Transcribe an audio file.
        
        Args:
            path: Path to the audio file.
            language: Language code (e.g., "en", "es", "fr").
            timestamps: Whether to include word-level timestamps.
            punctuation: Whether to include punctuation in output.
            dtype: Data type for inference.
            chunk_duration: If provided, process audio in chunks of this duration.
            overlap_duration: Overlap between chunks in seconds.
            chunk_callback: Callback function called after each chunk.
            
        Returns:
            TranscriptionResult with timing info, or just text if timestamps=False.
        """
        audio_path = Path(path)
        audio_data = load_audio(audio_path, self.audio_config.sample_rate, dtype)

        if chunk_duration is None:
            prompt_tokens = build_prompt(
                self.tokenizer,
                self.prompt_format,
                language,
                language,
                punctuation,
                timestamp=timestamps,
            )
            mel = compute_features(audio_data, self.audio_config)
            result = self.generate(mel, [prompt_tokens])[0]
            return result if timestamps else result.text

        audio_length_seconds = len(audio_data) / self.audio_config.sample_rate
        if audio_length_seconds <= chunk_duration:
            prompt_tokens = build_prompt(
                self.tokenizer,
                self.prompt_format,
                language,
                language,
                punctuation,
                timestamp=timestamps,
            )
            mel = compute_features(audio_data, self.audio_config)
            result = self.generate(mel, [prompt_tokens])[0]
            return result if timestamps else result.text

        chunk_samples = int(chunk_duration * self.audio_config.sample_rate)
        overlap_samples = int(overlap_duration * self.audio_config.sample_rate)
        all_tokens = []
        previous_text = ""

        for start in range(0, len(audio_data), chunk_samples - overlap_samples):
            end = min(start + chunk_samples, len(audio_data))

            if chunk_callback is not None:
                chunk_callback(end, len(audio_data))

            if end - start < self.audio_config.hop_length:
                break

            if self.prompt_format == "canary2" and previous_text:
                prompt_tokens = build_prompt(
                    self.tokenizer,
                    self.prompt_format,
                    language,
                    language,
                    punctuation,
                    context=previous_text,
                    timestamp=timestamps,
                )
            else:
                prompt_tokens = build_prompt(
                    self.tokenizer,
                    self.prompt_format,
                    language,
                    language,
                    punctuation,
                    timestamp=timestamps,
                )

            chunk_audio = audio_data[start:end]
            chunk_mel = compute_features(chunk_audio, self.audio_config)
            chunk_result = self.generate(chunk_mel, [prompt_tokens])[0]

            if chunk_result.text:
                previous_text = chunk_result.text

            chunk_offset = start / self.audio_config.sample_rate
            for sentence in chunk_result.sentences:
                for token in sentence.tokens:
                    token.start += chunk_offset
                    token.end = token.start + token.duration

            if all_tokens:
                all_tokens = merge_chunks(
                    all_tokens,
                    chunk_result.tokens,
                    overlap_duration=overlap_duration if timestamps else chunk_duration,
                )
            else:
                all_tokens = chunk_result.tokens

        result = sentences_to_result(tokens_to_sentences(all_tokens))
        return result if timestamps else result.text

    def generate(
        self,
        mel: mx.array,
        prompts: list[list[int]],
        *,
        decoding_config: DecodingConfig = DecodingConfig(),
    ) -> list[TranscriptionResult]:
        """Generate transcriptions from mel spectrograms."""
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)
        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)
        decoded = self._decode(features, prompts, lengths, config=decoding_config)

        def parse_time(token):
            try:
                s = self.tokenizer.decode([token])
                if s.startswith("<|") and s.endswith("|>"):
                    time_str = s[2:-2]
                    if time_str.isdigit():
                        return (
                            int(time_str)
                            * self.encoder_config.subsampling_factor
                            / self.audio_config.sample_rate
                            * self.audio_config.hop_length
                        )
            except (ValueError, AttributeError, KeyError):
                pass
            return None

        results = []
        for batch_idx, batch in enumerate(decoded):
            aligned = []
            batch_length = float(
                lengths[batch_idx] if batch_idx < len(lengths) else lengths[-1]
            )
            max_time = (
                batch_length
                * self.encoder_config.subsampling_factor
                / self.audio_config.sample_rate
                * self.audio_config.hop_length
            )

            timestamp_indices = {}
            for i, t in enumerate(batch):
                if t in self.tokenizer.special_tokens:
                    time = parse_time(t)
                    if time is not None:
                        timestamp_indices[i] = time

            for i, t in enumerate(batch):
                if t not in self.tokenizer.special_tokens:
                    start = None
                    for j in range(i - 1, -1, -1):
                        if j in timestamp_indices:
                            start = timestamp_indices[j]
                            break
                    if start is None:
                        start = 0.0

                    end = None
                    for j in range(i + 1, len(batch)):
                        if j in timestamp_indices:
                            end = timestamp_indices[j]
                            break
                    if end is None:
                        end = (
                            max_time
                            if i == len(batch) - 1 or not timestamp_indices
                            else start
                        )

                    aligned.append(
                        AlignedToken(
                            t,
                            self.tokenizer.decode([t], strip=False),
                            start,
                            max(0, end - start),
                        )
                    )

            results.append(sentences_to_result(tokens_to_sentences(aligned)))

        return results

    def _decode(
        self,
        features: mx.array,
        prompt: list[list[int]],
        lengths: Optional[mx.array] = None,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> list[list[int]]:
        """Internal decoding method."""
        if config.decoding == "greedy":
            outputs = []
            for batch, p in enumerate(prompt):
                tokens = []
                inputs = p.copy()
                cache = [
                    DecoderCache()
                    for _ in range(len(self.transf_decoder.layers))
                ]

                feat = features[batch : batch + 1]
                if lengths is not None:
                    feat = feat[:, : int(lengths[batch])]

                while len(tokens) + len(p) < config.max_length:
                    logits = self.head(
                        self.transf_decoder(mx.array([inputs]), feat, cache=cache)
                    )
                    next_token = cast(int, mx.argmax(logits[:, -1], axis=-1).item())

                    if next_token == self.tokenizer.eos_id:
                        break

                    inputs = [next_token]
                    tokens.append(next_token)

                outputs.append(tokens)
            return outputs
            
        elif config.decoding == "beam":
            outputs = []

            for batch, p in enumerate(prompt):
                beams = [([], p.copy(), 0)]
                cache = [
                    DecoderCache()
                    for _ in range(len(self.transf_decoder.layers))
                ]

                feat = features[batch : batch + 1]
                if lengths is not None:
                    feat = feat[:, : int(lengths[batch])]

                for _ in range(config.max_length - len(p)):
                    logits = self.head(
                        self.transf_decoder(
                            mx.array([beam[1] for beam in beams]),
                            mx.repeat(feat, len(beams), 0),
                            cache=cache,
                        )
                    )
                    logprobs = nn.log_softmax(
                        logits[:, -1] / max(config.temperature, 1e-8)
                    )
                    accumulated_logprobs = logprobs.flatten() + mx.array(
                        [beam[2] for beam in beams for _ in range(logprobs.shape[1])]
                    )

                    indices = mx.argpartition(accumulated_logprobs, -config.beam_size)[
                        -config.beam_size :
                    ]
                    beam_indices = indices // logprobs.shape[1]
                    token_indices = indices % logprobs.shape[1]

                    for c in cache:
                        if c.keys is not None and c.values is not None:
                            c.keys = c.keys[beam_indices]
                            c.values = c.values[beam_indices]
                    beams = [
                        (
                            beams[int(beam_indices[i])][0] + [int(token_indices[i])],
                            [int(token_indices[i])],
                            float(accumulated_logprobs[indices[i]]),
                        )
                        if beams[int(beam_indices[i])][1][0] != self.tokenizer.eos_id
                        else (
                            beams[int(beam_indices[i])][0],
                            [self.tokenizer.eos_id],
                            beams[int(beam_indices[i])][2],
                        )
                        for i in range(config.beam_size)
                    ]

                    if all(beam[1][0] == self.tokenizer.eos_id for beam in beams):
                        beams = list(sorted(beams, key=lambda x: x[2], reverse=True))
                        outputs.append(beams[0][0][:-1])
                        break

                if len(outputs) < batch + 1:
                    beams = list(sorted(beams, key=lambda x: x[2], reverse=True))
                    eos_beams = list(
                        filter(lambda x: x[1][0] == self.tokenizer.eos_id, beams)
                    )
                    if len(eos_beams) > 0:
                        outputs.append(eos_beams[0][0][:-1])
                    else:
                        outputs.append(beams[0][0])

            return outputs

        raise NotImplementedError(f"Decoding method '{config.decoding}' not implemented")


def build_prompt(
    tokenizer: CanaryTokenizer,
    prompt_format: Literal["canary", "canary2"],
    source_lang: str,
    target_lang: str,
    punctuation: bool,
    *,
    context: str = "",
    emotion: Literal["undefined", "neutral", "angry", "happy", "sad"] = "undefined",
    inverse_normalization: bool = False,
    timestamp: bool = False,
    diarize: bool = False,
) -> list[int]:
    """
    Build prompt tokens for the model.
    
    Args:
        tokenizer: The tokenizer to use.
        prompt_format: Either "canary" or "canary2".
        source_lang: Source language code.
        target_lang: Target language code.
        punctuation: Whether to include punctuation.
        context: Previous context for continuation.
        emotion: Emotion tag (canary2 only).
        inverse_normalization: Whether to apply ITN (canary2 only).
        timestamp: Whether to include timestamps (canary2 only).
        diarize: Whether to include diarization (canary2 only).
        
    Returns:
        List of prompt token IDs.
    """
    if prompt_format == "canary" and (
        len(context) > 0
        or emotion != "undefined"
        or inverse_normalization is True
        or timestamp is True
        or diarize is True
    ):
        raise ValueError(
            "`context`, `emotion`, `inverse_normalization`, `timestamp`, `diarize` "
            "are only supported in `canary2` prompt format."
        )

    src, tgt = f"<|{source_lang}|>", f"<|{target_lang}|>"
    pnc = "<|pnc|>" if punctuation else "<|nopnc|>"

    if prompt_format == "canary":
        task = "<|transcribe|>" if source_lang == target_lang else "<|translate|>"
        prompt_text = f"<|startoftranscript|>{src}{task}{tgt}{pnc}"
        return tokenizer.encode(prompt_text, lang_id="spl_tokens")

    emo = f"<|emo:{emotion}|>"
    itn = "<|itn|>" if inverse_normalization else "<|noitn|>"
    ts = "<|timestamp|>" if timestamp else "<|notimestamp|>"
    dia = "<|diarize|>" if diarize else "<|nodiarize|>"

    if context:
        ctx_tokens = tokenizer.encode(context, lang_id=target_lang)
        prompt_tokens = tokenizer.encode(
            f"<|startofcontext|><|startoftranscript|>{emo}{src}{tgt}{pnc}{itn}{ts}{dia}",
            lang_id="spl_tokens",
        )
        return prompt_tokens[:1] + ctx_tokens + prompt_tokens[1:]

    prompt_text = (
        f"<|startofcontext|><|startoftranscript|>{emo}{src}{tgt}{pnc}{itn}{ts}{dia}"
    )
    return tokenizer.encode(prompt_text, lang_id="spl_tokens")


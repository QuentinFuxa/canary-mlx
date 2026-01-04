# Canary MLX

NVIDIA Canary ASR 1b V2 model optimized for Apple Silicon using MLX.

## Installation

```bash
pip install canary-mlx
```

## Usage

### Basic Transcription

```python
from canary_mlx import load_model

model = load_model("qfuxa/canary-mlx") #You can also use a local model directory

result = model.transcribe("audio.wav", language="en")
print(result)
```


### With Timestamps

```python
result = model.transcribe("audio.wav", language="en", timestamps=True)

for sentence in result.sentences:
    print(f"[{sentence.start:.2f}s - {sentence.end:.2f}s] {sentence.text}")
```

### Long Audio with Chunking

```python
result = model.transcribe(
    "long_audio.wav",
    language="en",
    timestamps=True,
    chunk_duration=30.0,  # Process in 30-second chunks
    overlap_duration=15.0,  # 15-second overlap between chunks
)
```

### Translation

Translate audio from one language to another (speech-to-text translation):

```python
result = model.translate(
    "french_audio.wav",
    source_language="fr",
    target_language="en"
)
print(result)
```

## Supported Languages

Bulgarian (**bg**), Croatian (**hr**), Czech (**cs**), Danish (**da**), Dutch (**nl**), English (**en**), Estonian (**et**), Finnish (**fi**), French (**fr**), German (**de**), Greek (**el**), Hungarian (**hu**), Italian (**it**), Latvian (**lv**), Lithuanian (**lt**), Maltese (**mt**), Polish (**pl**), Portuguese (**pt**), Romanian (**ro**), Slovak (**sk**), Slovenian (**sl**), Spanish (**es**), Swedish (**sv**), Russian (**ru**), Ukrainian (**uk**)

## Benchmarks

### STT FLEURS WER (lower is better)

| Model | bg | cs | da | de | el | en | es | et | fi | fr | hr | hu | it | lt | lv | mt | nl | pl | pt | ro | ru | sk | sl | sv | uk |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| whisper-large-v3 | 12.86 | 11.33 | 12.57 | 4.30 | 27.03 | 4.25 | 3.12 | 19.12 | 7.70 | 6.31 | 11.07 | 14.11 | 2.31 | 22.34 | 18.29 | 68.89 | 5.57 | 4.74 | 3.65 | 8.24 | 4.17 | 8.40 | 19.93 | 7.88 | 6.51 |
| **Canary-1B-v2** | **9.25** | **7.86** | **11.25** | 4.40 | **9.21** | 4.50 | **2.90** | **12.55** | 8.59 | **5.02** | **8.29** | **12.90** | 3.07 | **12.36** | **9.66** | **18.31** | 6.12 | 6.64 | 4.39 | **6.61** | 6.90 | **5.74** | **13.32** | 9.57 | 10.50 |

For more detailed metrics, especially regarding translation, please refer to the appendix of the [Canary-1B v2 technical report](https://arxiv.org/pdf/2509.14128).

## Performance (Speed)

### Inference Speed on Apple M4

The following benchmarks were measured on an **Apple M4** processing a 10-minute audio file.

#### Canary-1B-v2 (MLX)

| Chunk Duration | Time Taken | RAM Consumption | Notes |
| :--- | :--- | :--- | :--- |
| 30s | 110.2s | ~4.7GB (start) to 5.25GB (end) | Recommended duration |
| 60s | 99.0s | - | |
| 120s | 90.5s | - | **Fails at the end** (hallucinations/loops) |

> [!IMPORTANT]
> NVIDIA recommends using a **chunk duration of less than 40 seconds** for Canary models to avoid transcription failures/hallucinations at the end of chunks.

#### Comparison: Whisper Large v3 (MLX)

| Model | Time Taken | RAM Consumption |
| :--- | :--- | :--- |
| Whisper Large v3 | 77.5s | 6.3GB (start) to 10.2GB (end) |



## API Reference

### `load_model(path_or_hf_id, dtype=mx.bfloat16)`

Load a Canary model from a local directory or HuggingFace Hub.

**Parameters:**
- `path_or_hf_id`: Local path or HuggingFace model ID (e.g., `"qfuxa/canary-mlx"`)
- `dtype`: Data type for model weights (default: `mx.bfloat16`)

**Returns:** `Canary` model instance

### `model.transcribe(...)`

Transcribe an audio file (same language in/out).

**Parameters:**
- `path`: Path to audio file
- `language`: Language code (e.g., "en")
- `timestamps`: Include word-level timestamps (default: `False`)
- `punctuation`: Include punctuation (default: `True`)
- `chunk_duration`: Process in chunks of this duration (optional)
- `overlap_duration`: Overlap between chunks in seconds (default: `15.0`)

**Returns:** `TranscriptionResult` if timestamps=True, else `str`

### `model.translate(...)`

Translate audio from one language to another.

**Parameters:**
- `path`: Path to audio file
- `source_language`: Language of the audio (e.g., "fr", "de")
- `target_language`: Target language for translation (default: "en")
- `timestamps`: Include word-level timestamps (default: `False`)
- `punctuation`: Include punctuation (default: `True`)
- `chunk_duration`: Process in chunks of this duration (optional)
- `overlap_duration`: Overlap between chunks in seconds (default: `15.0`)

**Returns:** `TranscriptionResult` if timestamps=True, else `str`

## Acknowledgements
- **Nvidia** for the impressive model
- **MLX project and community**
- **Senstella for Parakeet MLX** that has been a great help for the FastConformer mlx implementation
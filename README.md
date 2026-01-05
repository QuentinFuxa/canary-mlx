<h1 align="center">Canary MLX</h1>
<p align="center"><b>NVIDIA Canary ASR model optimized for Apple Silicon using MLX.</b></p>


<p align="center">
<a href="https://pypi.org/project/canary-mlx/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/canary-mlx?color=g"></a>
<a href="https://pepy.tech/project/canary-mlx"><img alt="PyPI Downloads" src="https://static.pepy.tech/personalized-badge/canary-mlx?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=installations"></a>
<a href="https://pypi.org/project/canary-mlx/"><img alt="Python Versions" src="https://img.shields.io/badge/python-3.9--3.15-dark_green"></a>
<a href="https://huggingface.co/qfuxa/canary-mlx">
  <img alt="Hugging Face Weights" src="https://img.shields.io/badge/🤗-Hugging%20Face%20Weights-yellow" />
</a>
<a href="https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-dark_green"></a>
</p>


## Installation

```bash
pip install canary-mlx
```

## Usage

### Basic Transcription

```python
from canary_mlx import load_model

model = load_model("qfuxa/canary-mlx")

# Transcribe an audio file
result = model.transcribe("audio.wav", language="en")
print(result)
```

### Load from Local Directory

```python
from canary_mlx import load_model

# Load from a local model directory
model = load_model("./path/to/model")
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
# Translate French audio to English
result = model.translate(
    "french_audio.wav",
    source_language="fr",
    target_language="en"
)
print(result)  # English text

# Translate German audio to Spanish
result = model.translate(
    "german_audio.wav",
    source_language="de",
    target_language="es"
)
```

## Supported Languages

Bulgarian (**bg**), Croatian (**hr**), Czech (**cs**), Danish (**da**), Dutch (**nl**), English (**en**), Estonian (**et**), Finnish (**fi**), French (**fr**), German (**de**), Greek (**el**), Hungarian (**hu**), Italian (**it**), Latvian (**lv**), Lithuanian (**lt**), Maltese (**mt**), Polish (**pl**), Portuguese (**pt**), Romanian (**ro**), Slovak (**sk**), Slovenian (**sl**), Spanish (**es**), Swedish (**sv**), Russian (**ru**), Ukrainian (**uk**)

## API Reference

### `load_model(path_or_hf_id, dtype=mx.bfloat16)`

Load a Canary model from a local directory or HuggingFace Hub.

**Parameters:**
- `path_or_hf_id`: Local path or HuggingFace model ID (e.g., `"your-username/canary-mlx"`)
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

## Model Conversion

To convert a NeMo Canary model to MLX format:

1. Install conversion dependencies:
   ```bash
   pip install "canary-mlx[convert]"
   ```

2. Download and extract the NeMo model:
   ```bash
   tar -xvf canary_model.nemo -C canary_untared
   ```

3. Run the conversion script:
   ```bash
   python convert_nemo.py
   ```

This creates a `model/` directory containing:
- `model.safetensors` - Model weights
- `config.json` - Model configuration  
- `tokenizer.model` - SentencePiece tokenizer

## Acknowledgements
- **Nvidia** for the model!
- **MLX project and community**
- **Senstella for Parakeet MLX** that has been a great help for the FastConformer mlx implementation!

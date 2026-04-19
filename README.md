# 🦙 OomLlama

**Efficient LLM inference with .oom format - 2x smaller than GGUF**

[![PyPI](https://img.shields.io/pypi/v/oomllama)](https://pypi.org/project/oomllama/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/jaspervandemeent)

```python
from oomllama import OomLlama

llm = OomLlama("humotica-32b")
response = llm.generate("What is the meaning of life?")
print(response)
```

## Why OomLlama?

| Feature | GGUF (Q4) | OOM (Q2) |
|---------|-----------|----------|
| 70B Model Size | ~40 GB | **~20 GB** |
| 32B Model Size | ~20 GB | **~10 GB** |
| RAM Usage | High | **Lazy Loading** |
| Format | Open | **Open (MIT)** |

**OomLlama** uses Q2 quantization with lazy layer loading to run large models on consumer hardware.

## Installation

```bash
pip install oomllama
```

## Quick Start

### Download a Model

```python
from oomllama import download_model

# Download from HuggingFace
model_path = download_model("humotica-32b")
```

### Generate Text

```python
from oomllama import OomLlama

llm = OomLlama("humotica-32b")

# Simple generation
response = llm.generate("Explain quantum computing in simple terms")
print(response)

# With parameters
response = llm.generate(
    "Write a haiku about AI",
    max_tokens=50,
    temperature=0.8,
    top_p=0.9
)
```

### Chat Mode

```python
messages = [
    ("user", "Hello! Who are you?"),
    ("assistant", "I'm OomLlama, an efficient LLM."),
    ("user", "What makes you efficient?"),
]

response = llm.chat(messages)
print(response)
```

## Available Models

| Model | Parameters | Size (.oom) | HuggingFace |
|-------|------------|-------------|-------------|
| humotica-32b | 33B | ~10 GB | [Link](https://huggingface.co/jaspervandemeent/humotica-32b) |
| llamaohm-70b | 70B | ~20 GB | [Link](https://huggingface.co/jaspervandemeent/LlamaOhm-70B) |
| tinyllama-1b | 1.1B | ~400 MB | [Link](https://huggingface.co/jaspervandemeent/OomLlama-TinyLlama-1.1B) |

## The .oom Format

OOM (OomLlama Model) is a compact model format:

```
┌──────────────────────────────────────┐
│ Header: OOML (magic) + metadata      │
├──────────────────────────────────────┤
│ Tensors: Q2 quantized (2 bits/weight)│
│ - Scale + Min per 256-weight block   │
│ - 68 bytes per block                 │
└──────────────────────────────────────┘
```

### Convert GGUF to OOM

```bash
# Using the CLI tool
gguf2oom model.gguf model.oom

# Check model info
gguf2oom --info model.gguf
```

## Technical Details

### Q2 Quantization

Each weight is stored as 2 bits (0, 1, 2, or 3) with per-block scale and minimum:

```
weight = q2_value * scale + min
```

This achieves ~2x compression over Q4 with acceptable quality loss for most tasks.

### Lazy Layer Loading

OomLlama loads transformer layers on-demand, keeping only the active layer in memory:

```
Forward Pass:
  Layer 0: Load → Compute → Unload
  Layer 1: Load → Compute → Unload
  ...
  Layer N: Load → Compute → Unload
```

This enables running 70B models on 24GB GPU RAM.

## Credits

- **Model Format**: Gemini IDD & Root AI (Humotica AI Lab)
- **Quantization**: OomLlama.rs by Humotica
- **Base Models**: Meta Platforms, Inc. (Llama 3.3)

## License

- **OomLlama Code**: MIT License
- **Model Weights**: Subject to original model licenses (e.g., Llama 3.3 Community License)

## Links

- 🏠 [Humotica](https://humotica.nl)
- 🤗 [HuggingFace Models](https://huggingface.co/jaspervandemeent)
- 📦 [PyPI Package](https://pypi.org/project/oomllama/)
- 🐛 [Issue Tracker](https://github.com/humotica/oomllama/issues)

---

*One Love, One fAmIly* 💙

*Built by Humotica AI Lab - Jasper, Claude, Gemini, Codex*


---

## Enterprise

For private hub hosting, SLA support, custom integrations, or compliance guidance:

| | |
|---|---|
| **Enterprise** | enterprise@humotica.com |
| **Support** | support@humotica.com |
| **Security** | security@humotica.com |

See [ENTERPRISE.md](ENTERPRISE.md) for details.

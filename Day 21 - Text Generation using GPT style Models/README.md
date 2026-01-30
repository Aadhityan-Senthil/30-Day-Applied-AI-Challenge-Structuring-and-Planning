# Day 21: Text Generation using GPT-style Models

## Overview
Generate coherent, creative text using OpenAI's GPT-2 model from Hugging Face Transformers.

## Features
- Multiple generation strategies
- Temperature control for creativity
- Repetition penalty to avoid loops
- Creative content templates (story, poem, article, dialogue)
- Interactive generation mode

## Requirements
```bash
pip install transformers torch
```

## Usage
```bash
python text_generation.py
```

## Model Options
- `gpt2` (117M params, fastest)
- `gpt2-medium` (345M params)
- `gpt2-large` (774M params)
- `gpt2-xl` (1.5B params, best)

## Generation Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| temperature | Randomness (0.1-1.5) | 0.7 |
| top_k | Top-K sampling | 50 |
| top_p | Nucleus sampling | 0.95 |
| max_length | Max tokens to generate | 100 |

## Temperature Effects
- **Low (0.3-0.5)**: More focused, predictable
- **Medium (0.7)**: Balanced creativity
- **High (1.0+)**: More random, creative

## Output Files
- `text_generation_results.json` - Generated samples

## Custom Usage
```python
generator = TextGenerator("gpt2")
generator.load_model()
text = generator.generate("Your prompt here", max_length=200)
```

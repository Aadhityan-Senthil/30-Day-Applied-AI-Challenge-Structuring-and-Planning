# Day 17: Text Summarization using Transformers

## Overview
Automatic text summarization using both extractive and abstractive approaches with pre-trained transformer models (BART/T5).

## Summarization Types

### Extractive
- Selects important sentences from original text
- Uses TF-IDF-like scoring
- Maintains original wording

### Abstractive
- Generates new text that captures meaning
- Uses transformer models (BART, T5)
- More human-like summaries

## Requirements
```bash
pip install transformers torch
```

## Usage
```bash
python text_summarization.py
```

## Models Supported
- `facebook/bart-large-cnn` (default, best quality)
- `sshleifer/distilbart-cnn-12-6` (faster)
- `t5-small` (smallest, fallback)

## Output Files
- `summarization_results.json` - All summaries with statistics

## Custom Usage
```python
from transformers import pipeline
summarizer = pipeline("summarization")
result = summarizer("Your long text here...", max_length=130)
print(result[0]['summary_text'])
```

## Compression Ratios
Typical summaries are 10-30% of original length while preserving key information.

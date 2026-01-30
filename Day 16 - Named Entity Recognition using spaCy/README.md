# Day 16: Named Entity Recognition using spaCy

## Overview
Extract and classify named entities from text including people, organizations, locations, dates, money, and more using spaCy's pre-trained NER models.

## Entity Types Detected
- **PERSON**: People names
- **ORG**: Organizations, companies
- **GPE**: Countries, cities, states
- **DATE**: Dates and periods
- **MONEY**: Monetary values
- **LOC**: Non-GPE locations
- **PRODUCT**: Products
- **EVENT**: Named events
- And more...

## Requirements
```bash
pip install spacy matplotlib
python -m spacy download en_core_web_sm
```

## Usage
```bash
python named_entity_recognition.py
```

## Output Files
- `entity_summary.json` - Extracted entities by type
- `ner_statistics.png` - Frequency and distribution charts
- `entities_visualization.html` - Interactive entity highlighting

## Use Your Own Text
```python
nlp = spacy.load("en_core_web_sm")
doc = nlp("Your text here...")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

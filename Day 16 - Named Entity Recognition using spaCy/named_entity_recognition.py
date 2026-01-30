"""
Day 16: Named Entity Recognition using spaCy
30-Day AI Challenge

Extract named entities (people, organizations, locations, etc.) from text
using spaCy's pre-trained NER models.
"""

import spacy
from collections import Counter
import matplotlib.pyplot as plt
import json

# Sample texts for NER demonstration
SAMPLE_TEXTS = [
    """Apple Inc. announced today that CEO Tim Cook will visit their new headquarters 
    in Cupertino, California next Monday. The tech giant reported $89.5 billion 
    in revenue for Q3 2024.""",
    
    """Elon Musk's SpaceX successfully launched the Falcon 9 rocket from Kennedy 
    Space Center in Florida. NASA administrator Bill Nelson congratulated the team 
    on Twitter.""",
    
    """The United Nations held a climate summit in Paris, France where President 
    Joe Biden and Prime Minister Rishi Sunak discussed environmental policies. 
    The European Union pledged €500 million for renewable energy.""",
    
    """Google's parent company Alphabet reported strong earnings. Sundar Pichai 
    mentioned that YouTube reached 2 billion monthly users. The stock rose 5% 
    on the New York Stock Exchange.""",
    
    """Microsoft acquired Activision Blizzard for $68.7 billion, the largest 
    gaming deal in history. Satya Nadella said the acquisition will help 
    Xbox compete with Sony's PlayStation.""",
]

# Entity color mapping for visualization
ENTITY_COLORS = {
    'PERSON': '#FF6B6B',
    'ORG': '#4ECDC4',
    'GPE': '#45B7D1',
    'DATE': '#96CEB4',
    'MONEY': '#FFEAA7',
    'TIME': '#DDA0DD',
    'PERCENT': '#98D8C8',
    'PRODUCT': '#F7DC6F',
    'EVENT': '#BB8FCE',
    'LOC': '#85C1E9',
    'FAC': '#F8B500',
    'NORP': '#E59866',
    'CARDINAL': '#AED6F1',
    'ORDINAL': '#D7BDE2',
}

def load_spacy_model():
    """Load spaCy model, download if not available."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    return nlp

def extract_entities(nlp, text):
    """Extract named entities from text."""
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char,
            'description': spacy.explain(ent.label_)
        })
    
    return entities, doc

def analyze_multiple_texts(nlp, texts):
    """Analyze multiple texts and aggregate entity statistics."""
    all_entities = []
    entity_counts = Counter()
    entity_type_counts = Counter()
    
    for text in texts:
        entities, _ = extract_entities(nlp, text)
        all_entities.extend(entities)
        
        for ent in entities:
            entity_counts[ent['text']] += 1
            entity_type_counts[ent['label']] += 1
    
    return all_entities, entity_counts, entity_type_counts

def create_entity_html(doc, output_file='entities_visualization.html'):
    """Create HTML visualization of entities."""
    from spacy import displacy
    
    html = displacy.render(doc, style="ent", page=True)
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"HTML visualization saved to '{output_file}'")

def plot_entity_statistics(entity_counts, entity_type_counts):
    """Plot entity statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top entities by frequency
    top_entities = entity_counts.most_common(10)
    if top_entities:
        names, counts = zip(*top_entities)
        colors = ['#4ECDC4'] * len(names)
        axes[0].barh(range(len(names)), counts, color=colors)
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels(names)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Frequency')
        axes[0].set_title('Top 10 Named Entities')
    
    # Entity types distribution
    if entity_type_counts:
        types, type_counts = zip(*entity_type_counts.most_common())
        colors = [ENTITY_COLORS.get(t, '#95A5A6') for t in types]
        axes[1].pie(type_counts, labels=types, colors=colors, autopct='%1.1f%%')
        axes[1].set_title('Entity Type Distribution')
    
    plt.tight_layout()
    plt.savefig('ner_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Statistics saved to 'ner_statistics.png'")

def create_entity_summary(all_entities):
    """Create a summary of all extracted entities."""
    summary = {}
    
    for ent in all_entities:
        label = ent['label']
        if label not in summary:
            summary[label] = {
                'description': ent['description'],
                'entities': set()
            }
        summary[label]['entities'].add(ent['text'])
    
    # Convert sets to lists for JSON serialization
    for label in summary:
        summary[label]['entities'] = list(summary[label]['entities'])
    
    return summary

def process_custom_text(nlp, text):
    """Process custom text and display results."""
    entities, doc = extract_entities(nlp, text)
    
    print("\n" + "=" * 60)
    print("EXTRACTED ENTITIES:")
    print("=" * 60)
    
    if not entities:
        print("No entities found.")
        return
    
    # Group by entity type
    by_type = {}
    for ent in entities:
        label = ent['label']
        if label not in by_type:
            by_type[label] = []
        by_type[label].append(ent['text'])
    
    for label, texts in by_type.items():
        desc = spacy.explain(label)
        print(f"\n{label} ({desc}):")
        for text in set(texts):
            print(f"  • {text}")
    
    return entities, doc

def main():
    print("=" * 50)
    print("Day 16: Named Entity Recognition using spaCy")
    print("=" * 50)
    
    # Load model
    print("\n[1] Loading spaCy model...")
    nlp = load_spacy_model()
    print(f"Loaded model: {nlp.meta['name']}")
    
    # Process sample texts
    print("\n[2] Processing sample texts...")
    all_entities, entity_counts, entity_type_counts = analyze_multiple_texts(nlp, SAMPLE_TEXTS)
    
    print(f"\nTotal entities found: {len(all_entities)}")
    print(f"Unique entities: {len(entity_counts)}")
    print(f"Entity types: {len(entity_type_counts)}")
    
    # Create summary
    print("\n[3] Creating entity summary...")
    summary = create_entity_summary(all_entities)
    
    for label, data in summary.items():
        print(f"\n{label} ({data['description']}): {len(data['entities'])} unique")
        print(f"  Examples: {', '.join(list(data['entities'])[:5])}")
    
    # Save summary to JSON
    with open('entity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nEntity summary saved to 'entity_summary.json'")
    
    # Generate visualizations
    print("\n[4] Generating visualizations...")
    plot_entity_statistics(entity_counts, entity_type_counts)
    
    # Create HTML visualization for first text
    _, doc = extract_entities(nlp, SAMPLE_TEXTS[0])
    create_entity_html(doc)
    
    # Demo: Process custom text
    print("\n[5] Demo: Custom text processing...")
    custom_text = """
    Amazon founder Jeff Bezos announced that Blue Origin will launch 
    astronauts to the International Space Station in 2025. The mission, 
    valued at $3.4 billion, will depart from Cape Canaveral, Florida.
    """
    process_custom_text(nlp, custom_text)
    
    print("\n" + "=" * 50)
    print("Day 16 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()

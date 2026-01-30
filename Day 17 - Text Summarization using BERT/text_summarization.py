"""
Day 17: Text Summarization using Transformers
30-Day AI Challenge

Automatic text summarization using pre-trained transformer models
(BART/T5) from Hugging Face.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import textwrap
import json

# Sample articles for summarization
SAMPLE_ARTICLES = [
    {
        "title": "AI Revolution in Healthcare",
        "text": """
        Artificial intelligence is transforming the healthcare industry in unprecedented ways. 
        Machine learning algorithms are now capable of analyzing medical images with accuracy 
        that rivals or exceeds human radiologists. Deep learning models can detect early signs 
        of diseases like cancer, diabetic retinopathy, and cardiovascular conditions from 
        routine scans. Beyond diagnostics, AI is revolutionizing drug discovery by predicting 
        molecular interactions and identifying potential therapeutic compounds in a fraction 
        of the time traditional methods require. Pharmaceutical companies are increasingly 
        partnering with AI startups to accelerate their research pipelines. In clinical 
        settings, natural language processing systems are being deployed to analyze electronic 
        health records, helping physicians identify patterns and make more informed treatment 
        decisions. Chatbots and virtual health assistants are improving patient engagement 
        and providing 24/7 support for basic health queries. However, the integration of AI 
        in healthcare also raises important ethical questions about data privacy, algorithmic 
        bias, and the changing role of medical professionals. Regulatory bodies worldwide 
        are working to establish frameworks that ensure AI systems meet rigorous safety and 
        efficacy standards while encouraging innovation.
        """
    },
    {
        "title": "Climate Change and Renewable Energy",
        "text": """
        The global transition to renewable energy is accelerating as countries race to meet 
        their climate commitments. Solar and wind power have become the cheapest sources of 
        new electricity generation in most parts of the world, driving unprecedented investment 
        in clean energy infrastructure. In 2023, renewable energy capacity additions broke 
        records, with solar alone accounting for over 70% of new installations. Battery storage 
        technology is advancing rapidly, addressing the intermittency challenges that have 
        historically limited renewable adoption. Electric vehicles are gaining market share 
        across all vehicle segments, with several countries announcing plans to phase out 
        internal combustion engines within the next decade. Major corporations are committing 
        to 100% renewable energy targets, creating additional demand for clean power. The 
        hydrogen economy is emerging as a potential solution for hard-to-decarbonize sectors 
        like heavy industry and long-distance transportation. Despite this progress, scientists 
        warn that current efforts are insufficient to limit global warming to 1.5 degrees 
        Celsius. The transition also presents challenges for communities dependent on fossil 
        fuel industries, necessitating comprehensive just transition policies. International 
        cooperation and significantly increased investment will be essential to achieve net-zero 
        emissions by mid-century.
        """
    },
    {
        "title": "The Future of Remote Work",
        "text": """
        The COVID-19 pandemic fundamentally transformed how we think about work, accelerating 
        trends that were already underway. What began as an emergency response has evolved 
        into a permanent shift in workplace culture. Surveys consistently show that a majority 
        of knowledge workers prefer hybrid arrangements that combine remote and in-office work. 
        Companies are redesigning their office spaces to prioritize collaboration and social 
        connection rather than individual desk work. Digital collaboration tools have become 
        essential infrastructure, with video conferencing, project management platforms, and 
        virtual whiteboards enabling distributed teams to work effectively. The rise of remote 
        work has also prompted workers to relocate from expensive urban centers to smaller 
        cities and rural areas, reshaping real estate markets and local economies. However, 
        remote work is not without challenges. Many workers report feelings of isolation and 
        difficulty maintaining work-life boundaries. Managers struggle with maintaining team 
        cohesion and company culture across distributed workforces. There are also concerns 
        about the impact on career development for junior employees who may miss out on 
        informal mentorship opportunities. Organizations are experimenting with various 
        solutions, from mandatory in-office days to virtual social events and enhanced digital 
        communication protocols.
        """
    }
]

def load_summarization_model(model_name="facebook/bart-large-cnn"):
    """Load a pre-trained summarization model."""
    print(f"Loading model: {model_name}")
    
    try:
        # Try to use the pipeline for simplicity
        summarizer = pipeline("summarization", model=model_name)
        return summarizer, None, None
    except Exception as e:
        print(f"Pipeline failed, trying alternative: {e}")
        # Fallback to a smaller model
        try:
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            return summarizer, None, None
        except:
            # Final fallback - use t5-small
            print("Using T5-small as fallback...")
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            return None, tokenizer, model

def summarize_text(text, summarizer=None, tokenizer=None, model=None, 
                   max_length=150, min_length=50):
    """Generate summary for given text."""
    # Clean text
    text = ' '.join(text.split())
    
    if summarizer:
        # Using pipeline
        summary = summarizer(text, max_length=max_length, min_length=min_length, 
                           do_sample=False)
        return summary[0]['summary_text']
    else:
        # Using T5 directly
        input_text = "summarize: " + text
        inputs = tokenizer.encode(input_text, return_tensors="pt", 
                                  max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=max_length, 
                                min_length=min_length, length_penalty=2.0,
                                num_beams=4, early_stopping=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extractive_summarize(text, num_sentences=3):
    """Simple extractive summarization based on sentence scoring."""
    import re
    from collections import Counter
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    if len(sentences) <= num_sentences:
        return text
    
    # Calculate word frequencies
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(words)
    
    # Remove common words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'as', 'into', 'through', 'during', 'before', 'after', 'and',
                  'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                  'not', 'only', 'own', 'same', 'than', 'too', 'very', 'that',
                  'this', 'these', 'those', 'it', 'its'}
    
    for word in stop_words:
        word_freq.pop(word, None)
    
    # Score sentences
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        words_in_sentence = re.findall(r'\b\w+\b', sentence.lower())
        score = sum(word_freq.get(word, 0) for word in words_in_sentence)
        score = score / (len(words_in_sentence) + 1)  # Normalize
        # Boost first sentences
        if i < 2:
            score *= 1.2
        sentence_scores.append((score, i, sentence))
    
    # Select top sentences and maintain order
    sentence_scores.sort(reverse=True)
    selected = sorted(sentence_scores[:num_sentences], key=lambda x: x[1])
    
    return ' '.join(s[2] for s in selected)

def compare_summaries(text, summarizer, tokenizer, model):
    """Compare extractive vs abstractive summarization."""
    extractive = extractive_summarize(text, num_sentences=3)
    abstractive = summarize_text(text, summarizer, tokenizer, model)
    
    return {
        'extractive': extractive,
        'abstractive': abstractive
    }

def calculate_compression_ratio(original, summary):
    """Calculate how much the text was compressed."""
    original_words = len(original.split())
    summary_words = len(summary.split())
    return summary_words / original_words

def main():
    print("=" * 50)
    print("Day 17: Text Summarization using Transformers")
    print("=" * 50)
    
    # Load model
    print("\n[1] Loading summarization model...")
    summarizer, tokenizer, model = load_summarization_model()
    print("Model loaded successfully!")
    
    # Process articles
    print("\n[2] Summarizing articles...")
    results = []
    
    for i, article in enumerate(SAMPLE_ARTICLES):
        print(f"\n{'='*60}")
        print(f"Article {i+1}: {article['title']}")
        print('='*60)
        
        original_text = article['text'].strip()
        
        # Generate summaries
        summaries = compare_summaries(original_text, summarizer, tokenizer, model)
        
        # Calculate stats
        original_words = len(original_text.split())
        extractive_ratio = calculate_compression_ratio(original_text, summaries['extractive'])
        abstractive_ratio = calculate_compression_ratio(original_text, summaries['abstractive'])
        
        print(f"\nOriginal ({original_words} words):")
        print(textwrap.fill(original_text[:300] + "...", width=70))
        
        print(f"\n--- Extractive Summary ({len(summaries['extractive'].split())} words, "
              f"{extractive_ratio:.1%} of original) ---")
        print(textwrap.fill(summaries['extractive'], width=70))
        
        print(f"\n--- Abstractive Summary ({len(summaries['abstractive'].split())} words, "
              f"{abstractive_ratio:.1%} of original) ---")
        print(textwrap.fill(summaries['abstractive'], width=70))
        
        results.append({
            'title': article['title'],
            'original_words': original_words,
            'extractive_summary': summaries['extractive'],
            'extractive_words': len(summaries['extractive'].split()),
            'abstractive_summary': summaries['abstractive'],
            'abstractive_words': len(summaries['abstractive'].split())
        })
    
    # Save results
    print("\n[3] Saving results...")
    with open('summarization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to 'summarization_results.json'")
    
    # Interactive demo
    print("\n[4] Demo: Custom text summarization")
    custom_text = """
    The James Webb Space Telescope has captured unprecedented images of distant 
    galaxies, revealing details about the early universe that were previously 
    impossible to observe. Scientists are particularly excited about observations 
    of galaxies that formed just a few hundred million years after the Big Bang. 
    These findings are challenging existing theories about galaxy formation and 
    evolution. The telescope's infrared capabilities allow it to peer through 
    cosmic dust that obscures visible light, opening new windows into stellar 
    nurseries and planetary atmospheres.
    """
    
    print("\nCustom text summary:")
    summary = summarize_text(custom_text.strip(), summarizer, tokenizer, model)
    print(textwrap.fill(summary, width=70))
    
    print("\n" + "=" * 50)
    print("Day 17 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()

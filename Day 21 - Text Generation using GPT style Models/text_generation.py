"""
Day 21: Text Generation using GPT-style Models
30-Day AI Challenge

Generate coherent text using pre-trained transformer language models
from Hugging Face (GPT-2).
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import json
from datetime import datetime

class TextGenerator:
    """Text generator using GPT-2 model."""
    
    def __init__(self, model_name="gpt2"):
        """Initialize with specified model."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.generator = None
        
    def load_model(self):
        """Load the GPT-2 model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            
            # Set padding token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create pipeline for easy generation
            self.generator = pipeline('text-generation', model=self.model, 
                                     tokenizer=self.tokenizer)
            
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate(self, prompt, max_length=100, num_return_sequences=1,
                 temperature=0.7, top_k=50, top_p=0.95, do_sample=True):
        """Generate text from a prompt."""
        if self.generator is None:
            return self._simple_generate(prompt)
        
        outputs = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return [output['generated_text'] for output in outputs]
    
    def generate_with_control(self, prompt, max_new_tokens=50, 
                              repetition_penalty=1.2, no_repeat_ngram=3):
        """Generate with more control over repetition."""
        if self.model is None:
            return self._simple_generate(prompt)
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _simple_generate(self, prompt):
        """Fallback simple text continuation."""
        # Simple Markov-like continuation
        words = prompt.split()
        continuations = [
            "and the journey continues with great anticipation.",
            "which leads to many interesting possibilities.",
            "creating new opportunities for innovation.",
            "inspiring people around the world.",
            "demonstrating the power of technology."
        ]
        import random
        return [prompt + " " + random.choice(continuations)]
    
    def complete_story(self, story_start, num_paragraphs=3, 
                       sentences_per_paragraph=3):
        """Generate a multi-paragraph story continuation."""
        story = story_start
        
        for i in range(num_paragraphs):
            continuation = self.generate_with_control(
                story,
                max_new_tokens=100,
                repetition_penalty=1.3
            )
            story = continuation
        
        return story
    
    def generate_variations(self, prompt, num_variations=3, temperature_range=(0.5, 1.0)):
        """Generate multiple variations with different temperatures."""
        variations = []
        temps = [temperature_range[0] + i * (temperature_range[1] - temperature_range[0]) / (num_variations - 1) 
                 for i in range(num_variations)]
        
        for temp in temps:
            result = self.generate(prompt, temperature=temp, max_length=100)
            variations.append({
                'temperature': temp,
                'text': result[0]
            })
        
        return variations


def generate_creative_content(generator, content_type, topic):
    """Generate specific types of creative content."""
    prompts = {
        'story': f"Once upon a time, in a world where {topic},",
        'poem': f"A poem about {topic}:\n\n",
        'article': f"The latest developments in {topic} have shown that",
        'dialogue': f"Two friends discussing {topic}:\n\nAlice: \"So, what do you think about {topic}?\"\nBob:",
        'description': f"Describing {topic}: It is",
        'news': f"BREAKING NEWS: Experts reveal new insights about {topic}. According to recent studies,",
    }
    
    prompt = prompts.get(content_type, f"Writing about {topic}:")
    return generator.generate(prompt, max_length=200)

def interactive_generation(generator):
    """Interactive text generation mode."""
    print("\n" + "=" * 50)
    print("INTERACTIVE TEXT GENERATION")
    print("=" * 50)
    print("Enter a prompt to generate text. Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        prompt = input("\nPrompt: ").strip()
        
        if prompt.lower() == 'quit':
            break
        
        if not prompt:
            continue
        
        print("\nGenerating...")
        results = generator.generate(prompt, max_length=150, num_return_sequences=2)
        
        for i, text in enumerate(results, 1):
            print(f"\n--- Generation {i} ---")
            print(text)

def main():
    print("=" * 50)
    print("Day 21: Text Generation using GPT-2")
    print("=" * 50)
    
    # Initialize generator
    print("\n[1] Initializing text generator...")
    generator = TextGenerator("gpt2")
    
    # Load model
    print("\n[2] Loading GPT-2 model...")
    success = generator.load_model()
    
    if not success:
        print("Using fallback generation mode.")
    
    # Demo: Basic generation
    print("\n[3] Basic Text Generation Demo:")
    prompts = [
        "Artificial intelligence will change the future by",
        "The best way to learn programming is",
        "In the year 2050, humans will",
    ]
    
    results = []
    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        generated = generator.generate(prompt, max_length=100)
        print(generated[0])
        results.append({'prompt': prompt, 'generated': generated[0]})
    
    # Demo: Temperature variations
    print("\n[4] Temperature Variation Demo:")
    test_prompt = "The secret to happiness is"
    variations = generator.generate_variations(test_prompt, num_variations=3)
    
    for var in variations:
        print(f"\n--- Temperature: {var['temperature']:.2f} ---")
        print(var['text'])
    
    # Demo: Creative content
    print("\n[5] Creative Content Generation:")
    content_types = ['story', 'article', 'dialogue']
    topic = "artificial intelligence"
    
    for content_type in content_types:
        print(f"\n--- {content_type.upper()} about '{topic}' ---")
        generated = generate_creative_content(generator, content_type, topic)
        print(generated[0][:300] + "..." if len(generated[0]) > 300 else generated[0])
    
    # Save results
    print("\n[6] Saving results...")
    output = {
        'basic_generations': results,
        'temperature_variations': variations,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('text_generation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Results saved to 'text_generation_results.json'")
    
    # Interactive mode info
    print("\n[7] For interactive mode, uncomment:")
    print("# interactive_generation(generator)")
    
    print("\n" + "=" * 50)
    print("Day 21 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()

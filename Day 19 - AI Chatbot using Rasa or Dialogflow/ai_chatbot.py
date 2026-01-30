"""
Day 19: AI Chatbot using Transformers
30-Day AI Challenge

Build a conversational AI chatbot using pre-trained transformer models.
Uses DialoGPT or similar conversational models from Hugging Face.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from datetime import datetime

class AIConversationalChatbot:
    """AI-powered chatbot using DialoGPT."""
    
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """Initialize the chatbot with a pre-trained model."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.chat_history_ids = None
        self.conversation_history = []
        self.max_history_length = 5
        
    def load_model(self):
        """Load the conversational model."""
        print(f"Loading model: {self.model_name}")
        print("This may take a moment...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to rule-based responses...")
            return False
    
    def generate_response(self, user_input, max_length=1000):
        """Generate a response to user input."""
        if self.model is None:
            return self._rule_based_response(user_input)
        
        # Encode user input
        new_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token,
            return_tensors='pt'
        )
        
        # Append to chat history
        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids
        
        # Generate response
        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        # Decode response
        response = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        
        # Store in conversation history
        self.conversation_history.append({
            'user': user_input,
            'bot': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Trim history to prevent context from getting too long
        if len(self.conversation_history) > self.max_history_length:
            self.reset_context()
        
        return response
    
    def _rule_based_response(self, user_input):
        """Fallback rule-based responses."""
        user_input_lower = user_input.lower()
        
        responses = {
            'hello': "Hello! How can I help you today?",
            'hi': "Hi there! What's on your mind?",
            'how are you': "I'm doing well, thank you for asking! How about you?",
            'what is your name': "I'm an AI chatbot created for the 30-day AI challenge!",
            'bye': "Goodbye! Have a great day!",
            'help': "I can chat with you about various topics. Just type your message!",
            'thanks': "You're welcome! Is there anything else I can help with?",
            'weather': "I don't have access to real-time weather data, but I hope it's nice where you are!",
            'joke': "Why did the AI go to therapy? Because it had too many neural issues! 🤖",
        }
        
        for key, response in responses.items():
            if key in user_input_lower:
                return response
        
        return "That's interesting! Tell me more about that."
    
    def reset_context(self):
        """Reset the conversation context."""
        self.chat_history_ids = None
        print("(Context reset)")
    
    def get_conversation_history(self):
        """Return the full conversation history."""
        return self.conversation_history
    
    def save_conversation(self, filename="conversation_history.json"):
        """Save conversation history to file."""
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        print(f"Conversation saved to '{filename}'")


class IntentRecognizer:
    """Simple intent recognition for chatbot."""
    
    def __init__(self):
        self.intents = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good evening'],
            'farewell': ['bye', 'goodbye', 'see you', 'take care'],
            'gratitude': ['thank', 'thanks', 'appreciate'],
            'question': ['what', 'how', 'why', 'when', 'where', 'who'],
            'help': ['help', 'assist', 'support'],
            'affirmation': ['yes', 'yeah', 'sure', 'okay', 'ok'],
            'negation': ['no', 'nope', 'not', "don't"],
        }
    
    def recognize(self, text):
        """Recognize the intent of the input text."""
        text_lower = text.lower()
        
        detected_intents = []
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_intents.append(intent)
                    break
        
        return detected_intents if detected_intents else ['general']


def run_interactive_chat(chatbot):
    """Run an interactive chat session."""
    print("\n" + "=" * 50)
    print("AI CHATBOT - Interactive Mode")
    print("=" * 50)
    print("Type 'quit' to exit, 'reset' to clear context")
    print("Type 'save' to save conversation history")
    print("-" * 50)
    
    intent_recognizer = IntentRecognizer()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye! 👋")
                break
            
            if user_input.lower() == 'reset':
                chatbot.reset_context()
                continue
            
            if user_input.lower() == 'save':
                chatbot.save_conversation()
                continue
            
            # Recognize intent
            intents = intent_recognizer.recognize(user_input)
            
            # Generate response
            response = chatbot.generate_response(user_input)
            
            print(f"\nBot: {response}")
            print(f"  [Intent: {', '.join(intents)}]")
            
        except KeyboardInterrupt:
            print("\n\nChat ended by user.")
            break

def demo_conversation(chatbot):
    """Run a demonstration conversation."""
    demo_inputs = [
        "Hello! How are you?",
        "What's your favorite topic to discuss?",
        "Tell me something interesting about AI.",
        "Do you think robots will take over the world?",
        "Thanks for chatting with me!",
    ]
    
    print("\n" + "=" * 50)
    print("DEMO CONVERSATION")
    print("=" * 50)
    
    for user_input in demo_inputs:
        print(f"\nUser: {user_input}")
        response = chatbot.generate_response(user_input)
        print(f"Bot: {response}")
    
    return chatbot.get_conversation_history()

def main():
    print("=" * 50)
    print("Day 19: AI Chatbot using Transformers")
    print("=" * 50)
    
    # Initialize chatbot
    print("\n[1] Initializing chatbot...")
    chatbot = AIConversationalChatbot("microsoft/DialoGPT-small")
    
    # Try to load the model
    print("\n[2] Loading language model...")
    model_loaded = chatbot.load_model()
    
    if not model_loaded:
        print("Using rule-based fallback mode.")
    
    # Run demo conversation
    print("\n[3] Running demo conversation...")
    history = demo_conversation(chatbot)
    
    # Save conversation
    print("\n[4] Saving conversation history...")
    chatbot.save_conversation()
    
    # Print summary
    print("\n[5] Conversation Summary:")
    print(f"  Total exchanges: {len(history)}")
    
    # Interactive mode option
    print("\n" + "-" * 50)
    print("To start interactive chat, uncomment the line below:")
    print("# run_interactive_chat(chatbot)")
    
    print("\n" + "=" * 50)
    print("Day 19 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()

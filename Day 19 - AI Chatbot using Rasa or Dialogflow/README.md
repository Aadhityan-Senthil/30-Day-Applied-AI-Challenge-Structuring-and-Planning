# Day 19: AI Chatbot using Transformers

## Overview
Conversational AI chatbot using Microsoft's DialoGPT model from Hugging Face Transformers, with intent recognition and conversation history management.

## Features
- Pre-trained conversational model (DialoGPT)
- Context-aware responses
- Intent recognition
- Conversation history tracking
- Fallback rule-based responses

## Requirements
```bash
pip install transformers torch
```

## Usage
```bash
python ai_chatbot.py
```

## Interactive Mode
Uncomment the last line in main() to enable interactive chat:
```python
run_interactive_chat(chatbot)
```

## Commands
- `quit` - Exit the chatbot
- `reset` - Clear conversation context
- `save` - Save conversation history

## Model Options
- `microsoft/DialoGPT-small` (fastest)
- `microsoft/DialoGPT-medium` (balanced)
- `microsoft/DialoGPT-large` (best quality)

## Output Files
- `conversation_history.json` - Saved conversations

## Custom Integration
```python
chatbot = AIConversationalChatbot()
chatbot.load_model()
response = chatbot.generate_response("Hello!")
```

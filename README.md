# PyChat

A versatile desktop application that lets you interact with multiple AI providers (OpenAI, Anthropic Claude, Google Gemini, and Ollama) through a unified interface. This PyQt5-based tool offers a seamless experience for using various large language models across different providers. Query your own documents with RAG for custom knowledge.

![PyChat Interface](https://github.com/Magnetron85/PyChat/raw/rag2/screenshot.png)

## Features

### Multiple AI Provider Support
- **OpenAI** (GPT-4, GPT-3.5, etc.)
- **Anthropic Claude** (Claude 3 Opus, Sonnet, Haiku, etc.)
- **Ollama** (for local open-source models)
- **Google Gemini** (Gemini Pro, Gemini Ultra, Gemini Flash, etc.)

### Rich Text Interface
- Syntax highlighting for code blocks
- Streaming support for real-time responses
- Copy code button for easy code reuse

### Advanced Features
- Preprompt system for reusable context templates
- "Show thinking" option for supported models
- Save and load conversations
- Customizable API endpoints
- Thread-based conversation management
- AI-to-AI conversation feature
- Support for OpenAI compatibility mode with Ollama
- Multi-provider usage in a single interface
- RAG (Retrieval-Augmented Generation) capabilities for document Q&A

## Important User Information

### Data Storage & Privacy
- **Local Storage**: All conversations are stored locally in an SQLite database (`chat_history.db`)
- **API Keys**: Your API keys are stored locally using Qt's settings mechanism and are not transmitted beyond the respective API services
- **Logging**: The application logs activities to `ai_chat_debug.log`, which may include message content for debugging purposes
- **No Data Sharing**: PyChat does not send your conversations to any servers except the AI provider APIs you configure

### Security Considerations
- **Credentials**: API keys are stored locally in Qt's settings storage; ensure your computer is secure
- **API Usage**: Your API usage with OpenAI, Anthropic, and Google will incur costs according to those services' pricing
- **Network Access**: The application requires internet access to communicate with remote APIs
- **Local Models**: Using Ollama allows you to run models locally with no data sent to external services

## Installation

### Prerequisites
- Python 3.6+
- Required packages:
  - PyQt5
  - Requests
  - qtconsole (for the Jupyter console integration)
  - fuzzywuzzy (for search functionality)
  - google-genai (for Gemini integration)
  - chromadb (for RAG capability)
  - langchain (for document processing and RAG pipeline)
  - langchain-community (for vectorstore integrations)
  - sentence-transformers (for text embeddings)

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/Magnetron85/PyChat.git
   ```

2. Install the required dependencies:
   ```
   pip install PyQt5 requests qtconsole fuzzywuzzy google-genai chromadb langchain sentence-transformers pymupdf numpy torch docx scikit-learn scipy
   ```

3. Run the application:
   ```
   python pychat.py
   ```

## Setting Up AI Providers

### OpenAI
1. Go to the "Settings" tab
2. Select "OpenAI" from the provider dropdown
3. Enter your OpenAI API key (get it from [OpenAI Platform](https://platform.openai.com))
4. Click "Save Settings"

### Anthropic Claude
1. Go to the "Settings" tab
2. Select "Claude (Anthropic)" from the provider dropdown
3. Enter your Anthropic API key (get it from [Anthropic Console](https://console.anthropic.com))
4. Click "Save Settings"

### Ollama
1. Install [Ollama](https://ollama.ai) on your system
2. Make sure Ollama is running and listening on 0.0.0.0
3. In the application, use the default URL (http://localhost:11434) or modify if Ollama is running elsewhere on the network
4. Click "Refresh Models" to load your installed Ollama models

### Google Gemini
1. Go to the "Settings" tab
2. Select "Gemini (Google)" from the provider dropdown
3. Enter your Google Gemini API key (get it from [Google AI Studio](https://aistudio.google.com/))
4. Click "Save Settings"

PyChat supports various Gemini models including Gemini Pro, Gemini Ultra, and Gemini Flash. The Gemini integration gives you access to Google's powerful multimodal AI capabilities.

## Usage Guide

### Using Preprompts
Preprompts let you store reusable contexts to add to your prompts in conversations (i.e., "keep it brief"):

1. Click on the "Preprompt" button to expand the preprompt panel
2. Click "New" to create a new preprompt
3. Enter a name and content for your preprompt
4. Click "Save" to store the preprompt
5. Select your preprompt before sending messages

You can set default preprompts or configure the application to always use the last selected preprompt.

### Sending Messages
1. Select your desired provider and model
2. Type your message in the input area
3. Click "Send" or press Ctrl+Enter to submit
4. View the AI's response in the chat area

### Conversation Management
- Conversations are organized into threads that are stored locally
- Use "File > New Thread" to start a new conversation
- The thread list on the left shows all your conversations
- Right-click on threads for options like rename, archive, or delete

### Saving Conversations
- Click "Save Chat" to export the current conversation to a text file
- Use "File > Export Thread" to save a conversation in JSON format
- Use "File > Import Thread" to load previously exported conversations

### Model Management
- For Ollama, you need to install models before they appear in the application
- For Google Gemini, OpenAI, and Anthropic, available models are loaded automatically when you have a valid API key

## Model Capabilities

Different AI providers offer unique capabilities through PyChat:

- **OpenAI**: Access to GPT models with strong general capabilities
- **Anthropic Claude**: Well-suited for longer contexts and complex reasoning
- **Ollama**: Run open-source models locally for privacy and no API costs
- **Google Gemini**: Excellent multimodal capabilities with text generation

## Retrieval-Augmented Generation (RAG)

PyChat now includes RAG capabilities, allowing you to chat with your documents and get more accurate, context-aware responses from AI models.

### What is RAG?
Retrieval-Augmented Generation is a technique that enhances AI responses by retrieving relevant information from a document database before generating an answer. This allows the AI to provide more accurate responses based on your specific documents.

### Features
- Upload and process various document formats (PDF, DOCX, TXT)
- Automatically chunk and embed documents using sentence-transformers
- Store document embeddings in a local ChromaDB vector database
- Query your documents using natural language
- Get context-enhanced responses from any supported AI model

### How to Use RAG
1. Go to the "Knowledge" tab
2. Upload your documents using the "Add Documents" button. Can keep related documents in seperate knowledge domains.
3. The system will process and embed your documents automatically
4. Switch to the chat interface and enable the "Use RAG" toggle
5. Select the correct domain in your Chat window. Ask questions about your documents, and PyChat will retrieve relevant information to enhance AI responses

### Available Document Operations
- Add new documents to the vector database
- View all uploaded documents
- Delete documents from the database
- Search within your document collection

## Troubleshooting

- **Models not loading**: Check your API keys in the Settings tab and ensure they're valid
- **Ollama not connecting**: Make sure Ollama is running on your system
- **Error messages in responses**: Check the application log file (`ai_chat_debug.log`) for details
- **API Key Issues**: If you receive authorization errors, verify your API keys are correct and have not expired
- **Database Issues**: If experiencing data loss, check file permissions for the `chat_history.db` file

### Data Cleanup
If you want to remove all stored data:

- Delete the `chat_history.db` file to remove all conversations
- API keys can be cleared through the Settings tab for each provider
- Delete the `ai_chat_debug.log` file to remove all debug logs

## Screenshots

![Thread Management](https://github.com/Magnetron85/PyChat/raw/main/screenshots/threads.png)

![Settings Interface](https://github.com/Magnetron85/PyChat/raw/main/screenshots/settings.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

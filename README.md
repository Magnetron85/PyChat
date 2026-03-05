# PyChat

A versatile desktop application that lets you interact with multiple AI providers (OpenAI, Anthropic Claude, Google Gemini, and Ollama) through a unified interface. This PyQt5-based tool offers a seamless experience for using various large language models across different providers. Query your own documents with RAG for custom knowledge, with enterprise-grade security and compliance features built in.

![PyChat Interface](https://github.com/Magnetron85/PyChat/raw/rag2/screenshot.png)

## Features

### Multiple AI Provider Support
- **OpenAI** (GPT-4, GPT-4o, GPT-3.5, etc.)
- **Anthropic Claude** (Claude 4, Claude 3.5, Claude 3 Opus/Sonnet/Haiku, etc.)
- **Ollama** (for local open-source models)
- **Google Gemini** (Gemini 2.0 Flash, Gemini 1.5 Pro/Flash, etc.)

### Rich Text Interface
- Syntax highlighting for code blocks
- Streaming support for real-time responses
- Copy code button for easy code reuse
- Response regeneration for quick iteration

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
- Token usage tracking and cost estimation per conversation
- Keyboard shortcuts (Ctrl+Enter to send, Escape to clear input)

### Security & Compliance
- **AES-256 encryption at rest** for API keys and sensitive data using Fernet (cryptography library)
- **HIPAA/SOC2-aligned audit logging** with timestamped event tracking
- **PII/PHI detection and sanitization** (SSN, phone, email, credit card, date of birth patterns)
- **Compliance dashboard** accessible from the Tools menu
- **Audit trail** with exportable logs for compliance review
- Encryption keys stored with restrictive file permissions (0600)

### Token Usage & Cost Tracking
- Automatic token estimation per message exchange
- Per-thread and per-provider usage statistics
- Cost estimation for all major models (OpenAI, Anthropic, Gemini)
- Local Ollama models tracked as zero-cost
- Session summary with today's usage breakdown
- Accessible from Tools > Token Usage

## Important User Information

### Data Storage & Privacy
- **Local Storage**: All conversations are stored locally in an SQLite database (`chat_history.db`)
- **Encrypted API Keys**: API keys are encrypted at rest using AES-256 encryption and stored at `~/.pychat/`
- **Audit Logging**: Security-relevant events are logged to a local SQLite audit trail for compliance
- **Application Logging**: The application logs activities to `ai_chat_debug.log` for debugging purposes
- **No Data Sharing**: PyChat does not send your conversations to any servers except the AI provider APIs you configure

### Security Considerations
- **Credentials**: API keys are encrypted with AES-256 before storage; encryption keys are protected with restrictive file permissions
- **API Usage**: Your API usage with OpenAI, Anthropic, and Google will incur costs according to those services' pricing
- **Network Access**: The application requires internet access to communicate with remote APIs
- **Local Models**: Using Ollama allows you to run models locally with no data sent to external services
- **PII Protection**: Built-in PII/PHI detection warns about sensitive data patterns before transmission

## Installation

### Prerequisites
- Python 3.8+
- Required packages (see below)

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/Magnetron85/PyChat.git
   ```

2. Install the required dependencies:
   ```
   pip install PyQt5 requests qtconsole python-Levenshtein fuzzywuzzy google-genai markdown langchain sentence-transformers pymupdf numpy torch python-docx scikit-learn scipy cryptography python-pptx openpyxl pytesseract
   ```

3. For image OCR support (optional), install Tesseract:
   ```
   # Ubuntu/Debian
   sudo apt install tesseract-ocr

   # macOS
   brew install tesseract

   # Windows: download from https://github.com/UB-Mannheim/tesseract/wiki
   ```

4. Run the application:
   ```
   python pychat.py
   ```

### Dependency Overview

| Package | Purpose |
|---------|---------|
| `PyQt5` | Desktop GUI framework |
| `requests` | HTTP client for API calls |
| `cryptography` | AES-256 encryption for API keys and sensitive data |
| `sentence-transformers` | Text embeddings for RAG |
| `scikit-learn` | TF-IDF retrieval for RAG |
| `pymupdf` | PDF text extraction and image OCR |
| `python-docx` | Word document processing |
| `python-pptx` | PowerPoint presentation processing |
| `openpyxl` | Excel spreadsheet processing |
| `pytesseract` | OCR for images (requires Tesseract) |
| `google-genai` | Google Gemini API integration |
| `fuzzywuzzy` | Fuzzy search for threads |
| `torch` | Backend for sentence-transformers |

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
5. Use the "Regenerate" button to get an alternative response

### Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| Ctrl+Enter | Send message |
| Escape | Clear input field |
| Ctrl+Shift+C | Copy last response |

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

### Tools Menu
- **Token Usage**: View per-thread and per-provider token consumption and estimated costs
- **System Health**: Check system status including database size, encryption status, and document counts
- **Compliance Status**: Review SOC2/HIPAA compliance posture (encryption, audit logging, PII detection)
- **Audit Log**: Browse and review the security audit trail

## Retrieval-Augmented Generation (RAG)

PyChat includes RAG capabilities, allowing you to chat with your documents and get more accurate, context-aware responses from AI models.

### What is RAG?
Retrieval-Augmented Generation is a technique that enhances AI responses by retrieving relevant information from a document database before generating an answer. This allows the AI to provide more accurate responses based on your specific documents.

### Supported File Types
| Format | Extensions | Processing |
|--------|-----------|------------|
| PDF | `.pdf` | Text extraction via PyMuPDF |
| Word | `.docx` | Paragraph and table extraction via python-docx |
| PowerPoint | `.pptx` | Slide-by-slide extraction including tables via python-pptx |
| Excel | `.xlsx`, `.xls` | Sheet-aware structured extraction via openpyxl |
| CSV | `.csv` | Structured row/column parsing |
| Images | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` | OCR via PyMuPDF + pytesseract fallback |
| Plain Text | `.txt`, `.md`, `.log`, etc. | Direct text reading |

### RAG Architecture
PyChat uses a two-stage retrieval pipeline:
1. **TF-IDF** for fast initial candidate retrieval
2. **Sentence-BERT** for semantic re-ranking of top candidates

This approach balances speed with retrieval quality.

### How to Use RAG
1. Go to the "Knowledge" tab
2. Upload your documents using the "Add Documents" button. You can keep related documents in separate knowledge domains.
3. The system will process and embed your documents automatically
4. Switch to the chat interface and enable the "Use RAG" toggle
5. Select the correct domain in your Chat window. Ask questions about your documents, and PyChat will retrieve relevant information to enhance AI responses

### Available Document Operations
- Add new documents to the knowledge base
- View all uploaded documents
- Delete documents from the database
- Search within your document collection

## Model Capabilities

Different AI providers offer unique capabilities through PyChat:

- **OpenAI**: Access to GPT models with strong general capabilities
- **Anthropic Claude**: Well-suited for longer contexts and complex reasoning
- **Ollama**: Run open-source models locally for privacy and no API costs
- **Google Gemini**: Excellent multimodal capabilities with text generation

## Troubleshooting

- **Models not loading**: Check your API keys in the Settings tab and ensure they're valid
- **Ollama not connecting**: Make sure Ollama is running on your system
- **Error messages in responses**: Check the application log file (`ai_chat_debug.log`) for details
- **API Key Issues**: If you receive authorization errors, verify your API keys are correct and have not expired
- **Database Issues**: If experiencing data loss, check file permissions for the `chat_history.db` file
- **OCR not working**: Ensure Tesseract is installed on your system (`tesseract --version` to verify)
- **Encryption key errors**: Check that `~/.pychat/` directory exists and has proper permissions

### Data Cleanup
If you want to remove all stored data:

- Delete the `chat_history.db` file to remove all conversations
- API keys can be cleared through the Settings tab for each provider
- Delete the `ai_chat_debug.log` file to remove all debug logs
- Delete `~/.pychat/` to remove encryption keys and audit logs

## Screenshots

![Thread Management](https://github.com/Magnetron85/PyChat/raw/main/screenshots/threads.png)

![Settings Interface](https://github.com/Magnetron85/PyChat/raw/main/screenshots/settings.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

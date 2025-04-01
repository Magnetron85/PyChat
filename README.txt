# PyChat: Your All-in-One AI Chat Client 🤖💬

[![Screenshot](screenshot.png)](screenshot.png)

Tired of juggling multiple AI platforms? PyChat is a versatile desktop application that brings the power of **OpenAI (GPT-4, GPT-3.5), Anthropic Claude (Claude 3 Opus, Sonnet, Haiku), Ollama (local models), and Google Gemini** into a single, user-friendly interface. Enjoy seamless conversations, code highlighting, and advanced features, all within a sleek PyQt5 application.

## ✨ Features

* **Multi-Provider Support:**
    * OpenAI (GPT-4, GPT-3.5, etc.)
    * Anthropic Claude (Claude 3 Opus, Sonnet, Haiku, etc.)
    * Ollama (for local open-source models)
    * Google Gemini
* **Rich Text Interface:**
    * Syntax highlighting for code blocks.
    * Streaming support for real-time responses.
    * "Copy Code" button for easy code reuse.
* **Advanced Features:**
    * Preprompt system for reusable context templates.
    * "Show thinking" option for DeepSeek.
    * Save and load conversations (JSON and Text).
    * Customizable API endpoints.
    * Thread-based conversation management.
    * AI-to-AI conversation feature.
* **User-Friendly Design:**
    * Simple and intuitive interface.
    * Streaming responses in real-time.
    * Easy configuration for multiple providers.

## 🔒 Important User Information: Data & Security

* **Local Storage:**
    * All conversations are stored locally in an SQLite database (`chat_history.db`).
    * API keys are stored locally using Qt's settings mechanism.
    * Logging to `ai_chat_debug.log` may include message content for debugging.
* **Privacy:**
    * PyChat does not send your conversations to any servers except the AI provider APIs you configure.
* **Security Considerations:**
    * Ensure your computer is secure, as API keys are stored locally.
    * API usage with OpenAI and Anthropic incurs costs according to their pricing.
    * Internet access is required for remote APIs.
    * Ollama allows local model usage without external data transfer.

## 📦 Dependencies

* Python 3.6+
* PyQt5
* Requests
* qtconsole
* fuzzywuzzy
* google-genai

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Magnetron85/PyChat.git](https://github.com/Magnetron85/PyChat.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install PyQt5 requests qtconsole fuzzywuzzy google-genai
    ```
3.  **Run the application:**
    ```bash
    python pychat.py
    ```

## ⚙️ Setup

### Setting up OpenAI

1.  Go to the "Settings" tab.
2.  Select "OpenAI" from the provider dropdown.
3.  Enter your OpenAI API key (from [OpenAI Platform](https://platform.openai.com/)).
4.  Click "Save Settings."

### Setting up Anthropic Claude

1.  Go to the "Settings" tab.
2.  Select "Claude (Anthropic)" from the provider dropdown.
3.  Enter your Anthropic API key (from [Anthropic Console](https://console.anthropic.com/)).
4.  Click "Save Settings."

### Setting up Ollama

1.  Install [Ollama](https://ollama.ai/) on your system.
2.  Ensure Ollama is running (default URL: `http://localhost:11434`).
3.  In PyChat, use the default URL or modify if needed.
4.  Click "Refresh Models" to load your installed Ollama models.

### Setting up Google Gemini

1.  Go to the "Settings" tab.
2.  Select "Gemini (Google)" from the provider dropdown.
3.  Enter your Google Gemini API key.
4.  Click "Save Settings."

## 📝 Usage

### Preprompts

* Click "Preprompt" to expand the panel.
* Click "New" to create a preprompt.
* Enter a name and content, then "Save."
* Select your preprompt before sending messages.

### Sending Messages

1.  Select your provider and model.
2.  Type your message in the input area.
3.  Click "Send" or press Ctrl+Enter.
4.  View the AI's response in the chat area.

### Conversation Management

* "File > New Thread" starts a new conversation.
* Thread list on the left shows all conversations.
* Right-click threads for rename, archive, or delete.

### Saving Conversations

* "Save Chat" exports to a text file.
* "File > Export Thread" exports to a JSON file.
* "File > Import Thread" imports a JSON file.

### Model Management

* Ollama: Install models via Ollama.
* Gemini, OpenAI, Anthropic: Models load automatically with valid API keys.

## 🛠️ Troubleshooting

* **Models not loading:** Verify API keys in Settings.
* **Ollama connection issues:** Ensure Ollama is running.
* **Error messages:** Check `ai_chat_debug.log`.
* **API Key issues:** Verify your keys are correct and active.
* **Database issues:** Check file permissions for `chat_history.db`.

## 🧹 Data Cleanup

* Delete `chat_history.db` to remove conversations.
* Clear API keys in the Settings tab.
* Delete `ai_chat_debug.log` to remove logs.

## 📜 License

MIT License

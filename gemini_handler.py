import logging
import sys
import subprocess
from PyQt5.QtCore import QThread, pyqtSignal

class GeminiRequestWorker(QThread):
    """Worker thread specifically for Google Gemini API requests using the official SDK"""
    finished = pyqtSignal(str, bool)  # Signal to emit response and success status
    chunk_received = pyqtSignal(str)  # Signal to emit chunks during streaming
    
    def __init__(self, api_url, model, prompt, api_key, stream=True, use_conversation=False):
        super().__init__()
        self.api_url = api_url  # Not used directly but kept for compatibility
        self.model = model
        self.prompt = prompt
        self.api_key = api_key
        self.stream = stream
        self.use_conversation = use_conversation
        
        # Check if google-generativeai package is installed, install if needed
        self.ensure_sdk_installed()
    
    def ensure_sdk_installed(self):
        """Make sure the google-generativeai package is installed"""
        try:
            import google.generativeai as genai
            logging.info("Google Generative AI SDK is already installed")
        except ImportError:
            logging.info("Installing Google Generative AI SDK...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
                logging.info("Google Generative AI SDK installed successfully")
            except Exception as e:
                logging.error(f"Failed to install Google Generative AI SDK: {str(e)}")
                # We'll handle this error in the run method
    
    def run(self):
        try:
            # Try to import the SDK
            try:
                import google.generativeai as genai
            except ImportError:
                self.finished.emit("Error: Google Generative AI SDK is not installed. Please install it manually with 'pip install google-generativeai'", False)
                return
            
            # Configure the API
            genai.configure(api_key=self.api_key)
            
            # Generation config
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
            
            try:
                if self.stream:
                    # Get a model instance
                    model = genai.GenerativeModel(
                        model_name=self.model,
                        generation_config=generation_config
                    )
                    
                    # Handle conversation history if needed
                    if self.use_conversation and isinstance(self.prompt, list):
                        # Create a chat session
                        chat = model.start_chat()
                        
                        # Add previous messages to the chat
                        for msg in self.prompt[:-1]:  # All but the last message
                            role = msg["role"]
                            content = msg["content"]
                            
                            if role == "user":
                                # Add user message to history without getting response
                                chat.history.append({
                                    "role": "user",
                                    "parts": [{"text": content}]
                                })
                            elif role == "assistant":
                                # Add assistant message to history
                                chat.history.append({
                                    "role": "model",
                                    "parts": [{"text": content}]
                                })
                        
                        # Get the last user message
                        last_msg = self.prompt[-1]
                        if last_msg["role"] == "user":
                            last_content = last_msg["content"]
                        else:
                            # If the last message is not from user, use a default
                            last_content = "Continue the conversation"
                        
                        # Send the message and stream the response
                        response = chat.send_message(last_content, stream=True)
                    else:
                        # Simple streaming response for a single message
                        response = model.generate_content(self.prompt, stream=True)
                    
                    # Accumulate the full response
                    full_response = ""
                    
                    # Process the streaming response
                    for chunk in response:
                        try:
                            if hasattr(chunk, 'text') and chunk.text:
                                chunk_text = chunk.text
                                full_response += chunk_text
                                self.chunk_received.emit(chunk_text)
                        except Exception as e:
                            logging.error(f"Error processing chunk: {str(e)}")
                    
                    # If we got no content but no errors, return a default message
                    if not full_response:
                        logging.warning("Empty response from Gemini SDK")
                        self.finished.emit("No response content was received from the model. Please try again.", False)
                    else:
                        # Emit the complete response
                        self.finished.emit(full_response.strip(), True)
                
                else:
                    # Non-streaming request
                    model = genai.GenerativeModel(
                        model_name=self.model,
                        generation_config=generation_config
                    )
                    
                    # Handle conversation history if needed
                    if self.use_conversation and isinstance(self.prompt, list):
                        # Create a chat session
                        chat = model.start_chat()
                        
                        # Add previous messages to the chat
                        for msg in self.prompt[:-1]:  # All but the last message
                            role = msg["role"]
                            content = msg["content"]
                            
                            if role == "user":
                                # Add user message to history without getting response
                                chat.history.append({
                                    "role": "user",
                                    "parts": [{"text": content}]
                                })
                            elif role == "assistant":
                                # Add assistant message to history
                                chat.history.append({
                                    "role": "model",
                                    "parts": [{"text": content}]
                                })
                        
                        # Get the last user message
                        last_msg = self.prompt[-1]
                        if last_msg["role"] == "user":
                            last_content = last_msg["content"]
                        else:
                            # If the last message is not from user, use a default
                            last_content = "Continue the conversation"
                        
                        # Send the message and get the response
                        response = chat.send_message(last_content)
                    else:
                        # Simple response for a single message
                        response = model.generate_content(self.prompt)
                    
                    # Check if the response has text
                    if hasattr(response, 'text') and response.text:
                        content = response.text
                        self.finished.emit(content, True)
                    else:
                        logging.warning("No text in response")
                        self.finished.emit("No text content was received from the model. Please try again.", False)
                
            except Exception as e:
                error_msg = str(e)
                logging.error(f"Gemini SDK error: {error_msg}")
                
                # Check for known error patterns
                if "API key not valid" in error_msg:
                    self.finished.emit("Error: Invalid API key. Please check your Google API key.", False)
                elif "Model not found" in error_msg:
                    self.finished.emit(f"Error: Model '{self.model}' not found or not available with your API key.", False)
                elif "Permission denied" in error_msg:
                    self.finished.emit("Error: Permission denied. Your API key may not have access to this model.", False)
                elif "exceeded the rate limit" in error_msg:
                    self.finished.emit("Error: Rate limit exceeded. Please try again later.", False)
                else:
                    self.finished.emit(f"Error: {error_msg}", False)
        
        except Exception as e:
            logging.error(f"Error in Gemini request: {str(e)}")
            self.finished.emit(f"Error: {str(e)}", False)


class GeminiModelsWorker(QThread):
    """Worker thread to load available Gemini models using the SDK"""
    finished = pyqtSignal(list, bool)
    
    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key
        
        # Check if google-generativeai package is installed, install if needed
        self.ensure_sdk_installed()
    
    def ensure_sdk_installed(self):
        """Make sure the google-generativeai package is installed"""
        try:
            import google.generativeai as genai
            logging.info("Google Generative AI SDK is already installed")
        except ImportError:
            logging.info("Installing Google Generative AI SDK...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
                logging.info("Google Generative AI SDK installed successfully")
            except Exception as e:
                logging.error(f"Failed to install Google Generative AI SDK: {str(e)}")
                # We'll handle this error in the run method
    
    def run(self):
        try:
            # Try to import the SDK
            try:
                import google.generativeai as genai
            except ImportError:
                self.finished.emit([], False)
                return
            
            # Configure the API
            genai.configure(api_key=self.api_key)
            
            try:
                # Get available models from the API
                models_info = genai.list_models()
                
                # Filter for models that support text generation
                available_models = []
                
                for model in models_info:
                    supports_text = "generateContent" in model.supported_generation_methods
                    if supports_text and "gemini" in model.name.lower():
                        # Extract just the model name without the full path
                        model_name = model.name.split('/')[-1]
                        available_models.append(model_name)
                
                # If we couldn't get any models, use a default list
                if not available_models:
                    available_models = [
                        "gemini-pro",
                        "gemini-pro-vision",
                        "gemini-1.5-pro",
                        "gemini-1.5-flash"
                    ]
                
                logging.info(f"Available Gemini models: {available_models}")
                self.finished.emit(available_models, True)
                
            except Exception as e:
                logging.error(f"Failed to get models list: {str(e)}")
                
                # Use a default list of models
                default_models = [
                    "gemini-pro",
                    "gemini-pro-vision",
                    "gemini-1.5-pro",
                    "gemini-1.5-flash"
                ]
                
                logging.info(f"Using default models list: {default_models}")
                self.finished.emit(default_models, True)
                
        except Exception as e:
            logging.error(f"Error in GeminiModelsWorker: {str(e)}")
            self.finished.emit([], False)
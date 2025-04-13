import re
import logging
import markdown
from PyQt5.QtWidgets import QTextEdit, QApplication
from PyQt5.QtCore import QMimeData, Qt, QRect
from PyQt5.QtGui import (QFont, QTextCursor, QColor, QTextCharFormat, QSyntaxHighlighter, 
                        QTextBlockFormat, QPainter, QTextFormat, QBrush, QPen, QTextOption)

class CodeHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for code blocks"""
    
    def __init__(self, parent=None, language="python"):
        super().__init__(parent)
        self.language = language.lower()
        self.highlighting_rules = []
        
        if self.language in ["python", "py"]:
            self._setup_python_rules()
        elif self.language in ["javascript", "js"]:
            self._setup_javascript_rules()
        else:
            # Default rules for other languages
            self._setup_generic_rules()
            
    def _setup_python_rules(self):
        # Python keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#0000FF"))
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            "and", "as", "assert", "break", "class", "continue", "def",
            "del", "elif", "else", "except", "False", "finally", "for",
            "from", "global", "if", "import", "in", "is", "lambda", "None",
            "not", "or", "pass", "raise", "return", "True", "try", "while",
            "with", "yield"
        ]
        
        for word in keywords:
            pattern = r'\b' + word + r'\b'
            self.highlighting_rules.append((re.compile(pattern), keyword_format))
        
        # String literals
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#008000"))
        self.highlighting_rules.append((re.compile(r'"[^"\\]*(\\.[^"\\]*)*"'), string_format))
        self.highlighting_rules.append((re.compile(r"'[^'\\]*(\\.[^'\\]*)*'"), string_format))
        
        # Function calls
        function_format = QTextCharFormat()
        function_format.setForeground(QColor("#800080"))
        self.highlighting_rules.append((re.compile(r'\b[A-Za-z0-9_]+(?=\()'), function_format))
        
        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#FF8000"))
        self.highlighting_rules.append((re.compile(r'\b\d+\b'), number_format))
        
        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#808080"))
        self.highlighting_rules.append((re.compile(r'#[^\n]*'), comment_format))
    
    def _setup_javascript_rules(self):
        # JavaScript keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#0000FF"))
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            "break", "case", "catch", "class", "const", "continue", "debugger",
            "default", "delete", "do", "else", "export", "extends", "false",
            "finally", "for", "function", "if", "import", "in", "instanceof",
            "new", "null", "return", "super", "switch", "this", "throw", "true",
            "try", "typeof", "var", "void", "while", "with", "yield", "let"
        ]
        
        for word in keywords:
            pattern = r'\b' + word + r'\b'
            self.highlighting_rules.append((re.compile(pattern), keyword_format))
        
        # String literals
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#008000"))
        self.highlighting_rules.append((re.compile(r'"[^"\\]*(\\.[^"\\]*)*"'), string_format))
        self.highlighting_rules.append((re.compile(r"'[^'\\]*(\\.[^'\\]*)*'"), string_format))
        self.highlighting_rules.append((re.compile(r"`[^`\\]*(\\.[^`\\]*)*`"), string_format))
        
        # Function calls
        function_format = QTextCharFormat()
        function_format.setForeground(QColor("#800080"))
        self.highlighting_rules.append((re.compile(r'\b[A-Za-z0-9_]+(?=\()'), function_format))
        
        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#FF8000"))
        self.highlighting_rules.append((re.compile(r'\b\d+\b'), number_format))
        
        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#808080"))
        self.highlighting_rules.append((re.compile(r'//[^\n]*'), comment_format))
        self.highlighting_rules.append((re.compile(r'/\*[\s\S]*?\*/', re.MULTILINE), comment_format))
    
    def _setup_generic_rules(self):
        # Generic patterns for most languages
        # String literals
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#008000"))
        self.highlighting_rules.append((re.compile(r'"[^"\\]*(\\.[^"\\]*)*"'), string_format))
        self.highlighting_rules.append((re.compile(r"'[^'\\]*(\\.[^'\\]*)*'"), string_format))
        
        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#FF8000"))
        self.highlighting_rules.append((re.compile(r'\b\d+\b'), number_format))
    
    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), format)


class EnhancedChatBrowser(QTextEdit):
    """Custom QTextEdit with beautiful code block formatting and copy capability"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Segoe UI", 10))
         
        # Configure document margin and line spacing
        self.document().setDocumentMargin(15)
        
        # Set up options
        option = self.document().defaultTextOption()
        self.document().setDefaultTextOption(option)
        
        # Set line spacing using text block format
        block_format = QTextBlockFormat()
        block_format.setLineHeight(150, QTextBlockFormat.ProportionalHeight)
        
        # Apply the format to the current block
        cursor = self.textCursor()
        cursor.setBlockFormat(block_format)
        self.setTextCursor(cursor)
        
        # State tracking for code blocks
        self.in_code_block = False
        self.code_block_buffer = ""
        self.code_language = ""
        
        # Track if we're in a streaming response
        self.is_streaming = False
        self.streaming_text = ""
        self.streaming_started = False
        self.streaming_start_position = -1
        
        # Add a dictionary to track streaming responses by unique IDs
        self.streaming_responses = {}
        self.current_streaming_id = None
        
        # Initialize search-related variables
        self.search_match_positions = []
        self.current_search_match_index = -1
        self.current_search_term = ""
        
        # Style the widget
        self.setStyleSheet("""
            QTextEdit {
                background-color: white;
                color: #24292e;
                border: 1px solid #e1e4e8;
                border-radius: 6px;
                selection-background-color: #b3d7ff;
                padding: 5px;
            }
        """)
    
    def clear(self):
        """Clear the chat display"""
        super().clear()
    
    def append(self, text, message_id=None):
        """Add text to the chat display with proper formatting."""
        # Reset streaming state for a complete message
        self.is_streaming = False
        self.streaming_text = ""
        
        # If message_id is provided, append a hidden marker.
        if message_id:
            marker = f"<!--msgID:{message_id}-->"
            text += "\n" + marker + "\n"
        
        # Get the text cursor and move to the end for insertion.
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Determine formatting based on message type
        if text.startswith("[SYSTEM] Response from:"):
            # Create a left-aligned block format
            block_format = QTextBlockFormat()
            block_format.setAlignment(Qt.AlignRight)
            block_format.setBackground(QColor("#FAFAFA"))  # Very very light gray
            cursor.setBlockFormat(block_format)
            
            # Apply text formatting
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("#9199a2"))
            fmt.setFontItalic(True)
            cursor.insertText(text, fmt)
        
        elif text.startswith("[SYSTEM]"):
            # Create a left-aligned block format
            block_format = QTextBlockFormat()
            block_format.setAlignment(Qt.AlignRight)
            cursor.setBlockFormat(block_format)
            
            # Apply text formatting
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("#9199a2"))
            fmt.setFontItalic(True)
            cursor.insertText(text, fmt)
            
        elif text.startswith(">"):
            block_format = QTextBlockFormat()
            block_format.setBackground(QColor("#F0F0F0"))  # Very light gray
            block_format.setAlignment(Qt.AlignLeft)
            cursor.setBlockFormat(block_format)
            
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("#000000"))
            fmt.setFontWeight(QFont.Bold)
            cursor.insertText(text + "\n", fmt)
        elif not text.strip():
            block_format = QTextBlockFormat()
            block_format.setAlignment(Qt.AlignLeft)
            cursor.setBlockFormat(block_format)
            
            cursor.insertBlock()
        else:
            block_format = QTextBlockFormat()
            block_format.setAlignment(Qt.AlignLeft)
            cursor.setBlockFormat(block_format)
            
            # For assistant messages, handle code blocks and markdown
            if "```" in text:
                # Split by code blocks
                pattern = r'(```(?:\w*)\n[\s\S]*?\n```)'
                parts = re.split(pattern, text)
                
                for part in parts:
                    if part.strip() and part.startswith("```") and part.endswith("```"):
                        # This is a code block - format it properly
                        self._insert_code_block(cursor, part)
                    elif part.strip():
                        # Normal text part - process for markdown
                        processed_part = self.process_markdown(part)
                        cursor.insertHtml(processed_part)
                        cursor.insertBlock()  # Add a block after each processed part
                
                # Add final newline
                cursor.insertBlock()
            else:
                # No code blocks, apply markdown processing
                block_format = QTextBlockFormat()
                block_format.setAlignment(Qt.AlignLeft)
                cursor.setBlockFormat(block_format)
            
                processed_text = self.process_markdown(text)
                cursor.insertHtml(processed_text)
                cursor.insertBlock()  # Add a block after content
        
        # Update cursor and make sure the view scrolls to the bottom
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def process_markdown(self, text):
        """Convert comprehensive Markdown formatting to rich text HTML while preserving newlines"""
        # Split the text into lines to preserve paragraph structure
        lines = text.split('\n')
        processed_lines = []
        
        # Track if we're in a list, table, paragraph, or blockquote
        in_unordered_list = False
        in_ordered_list = False
        in_table = False
        table_rows = []
        list_item_count = 0
        in_paragraph = False
        in_blockquote = False
        
        # Process each line
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Escape any existing HTML to prevent conflicts
            processed_line = line.replace("<", "&lt;").replace(">", "&gt;")
            
            # Process tables
            if processed_line.strip().startswith('|') and processed_line.strip().endswith('|'):
                if not in_table:
                    in_table = True
                    table_rows = []
                
                # Parse table row
                cells = [cell.strip() for cell in processed_line.strip('|').split('|')]
                table_rows.append(cells)
                
                # Check if the next line is not a table row
                if i + 1 >= len(lines) or not lines[i + 1].strip().startswith('|'):
                    in_table = False
                    # Render table
                    table_html = ['<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">']
                    for row_idx, row in enumerate(table_rows):
                        table_html.append('<tr>')
                        for cell in row:
                            tag = 'th' if row_idx == 0 else 'td'
                            table_html.append(f'<{tag} style="border: 1px solid #ddd; padding: 8px; text-align: left;">{self._process_inline_formatting(cell)}</{tag}>')
                        table_html.append('</tr>')
                    table_html.append('</table>')
                    processed_lines.append(''.join(table_html))
                    table_rows = []
                
                i += 1
                continue
            
            # Process headers (### Header)
            if processed_line.strip().startswith('#'):
                # Close any open paragraphs or lists before adding a header
                if in_paragraph:
                    processed_lines.append("</p>")
                    in_paragraph = False
                if in_unordered_list:
                    processed_lines.append("</ul>")
                    in_unordered_list = False
                if in_ordered_list:
                    processed_lines.append("</ol>")
                    in_ordered_list = False
                
                if processed_line.startswith('### '):
                    processed_line = f"<h3>{processed_line[4:].strip()}</h3>\n"
                elif processed_line.startswith('## '):
                    processed_line = f"<h2>{processed_line[3:].strip()}</h2>\n"
                elif processed_line.startswith('# '):
                    processed_line = f"<h1>{processed_line[2:].strip()}</h1>\n"
                else:
                    # Not a proper header, treat as normal text
                    processed_line = self._process_inline_formatting(processed_line)
                    if not in_paragraph and processed_line.strip():
                        processed_line = f"<p style='margin: 8px 0;'>{processed_line}"
                        in_paragraph = True
            
            # Process blockquotes (> quoted text)
            elif processed_line.strip().startswith('> '):
                quoted_text = processed_line[2:].strip()
                quoted_text = self._process_inline_formatting(quoted_text)
                
                # Check if we're already in a blockquote
                if in_blockquote:
                    processed_line = f"{quoted_text}<br>"
                else:
                    # Close any open paragraph first
                    if in_paragraph:
                        processed_lines.append("</p>")
                        in_paragraph = False
                    
                    processed_line = f"<blockquote style='border-left: 4px solid #ddd; margin: 8px 0; padding-left: 10px;'>{quoted_text}"
                    in_blockquote = True
                
                # Check if next line is not a blockquote or is the end
                if i + 1 >= len(lines) or not lines[i + 1].strip().startswith('> '):
                    processed_line += "</blockquote>"
                    in_blockquote = False
            
            # Process horizontal rules
            elif processed_line.strip() in ('---', '***', '___'):
                if in_paragraph:
                    processed_lines.append("</p>")
                    in_paragraph = False
                processed_line = ""
            
            # Process unordered lists (* Item or - Item)
            elif processed_line.strip().startswith(('* ', '- ')):
                # Close any open paragraph first
                if in_paragraph:
                    processed_lines.append("</p>")
                    in_paragraph = False
                
                item_text = processed_line.strip()[2:].strip()
                item_text = self._process_inline_formatting(item_text)
                
                if not in_unordered_list:
                    # Start a new list
                    processed_line = f"<ul style='margin: 8px 0; padding-left: 20px;'>\n<li>{item_text}</li>"
                    in_unordered_list = True
                else:
                    # Continue the list
                    processed_line = f"<li>{item_text}</li>"
            
            # Process ordered lists (1. Item)
            elif re.match(r'^\s*\d+\.\s', processed_line):
                # Close any open paragraph first
                if in_paragraph:
                    processed_lines.append("</p>")
                    in_paragraph = False
                
                item_text = re.sub(r'^\s*\d+\.\s', '', processed_line).strip()
                item_text = self._process_inline_formatting(item_text)
                
                if not in_ordered_list:
                    # Start a new list
                    processed_line = f"<ol style='margin: 8px 0; padding-left: 20px;'>\n<li>{item_text}</li>"
                    in_ordered_list = True
                    list_item_count = 1
                else:
                    # Continue the list
                    processed_line = f"<li>{item_text}</li>"
                    list_item_count += 1
            
            # Process empty lines - they may indicate paragraph breaks
            elif not processed_line.strip():
                # Close any open lists
                if in_unordered_list:
                    processed_line = "</ul>"
                    in_unordered_list = False
                elif in_ordered_list:
                    processed_line = "</ol>"
                    in_ordered_list = False
                    list_item_count = 0
                
                # Close any open paragraph
                if in_paragraph:
                    processed_line = "</p>"
                    in_paragraph = False
            
            # Process normal text
            else:
                # Check if we need to close any open lists
                if in_unordered_list and not lines[i+1].strip().startswith(('* ', '- ')) if i+1 < len(lines) else True:
                    processed_line = "</ul>\n" + self._process_inline_formatting(processed_line)
                    in_unordered_list = False
                elif in_ordered_list and not re.match(r'^\s*\d+\.\s', lines[i+1]) if i+1 < len(lines) else True:
                    processed_line = "</ol>\n" + self._process_inline_formatting(processed_line)
                    in_ordered_list = False
                    list_item_count = 0
                else:
                    # Just normal text
                    processed_line = self._process_inline_formatting(processed_line)
                
                # Handle paragraph creation
                if not in_paragraph and processed_line.strip():
                    processed_line = f"<p style='margin: 8px 0;'>{processed_line}"
                    in_paragraph = True
                elif in_paragraph and not processed_line.strip():
                    processed_line = f"{processed_line}</p>"
                    in_paragraph = False
            
            processed_lines.append(processed_line)
            i += 1
        
        # Close any open tags at the end
        if in_unordered_list:
            processed_lines.append("</ul>")
        if in_ordered_list:
            processed_lines.append("</ol>")
        if in_paragraph:
            processed_lines.append("</p>")
        if in_blockquote:
            processed_lines.append("</blockquote>")
        if in_table and table_rows:
            # Render any remaining table
            table_html = ['<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">']
            for row_idx, row in enumerate(table_rows):
                table_html.append('<tr>')
                for cell in row:
                    tag = 'th' if row_idx == 0 else 'td'
                    table_html.append(f'<{tag} style="border: 1px solid #ddd; padding: 8px; text-align: left;">{self._process_inline_formatting(cell)}</{tag}>')
                table_html.append('</tr>')
            table_html.append('</table>')
            processed_lines.append(''.join(table_html))
        
        # Join the lines back together with HTML
        html = "\n".join(processed_lines)
        
        # Clean up any empty paragraphs
        html = re.sub(r'<p>\s*</p>', '', html)
        
        return html

    def _process_inline_formatting(self, text):
        """Process inline Markdown formatting elements with proper spacing"""
        # Process bold (** or __) with space preservation
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)
        
        # Process italic (* or _) with better regex to preserve spacing
        text = re.sub(r'(?<!\*)\*(?!\*|\s)(.*?)(?<!\s)\*(?!\*)', r'<i>\1</i>', text)
        text = re.sub(r'(?<!_)_(?!_|\s)(.*?)(?<!\s)_(?!_)', r'<i>\1</i>', text)
        
        # Ensure space after emphasis markers when needed
        text = re.sub(r'(</[bi]>)(\w)', r'\1 \2', text)  # Add space after tag if followed by word
        text = re.sub(r'(\w)(<[bi]>)', r'\1 \2', text)   # Add space before tag if preceded by word
        
        # Process strikethrough (~~)
        text = re.sub(r'~~(.*?)~~', r'<s>\1</s>', text)
        
        # Process inline code (`code`) - preserve spaces
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        
        # Process links ([text](url))
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
        
        return text
    
    def begin_streaming_response(self):
        """Start a new streaming response session"""
        # Generate a unique ID for this response
        import uuid
        self.current_streaming_id = str(uuid.uuid4())
        
        # Create a placeholder for this response
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Mark this position as the start of streaming content
        self.streaming_start_position = cursor.position()
        
        # Initialize empty response in dictionary
        self.streaming_responses[self.current_streaming_id] = ""
        
        # Add a raw text buffer for accumulating chunks
        self.streaming_raw_text = ""
        
        # Return the streaming ID
        return self.current_streaming_id

    def append_streaming_chunk(self, streaming_id, chunk_text):
        """Add a chunk to an existing streaming response"""
        if streaming_id not in self.streaming_responses:
            return False
        
        # Store the chunk temporarily for display
        self.streaming_responses[streaming_id] = chunk_text
        
        # Add to our accumulated raw text without displaying it yet
        self.streaming_raw_text += chunk_text
        
        # Display just the new chunk
        self.display_raw_streaming_text(streaming_id)
        
        return True
        
    def display_raw_streaming_text(self, streaming_id, clear_only=False, append_only=True):
        """Display the raw streaming text without processing formatting"""
        if streaming_id not in self.streaming_responses:
            return False
            
        # Only get the text if we're not just clearing
        current_text = "" if clear_only else self.streaming_responses[streaming_id]
        
        # Create a normal text format (not bold)
        normal_format = QTextCharFormat()
        normal_format.setForeground(QColor("#24292e"))
        normal_format.setFontWeight(QFont.Normal)  # Explicitly set normal weight
        
        # Get document
        doc = self.document()
        
        if clear_only:
            # If we're clearing, remove all streaming content
            if self.streaming_start_position >= 0:
                cursor = QTextCursor(doc)
                cursor.setPosition(self.streaming_start_position)
                cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
                cursor.removeSelectedText()
                self.streaming_started = False
        elif not self.streaming_started:
            # First chunk - clear any existing content and start fresh
            if self.streaming_start_position >= 0:
                cursor = QTextCursor(doc)
                cursor.setPosition(self.streaming_start_position)
                cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
                cursor.removeSelectedText()
                cursor.insertText(current_text, normal_format)
                self.streaming_started = True
        else:
            # For subsequent chunks, simply replace everything from start position
            # with the complete accumulated text
            if self.streaming_start_position >= 0:
                # Add the new chunk to our accumulated text
                accumulated_text = self.streaming_raw_text
                
                # Replace everything from the start position with the accumulated text
                cursor = QTextCursor(doc)
                cursor.setPosition(self.streaming_start_position)
                cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
                cursor.removeSelectedText()
                cursor.insertText(accumulated_text, normal_format)
        
        # Clear the stored response since we've already displayed it
        if not clear_only:
            self.streaming_responses[streaming_id] = ""
        
        # Update cursor and ensure it's visible
        cursor = QTextCursor(doc)
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
        
        return True

    def update_streaming_response(self, streaming_id):
        """Update the display with the current state of the streaming response"""
        if streaming_id not in self.streaming_responses:
            return False
        
        current_text = self.streaming_responses[streaming_id]
        
        # Find the position where we need to start replacing content
        doc_text = self.toPlainText()
        start_pos = doc_text.rfind("[SYSTEM] Processing request...")
        
        if start_pos >= 0:
            # Position cursor after this system message and its newline
            cursor = QTextCursor(self.document())
            cursor.setPosition(start_pos)
            cursor.movePosition(QTextCursor.EndOfLine)
            cursor.movePosition(QTextCursor.Right, QTextCursor.MoveAnchor, 1)  # Move past the newline
            
            # Select all text from this point to the end
            cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            
            # Remove existing content
            cursor.removeSelectedText()
            
            # Process and insert the text with code blocks and markdown
            if "```" in current_text:
                # Handle code blocks separately
                pattern = r'(```(?:\w*)\n[\s\S]*?\n```)'
                parts = re.split(pattern, current_text)
                for part in parts:
                    if part.strip() and part.startswith("```") and part.endswith("```"):
                        self._insert_code_block(cursor, part)
                    elif part.strip():
                        # Process this part for markdown
                        processed_part = self.process_markdown(part)
                        cursor.insertHtml(processed_part)
                        cursor.insertBlock()  # Add a block after each part
            else:
                # Simple case: no code blocks, just insert as HTML
                processed_text = self.process_markdown(current_text)
                cursor.insertHtml(processed_text)
        else:
            # If we can't find the system message, just append at the end
            cursor = QTextCursor(self.document())
            cursor.movePosition(QTextCursor.End)
            processed_text = self.process_markdown(current_text)
            cursor.insertHtml(processed_text)
        
        # Update cursor and ensure it's visible
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
        
        return True

    def end_streaming_response(self, streaming_id):
        """Finalize a streaming response"""
        if streaming_id not in self.streaming_responses:
            return False
        
        # Use the accumulated raw text for the final formatting
        self.streaming_responses[streaming_id] = self.streaming_raw_text
        
        # Now that streaming is complete, apply full formatting
        self.apply_full_formatting(streaming_id)
        
        # Clean up the tracking
        self.streaming_responses.pop(streaming_id)
        if self.current_streaming_id == streaming_id:
            self.current_streaming_id = None
        self.streaming_raw_text = ""
        
        # Reset streaming state
        self.streaming_started = False
        
        return True
        
    def apply_full_formatting(self, streaming_id):
        """Apply full markdown and code block formatting to the completed response"""
        if streaming_id not in self.streaming_responses:
            return False
        
        # Store the complete text
        complete_text = self.streaming_responses[streaming_id]
        if not complete_text and hasattr(self, 'streaming_raw_text'):
            complete_text = self.streaming_raw_text
        
        # Create a cursor to completely clear the content
        cursor = QTextCursor(self.document())
        
        if self.streaming_start_position >= 0:
            # Use the marked position
            cursor.setPosition(self.streaming_start_position)
            cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
        else:
            # Fallback to old method
            doc_text = self.toPlainText()
            start_pos = doc_text.rfind("[SYSTEM] Processing request...")
            
            if start_pos >= 0:
                # Position cursor after the system message
                cursor.setPosition(start_pos)
                cursor.movePosition(QTextCursor.EndOfLine)
                cursor.movePosition(QTextCursor.Right, QTextCursor.MoveAnchor, 1)  # Move past newline
                
                # Select all text from this point to the end
                cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
                
                # Remove all existing content
                cursor.removeSelectedText()
            else:
                # If we can't find the system message, just go to the end
                cursor.movePosition(QTextCursor.End)
        
        # Now, with all streaming content cleared, apply the full formatting
        if "```" in complete_text:
            # Split by code blocks
            pattern = r'(```(?:\w*)\n[\s\S]*?\n```)'
            parts = re.split(pattern, complete_text)
            
            for part in parts:
                if part.strip() and part.startswith("```") and part.endswith("```"):
                    # This is a code block - format it properly
                    self._insert_code_block(cursor, part)
                elif part.strip():
                    # Normal text part - process for markdown
                    processed_part = self.process_markdown(part)
                    cursor.insertHtml(processed_part)
                    cursor.insertBlock()  # Add a block after each processed part
        else:
            # No code blocks, apply markdown processing
            processed_text = self.process_markdown(complete_text)
            cursor.insertHtml(processed_text)
            cursor.insertBlock()  # Add a block after content
        
        # Update cursor and ensure it's visible
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
        
        # Clean up tracking variables
        self.streaming_started = False
        self.streaming_start_position = -1  # Reset the position
        
        return True
    
    def insertHtml(self, html):
        """Handle streaming content with proper text flow"""
        cursor = self.textCursor()
        
        # Get current position and text
        cursor.movePosition(QTextCursor.End)
        
        # For code blocks, use the existing logic
        if "```" in html:
            # (existing code block handling)
            return
        
        # KEY FIX: For normal text, don't create new blocks or paragraphs
        # Strip HTML tags for safety
        plain_text = re.sub(r'<[^>]*>', '', html)
        
        # Find the current paragraph and text
        cursor.movePosition(QTextCursor.StartOfBlock)
        has_text = not cursor.atEnd()
        
        # Format for text insertion
        format = QTextCharFormat()
        format.setForeground(QColor("#24292e"))
        
        # If we're continuing a paragraph, we want to append in-place
        # rather than creating a new line or paragraph
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(plain_text, format)
        
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
        
        # If we have a pending code block, keep accumulating
        if self.in_code_block:
            self.code_block_buffer += html
            return
        
        # For normal text in streaming mode, accumulate and insert
        # Strip HTML tags for safety
        plain_text = re.sub(r'<[^>]*>', '', html)
        self.streaming_text += plain_text
        
        # Update the display with the current accumulated text
        # Find the start of our streaming text
        if cursor.block().text() == "":
            # If we're at a blank line, start a new block
            format = QTextCharFormat()
            format.setForeground(QColor("#24292e"))
            cursor.insertText(self.streaming_text, format)
        else:
            # Otherwise, replace the current line with our accumulated text
            cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
            format = QTextCharFormat()
            format.setForeground(QColor("#24292e"))
            cursor.insertText(self.streaming_text, format)
        
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
    
    def setHtml(self, html):
        # Use a regex that excludes tags starting with "!--msgID:" (our markers)
        plain_text = re.sub(r'<(?!!--msgID:).*?>', '', html)
        self.setPlainText(plain_text)

    
    def _insert_code_block(self, cursor, code_block_text):
        """Insert a formatted code block at the cursor position"""
        # Extract language and code
        if "\n" in code_block_text:
            first_line = code_block_text.split("\n", 1)[0]
            language = first_line.replace("```", "").strip() or "python"
            code = code_block_text.split("\n", 1)[1].rsplit("```", 1)[0]
        else:
            language = "python"
            code = code_block_text.replace("```", "").strip()
        
        # Insert a divider before the code block
        cursor.insertBlock()
        
        # Create a background block format for the code
        block_format = QTextBlockFormat()
        block_format.setBackground(QColor("#F6F8FA"))
        block_format.setLeftMargin(10)
        block_format.setRightMargin(10)
        block_format.setTopMargin(5)
        block_format.setBottomMargin(5)
        cursor.setBlockFormat(block_format)
        
        # Insert the language label
        lang_format = QTextCharFormat()
        lang_format.setForeground(QColor("#6A737D"))
        lang_format.setFontWeight(QFont.Bold)
        lang_format.setFontFamily("Segoe UI")
        cursor.insertText(f"[{language}]", lang_format)
        cursor.insertBlock()
        
        # Insert the code with monospace font
        code_format = QTextCharFormat()
        code_format.setFontFamily("Consolas, Liberation Mono, Menlo, monospace")
        code_format.setFontPointSize(9)
        
        # We'll highlight each line according to the language
        highlighter = CodeHighlighter(None, language)
        
        # Insert code line by line with highlighting
        lines = code.split('\n')
        for i, line in enumerate(lines):
            # Apply syntax highlighting
            if line.strip():
                current_pos = 0
                # Find matches for each rule
                matches = []
                for rule_pattern, rule_format in highlighter.highlighting_rules:
                    for match in rule_pattern.finditer(line):
                        matches.append((match.start(), match.end(), rule_format))
                
                # Sort matches by start position
                matches.sort(key=lambda x: x[0])
                
                # Apply formatting and insert text
                for start, end, rule_format in matches:
                    # Insert text before match with default format
                    if start > current_pos:
                        cursor.insertText(line[current_pos:start], code_format)
                    
                    # Insert matched text with special format
                    format_to_use = QTextCharFormat(code_format)
                    format_to_use.setForeground(rule_format.foreground())
                    if rule_format.fontWeight() > QFont.Normal:
                        format_to_use.setFontWeight(rule_format.fontWeight())
                    cursor.insertText(line[start:end], format_to_use)
                    current_pos = end
                
                # Insert any remaining text
                if current_pos < len(line):
                    cursor.insertText(line[current_pos:], code_format)
            else:
                cursor.insertText(line, code_format)
            
            # Don't add a new line after the last line
            if i < len(lines) - 1:
                cursor.insertBlock()
                cursor.setBlockFormat(block_format)
        
        # Add a block for the copy button
        cursor.insertBlock()
        
        # Reset the block format
        normal_block = QTextBlockFormat()
        cursor.setBlockFormat(normal_block)
        
        # Add copy button
        copy_format = QTextCharFormat()
        copy_format.setForeground(QColor("#0366d6"))
        copy_format.setAnchor(True)
        # Store the code to copy in the URL with a special prefix
        copy_format.setAnchorHref(f"copy:{code}")
        cursor.insertText("[Copy Code]", copy_format)
        cursor.insertBlock()
    
    def mousePressEvent(self, event):
        """Track mouse press events for copy link clicks and URL navigation"""
        if event.button() == Qt.LeftButton:
            cursor = self.cursorForPosition(event.pos())
            char_format = cursor.charFormat()
            if char_format.isAnchor():
                url = char_format.anchorHref()
                if url.startswith("copy:"):
                    # Extract the code to copy
                    code = url[5:]
                    
                    # Copy to clipboard
                    clipboard = QApplication.clipboard()
                    mime_data = QMimeData()
                    mime_data.setText(code)
                    clipboard.setMimeData(mime_data)
                    
                    # Add a status message
                    self.append("\n[SYSTEM] Code copied to clipboard")
                    
                    # Prevent default handling
                    event.accept()
                    return
                elif url:
                    # Handle regular URL
                    from PyQt5.QtCore import QUrl
                    from PyQt5.QtGui import QDesktopServices
                    QDesktopServices.openUrl(QUrl(url))
                    event.accept()
                    return
        
        super().mousePressEvent(event)
    
    def set_dark_mode(self, enabled):
        """Toggle between dark and light mode"""
        if enabled:
            # Dark mode
            self.setStyleSheet("""
                QTextEdit {
                    background-color: #0d1117;
                    color: #e1e4e8;
                    border: 1px solid #30363d;
                    border-radius: 6px;
                    selection-background-color: #3b4a63;
                    padding: 5px;
                }
            """)
        else:
            # Light mode
            self.setStyleSheet("""
                QTextEdit {
                    background-color: white;
                    color: #24292e;
                    border: 1px solid #e1e4e8;
                    border-radius: 6px;
                    selection-background-color: #b3d7ff;
                    padding: 5px;
                }
            """)
            
    def highlight_search_terms(self, search_term):
        """Highlight all occurrences of the search term in the chat display and return positions"""
        if not search_term:
            logging.debug("No search term provided for highlighting")
            return []
            
        # Store the search term
        self.current_search_term = search_term
            
        # Clear any existing highlighting first
        self.clear_search_highlights()
        
        # Create a list to store the positions of matches
        match_positions = []
        
        # Get the document
        document = self.document()
        text = document.toPlainText()
        
        logging.debug(f"Searching for term '{search_term}' in text of length {len(text)}")
        
        # Case insensitive search
        search_term = search_term.lower()
        
        # Find all occurrences
        start_pos = 0
        while start_pos < len(text):
            pos = text.lower().find(search_term, start_pos)
            if pos == -1:
                break
            
            # Create a cursor for this position
            cursor = QTextCursor(document)
            cursor.setPosition(pos)
            cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, len(search_term))
            
            # Apply highlighting format
            highlight_format = QTextCharFormat()
            highlight_format.setBackground(QColor("yellow"))
            highlight_format.setForeground(QColor("black"))
            cursor.mergeCharFormat(highlight_format)
            
            # Store the position of this match
            match_positions.append(pos)
            
            # Move to the next position
            start_pos = pos + len(search_term)
        
        # Store the match positions and current index
        self.search_match_positions = match_positions
        self.current_search_match_index = -1  # Start before the first match
        
        logging.debug(f"Found {len(match_positions)} matches for '{search_term}'")
        
        return match_positions

    def clear_search_highlights(self):
        """Clear all search highlighting"""
        # Create default format
        default_format = QTextCharFormat()
        default_format.setBackground(QColor("transparent"))
        
        # Reset the whole document format
        cursor = QTextCursor(self.document())
        cursor.select(QTextCursor.Document)
        cursor.mergeCharFormat(default_format)
        
        # Reset search state
        self.search_match_positions = []
        self.current_search_match_index = -1

    def goto_next_match(self):
        """Move to the next search match"""
        if not hasattr(self, 'search_match_positions') or not self.search_match_positions:
            logging.debug("No search matches to navigate to")
            return False
        # Increment index and wrap around if needed
        self.current_search_match_index = (self.current_search_match_index + 1) % len(self.search_match_positions)
        logging.debug(f"Navigating to next match: {self.current_search_match_index + 1} of {len(self.search_match_positions)}")
        return self.scroll_to_match(self.current_search_match_index)

    def goto_prev_match(self):
        """Move to the previous search match"""
        if not hasattr(self, 'search_match_positions') or not self.search_match_positions:
            return False
        if self.current_search_match_index <= 0:
            self.current_search_match_index = len(self.search_match_positions) - 1
        else:
            self.current_search_match_index -= 1
        return self.scroll_to_match(self.current_search_match_index)

    def scroll_to_match(self, index):
        """Scroll to the match at the given index and highlight it specially"""
        if not hasattr(self, 'search_match_positions') or not self.search_match_positions:
            logging.debug("No search match positions available")
            return False
            
        if index >= len(self.search_match_positions):
            logging.debug(f"Invalid match index: {index}, max is {len(self.search_match_positions)-1}")
            return False
            
        # Get the position of the match
        pos = self.search_match_positions[index]
        
        # Ensure we have the current search term
        if not hasattr(self, 'current_search_term') or not self.current_search_term:
            logging.debug("No current search term available")
            self.current_search_term = "test"  # Fallback to avoid errors
        
        # Create a cursor and move to the match
        cursor = QTextCursor(self.document())
        cursor.setPosition(pos)
        
        # Set this cursor as the current cursor
        self.setTextCursor(cursor)
        
        # Ensure the cursor is visible
        self.ensureCursorVisible()
        
        # Create a special format for the current match
        cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, len(self.current_search_term))
        
        current_match_format = QTextCharFormat()
        current_match_format.setBackground(QColor("orange"))
        current_match_format.setForeground(QColor("black"))
        cursor.mergeCharFormat(current_match_format)
        
        logging.debug(f"Scrolled to match at position {pos} (term: '{self.current_search_term}')")
        
        return True
        
    def goto_match_by_index(self, index):
        """Scroll to the search match at the given index from the precomputed match positions."""
        if not self.search_match_positions:
            logging.debug("No search match positions available.")
            return False
        if index < len(self.search_match_positions):
            pos = self.search_match_positions[index]
            cursor = self.textCursor()
            cursor.setPosition(pos)
            self.setTextCursor(cursor)
            self.ensureCursorVisible()
            return True
        else:
            logging.debug(f"Requested match index {index} is out of range.")
            return False

    def goto_match_by_message_id(self, message_id):
        # Assume that each message was appended with a unique marker like "msgID:<message_id>"
        target_marker = f"msgID:{message_id}"
        # Start a new cursor at the beginning of the document
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.Start)
        # Use QTextEdit.find() to look for the marker in the document
        found = self.find(target_marker)
        if found:
            # The find() method moves the cursor, so update the view accordingly
            self.setTextCursor(self.textCursor())
            self.ensureCursorVisible()
        else:
            # Fallback if the marker is not found: go to the first match
            self.goto_match_by_message_id()

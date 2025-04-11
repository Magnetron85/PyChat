# simple_rag_ui.py
import os
import logging
from typing import List, Dict, Any, Optional

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QListWidget, QListWidgetItem, QFileDialog,
                            QTextEdit, QCheckBox, QMessageBox, QProgressBar,
                            QDialog, QDialogButtonBox, QFormLayout, QComboBox, QInputDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSettings
from PyQt5.QtGui import QFont, QColor

from simple_rag_manager import SimpleRAGManager
from knowledge_base_worker import KnowledgeBaseWorker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGDocumentItem(QListWidgetItem):
    """Custom list widget item for RAG documents"""
    
    def __init__(self, doc_id: str, filename: str, metadata: Dict[str, Any]):
        super().__init__()
        self.doc_id = doc_id
        self.filename = filename
        self.metadata = metadata
        
        # Set display text
        self.setText(filename)
        
        # Add tooltip with metadata
        tooltip = f"Document: {filename}\n"
        if metadata.get("format"):
            tooltip += f"Format: {metadata.get('format')}\n"
        if metadata.get("line_count"):
            tooltip += f"Lines: {metadata.get('line_count')}\n"
            
        self.setToolTip(tooltip)

class RAGContextDisplay(QTextEdit):
    """Text display for showing retrieved context"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText("Retrieved context will appear here")
        self.setAcceptRichText(True)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
        """)
    
    def set_context(self, chunks: List[Dict[str, Any]]):
        """Display retrieved context chunks"""
        self.clear()
        
        if not chunks:
            self.setPlainText("No relevant context found.")
            return
        
        html = "<html><body style='font-family: Segoe UI, Arial, sans-serif;'>"
        
        for i, chunk in enumerate(chunks):
            filename = chunk.get("filename", "Unknown")
            content = chunk.get("content", "")
            
            html += f"<div style='margin-bottom: 10px; padding: 8px; background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 4px;'>"
            html += f"<div style='font-weight: bold; color: #555555; margin-bottom: 5px;'>From: {filename}</div>"
            html += f"<div style='color: #333333;'>{content}</div>"
            html += "</div>"
        
        html += "</body></html>"
        self.setHtml(html)

class RAGSettings(QDialog):
    """Dialog for configuring RAG parameters"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RAG Settings")
        self.init_ui()
    
    # In simple_rag_ui.py, modify the RAGSettings class initialization:

    def init_ui(self):
        layout = QVBoxLayout()
        
        form = QFormLayout()
        
        # Number of chunks to retrieve
        self.chunks_input = QComboBox()
        for i in range(1, 11):
            self.chunks_input.addItem(str(i), i)
        self.chunks_input.setCurrentIndex(9)  # Default to 10 chunks (index 9)
        form.addRow("TF-IDF context chunks:", self.chunks_input)
        
        # NEW: Add SBERT re-ranking chunks setting
        self.sbert_chunks_input = QComboBox()
        for i in range(1, 11):
            self.sbert_chunks_input.addItem(str(i), i)
        self.sbert_chunks_input.setCurrentIndex(2)  # Default to 3 chunks (index 2)
        form.addRow("SBERT re-ranking chunks:", self.sbert_chunks_input)
        
        # Prepend or append context
        self.context_position = QComboBox()
        self.context_position.addItem("Before query", "prepend")
        self.context_position.addItem("After query", "append")
        self.context_position.setCurrentIndex(0)  # Default to prepend
        form.addRow("Context position:", self.context_position)
        
        # Button for dialog
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        layout.addLayout(form)
        layout.addWidget(self.button_box)
        
        self.setLayout(layout)

    def get_settings(self):
        """Get settings from dialog"""
        return {
            "chunks": int(self.chunks_input.currentData()),
            "sbert_chunks": int(self.sbert_chunks_input.currentData()),
            "position": self.context_position.currentData()
        }

class RAGPanel(QWidget):
    """Main RAG panel for PyChat integration"""
    
    context_retrieved = pyqtSignal(str, str)  # Emitted when context is retrieved (orig_query, query_with_context)
    document_added = pyqtSignal()  # Emitted when a document is added
    document_removed = pyqtSignal()  # Emitted when a document is removed
    
    # In RAGPanel's __init__ method, add these lines:

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rag_manager = SimpleRAGManager()
        self.init_ui()
        
        # Default settings
        self.chunks_to_retrieve = 10  # Changed from 3 to 10
        self.sbert_chunks_to_retrieve = 3  # New setting for SBERT re-ranking
        self.context_position = "prepend"  # 'prepend' or 'append'
        
        # Load settings from QSettings if available
        settings = QSettings("AI Chat App", "MultiProviderChat")
        self.chunks_to_retrieve = settings.value("rag/chunks_to_retrieve", 10, type=int)
        self.sbert_chunks_to_retrieve = settings.value("rag/sbert_chunks_to_retrieve", 3, type=int)
        self.context_position = settings.value("rag/context_position", "prepend", type=str)
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Knowledge Base selector
        kb_header = QHBoxLayout()
        kb_label = QLabel("Knowledge Base:")
        self.kb_dropdown = QComboBox()
        self.refresh_kb_btn = QPushButton("⟳")
        self.refresh_kb_btn.setToolTip("Refresh knowledge bases")
        self.refresh_kb_btn.setMaximumWidth(30)
        self.refresh_kb_btn.clicked.connect(self.load_knowledge_bases)

        self.create_kb_btn = QPushButton("+")
        self.create_kb_btn.setToolTip("Create new knowledge base")
        self.create_kb_btn.setMaximumWidth(30)
        self.create_kb_btn.clicked.connect(self.create_knowledge_base)

        self.remove_kb_btn = QPushButton("-")
        self.remove_kb_btn.setToolTip("Remove knowledge base")
        self.remove_kb_btn.setMaximumWidth(30)
        self.remove_kb_btn.clicked.connect(self.remove_knowledge_base)

        kb_header.addWidget(kb_label)
        kb_header.addWidget(self.kb_dropdown, 1)
        kb_header.addWidget(self.refresh_kb_btn)
        kb_header.addWidget(self.create_kb_btn)
        kb_header.addWidget(self.remove_kb_btn)

        layout.addLayout(kb_header)
        
        # Document section
        doc_section = QWidget()
        doc_layout = QVBoxLayout(doc_section)
        
        # Document header
        doc_header = QHBoxLayout()
        doc_label = QLabel("Documents")
        doc_label.setStyleSheet("font-weight: bold;")
        
        # Buttons for document management
        self.add_doc_btn = QPushButton("Add")
        self.add_doc_btn.setToolTip("Add document to knowledge base")
        self.add_doc_btn.clicked.connect(self.add_document)
        
        self.remove_doc_btn = QPushButton("Remove")
        self.remove_doc_btn.setToolTip("Remove selected document")
        self.remove_doc_btn.clicked.connect(self.remove_document)
        
        doc_header.addWidget(doc_label)
        doc_header.addStretch()
        doc_header.addWidget(self.add_doc_btn)
        doc_header.addWidget(self.remove_doc_btn)
        
        doc_layout.addLayout(doc_header)
        
        # Document list
        self.doc_list_widget = QListWidget()
        self.doc_list_widget.setStyleSheet("""
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:selected {
                background-color: #e0e0e0;
                color: #333;
            }
        """)
        self.doc_list_widget.setSelectionMode(QListWidget.SingleSelection)
        
        doc_layout.addWidget(self.doc_list_widget)
        
        layout.addWidget(doc_section)
        
        # Context section
        context_widget = QWidget()
        context_layout = QVBoxLayout(context_widget)
        
        # Context header
        context_header = QHBoxLayout()
        context_label = QLabel("Context Preview")
        context_label.setStyleSheet("font-weight: bold;")
        
        # RAG controls
        self.rag_enabled = QCheckBox("Enable RAG")
        self.rag_enabled.setChecked(True)
        
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.show_settings)
        
        context_header.addWidget(context_label)
        context_header.addStretch()
        context_header.addWidget(self.rag_enabled)
        context_header.addWidget(self.settings_btn)
        
        context_layout.addLayout(context_header)
        
        # Context display
        self.context_display = RAGContextDisplay()
        context_layout.addWidget(self.context_display)
        
        layout.addWidget(context_widget)
        
        self.setLayout(layout)
        
        # Load knowledge bases after UI is set up
        self.load_knowledge_bases()
        self.kb_dropdown.currentIndexChanged.connect(self.on_knowledge_base_changed)
    
    def load_knowledge_bases(self):
        """Load knowledge bases into dropdown"""
        # Remember current selection if any
        current_kb = None
        if self.kb_dropdown.count() > 0:
            current_kb = self.kb_dropdown.currentData()
        
        self.kb_dropdown.clear()
        
        knowledge_bases = self.rag_manager.get_knowledge_bases()
        for kb in knowledge_bases:
            self.kb_dropdown.addItem(kb["name"], kb["id"])
        
        # If no knowledge bases or current_kb not found, select the first one if available
        if self.kb_dropdown.count() > 0:
            # Restore selection if possible
            if current_kb:
                index_found = False
                for i in range(self.kb_dropdown.count()):
                    if self.kb_dropdown.itemData(i) == current_kb:
                        self.kb_dropdown.setCurrentIndex(i)
                        index_found = True
                        break
                
                # If previous selection not found, select the first one
                if not index_found:
                    self.kb_dropdown.setCurrentIndex(0)
            else:
                # No previous selection, select the first one
                self.kb_dropdown.setCurrentIndex(0)
            
            # Make sure the selected knowledge base is set in the manager
            kb_id = self.kb_dropdown.itemData(self.kb_dropdown.currentIndex())
            self.set_knowledge_base_async(kb_id)
    
    # Fix the layout issue by modifying the set_knowledge_base_async method in simple_rag_ui.py

    def set_knowledge_base_async(self, kb_id):
        """Set the knowledge base asynchronously with progress indication"""
        # Terminate any existing worker
        if hasattr(self, 'kb_worker') and self.kb_worker.isRunning():
            self.kb_worker.terminate()
            self.kb_worker.wait()
        
        # Check if we already have a progress bar
        if hasattr(self, 'progress'):
            # Remove existing progress bar before creating a new one
            self.layout().removeWidget(self.progress)
            self.progress.deleteLater()
        
        # Add progress bar to the UI
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 0)  # Indeterminate progress
        self.progress.setFormat("Loading knowledge base...")
        self.layout().addWidget(self.progress)
        
        # Define callback for when operation completes
        def on_kb_loaded(success, message):
            # Only remove progress bar if it still exists
            if hasattr(self, 'progress'):
                try:
                    self.layout().removeWidget(self.progress)
                    self.progress.deleteLater()
                    delattr(self, 'progress')
                except Exception as e:
                    logger.error(f"Error removing progress bar: {str(e)}")
            
            if success:
                # Load documents for the selected knowledge base
                self.load_documents()
                self.update_context_preview()
            else:
                QMessageBox.warning(
                    self,
                    "Knowledge Base Error",
                    f"Error loading knowledge base: {message}"
                )
        
        # Use the asynchronous method
        self.rag_manager.set_knowledge_base_async(kb_id, on_kb_loaded)
    
    def load_documents(self):
        """Load documents from the RAG manager"""
        self.doc_list_widget.clear()
        
        documents = self.rag_manager.get_all_documents()
        for doc in documents:
            item = RAGDocumentItem(
                doc_id=doc["id"],
                filename=doc["filename"],
                metadata=doc["metadata"]
            )
            self.doc_list_widget.addItem(item)
    
    def on_knowledge_base_changed(self, index):
        """Handle knowledge base selection change"""
        if index >= 0:
            kb_id = self.kb_dropdown.itemData(index)
            self.set_knowledge_base_async(kb_id)
    
    def remove_knowledge_base(self):
        """Remove the selected knowledge base"""
        # Get selected knowledge base
        current_index = self.kb_dropdown.currentIndex()
        if current_index < 0:
            return
            
        kb_id = self.kb_dropdown.itemData(current_index)
        kb_name = self.kb_dropdown.itemText(current_index)
        
        # Don't allow deleting the default knowledge base
        if kb_id == "default":
            QMessageBox.warning(
                self,
                "Cannot Remove",
                "The default knowledge base cannot be removed."
            )
            return
        
        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            "Remove Knowledge Base",
            f"Are you sure you want to remove the knowledge base '{kb_name}'?\n\nThis will permanently delete all documents in this knowledge base.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            success = self.rag_manager.delete_knowledge_base(kb_id)
            if success:
                # Reload knowledge bases
                self.load_knowledge_bases()
                QMessageBox.information(
                    self,
                    "Knowledge Base Removed",
                    f"Successfully removed knowledge base '{kb_name}'."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Removal Failed",
                    f"Failed to remove knowledge base '{kb_name}'. See logs for details."
                )
    
    def add_document(self):
        """Add a document to the RAG system"""
        # Open file dialog
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Document(s)",
            "",
            "All Supported Files (*.txt *.md *.pdf *.docx);;Text Documents (*.txt *.md);;Word Documents (*.docx);;PDF Documents (*.pdf);;All Files (*)"
        )
        
        if not file_paths:
            return
        
        # Add progress bar to the UI
        self.progress = QProgressBar(self)
        self.progress.setRange(0, len(file_paths))
        self.progress.setValue(0)
        self.layout().addWidget(self.progress)
        
        # Process each file asynchronously
        self.remaining_files = len(file_paths)
        self.success_count = 0
        
        for file_path in file_paths:
            # Progress callback for individual document processing
            def update_progress(current, total):
                if hasattr(self, 'progress'):
                    # Map the sub-progress to our overall progress
                    file_progress = current / total
                    file_index = len(file_paths) - self.remaining_files
                    overall_progress = file_index + file_progress
                    self.progress.setValue(int(overall_progress))
                    self.progress.setFormat(f"Processing {os.path.basename(file_path)} ({current}/{total})")
            
            # Completion callback for individual document
            def on_document_added(success, message):
                self.remaining_files -= 1
                if success:
                    self.success_count += 1
                
                # If all files processed, update UI
                if self.remaining_files == 0:
                    # Remove progress bar
                    if hasattr(self, 'progress'):
                        self.layout().removeWidget(self.progress)
                        self.progress.deleteLater()
                        delattr(self, 'progress')
                    
                    # Reload documents
                    self.load_documents()
                    
                    # Show result message
                    if self.success_count > 0:
                        QMessageBox.information(
                            self,
                            "Documents Added",
                            f"Successfully added {self.success_count} document(s) to the knowledge base."
                        )
                        self.document_added.emit()
                    else:
                        QMessageBox.warning(
                            self,
                            "Adding Failed",
                            "Failed to add documents. See logs for details."
                        )
            
            # Start asynchronous document addition
            self.rag_manager.add_document_async(
                file_path,
                progress_callback=update_progress,
                completion_callback=on_document_added
            )
    
    def remove_document(self):
        """Remove selected document from the RAG system"""
        # Get selected item
        item = self.doc_list_widget.currentItem()
        if not item or not isinstance(item, RAGDocumentItem):
            return
        
        reply = QMessageBox.question(
            self,
            "Remove Document",
            f"Are you sure you want to remove '{item.filename}' from the knowledge base?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            success = self.rag_manager.delete_document(item.doc_id)
            if success:
                self.doc_list_widget.takeItem(self.doc_list_widget.row(item))
                self.document_removed.emit()
    
    def create_knowledge_base(self):
        """Create a new knowledge base"""
        name, ok = QInputDialog.getText(
            self, 
            "Create Knowledge Base",
            "Enter a name for the new knowledge base:"
        )
        
        if ok and name:
            description, ok = QInputDialog.getText(
                self,
                "Knowledge Base Description",
                "Enter a description (optional):"
            )
            
            if ok:  # User might cancel the description dialog
                kb_id = self.rag_manager.create_knowledge_base(name, description)
                if kb_id:
                    self.load_knowledge_bases()
                    
                    # Select the newly created knowledge base
                    for i in range(self.kb_dropdown.count()):
                        if self.kb_dropdown.itemData(i) == kb_id:
                            self.kb_dropdown.setCurrentIndex(i)
                            break
    
    # In the update_context_preview method, update the call to retrieve_relevant:

    def update_context_preview(self):
        """Update context preview with a sample query"""
        if self.rag_enabled.isChecked():
            sample_query = "Help me understand this topic."
            
            # Show loading state
            self.context_display.setPlainText("Loading context preview...")
            
            # Retrieve context asynchronously
            def on_context_retrieved(chunks):
                self.context_display.set_context(chunks)
            
            # Use a QTimer to give UI time to update
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, lambda: on_context_retrieved(
                self.rag_manager.retrieve_relevant(sample_query, self.chunks_to_retrieve, self.sbert_chunks_to_retrieve)
            ))
        else:
            self.context_display.clear()
            self.context_display.setPlainText("RAG is disabled. Enable it to use document context.")
    
    def get_context_for_query(self, query: str, show_context: bool = True) -> str:
        """
        Get context-enhanced query string
        
        Args:
            query: Original query from user
            show_context: Whether to show context in the prompt to the user
            
        Returns:
            query_with_context: Query enhanced with retrieved context
        """
        if not self.rag_enabled.isChecked():
            return query
        
        # Get currently selected knowledge base name
        kb_name = self.kb_dropdown.currentText()
        
        # Retrieve relevant chunks - pass both TF-IDF and SBERT chunk settings
        chunks = self.rag_manager.retrieve_relevant(
            query, 
            self.chunks_to_retrieve, 
            self.sbert_chunks_to_retrieve
        )
        
        # Update context display
        self.context_display.set_context(chunks)
        
        if not chunks:
            return query
        
        # Format context
        context = ""
        for i, chunk in enumerate(chunks):
            # Add separator between chunks
            if i > 0:
                context += "\n---\n"
            
            # Add chunk content with source info
            filename = chunk.get("filename", "Unknown")
            content = chunk.get("content", "")
            similarity = chunk.get("similarity", 0)
            
            context += f"[{filename} - similarity: {similarity:.2f}]: {content}"
        
        # Create the full context-enhanced query regardless of show_context setting
        if self.context_position == "prepend":
            query_with_context = f"Here is some context from knowledge base '{kb_name}' to help answer the question:\n\n{context}\n\nUser's question: {query}"
        else:  # append
            query_with_context = f"User's question: {query}\n\nHere is some context from knowledge base '{kb_name}' that might help:\n\n{context}"
        
        # Emit signal with original query and the constructed query_with_context
        # This ensures the calling function knows both versions
        self.context_retrieved.emit(query, query_with_context)
        
        # Return the fully enhanced query regardless of show_context
        # The calling function will decide what to display to the user
        return query_with_context
    
    def check_rag_documents(self):
        """Check if there are documents in the current knowledge base"""
        # Get currently selected knowledge base
        kb_id = self.kb_dropdown.currentData() if hasattr(self, 'kb_dropdown') else None
        
        if kb_id:
            # Set the knowledge base in the RAG manager
            self.rag_manager.set_knowledge_base(kb_id)
        
        # Check if there are documents
        docs = self.rag_manager.get_all_documents()
        if not docs and self.rag_enabled.isChecked():
            # No documents but RAG is enabled
            return False
        return True
        
    def show_settings(self):
        """Show RAG settings dialog"""
        dialog = RAGSettings(self)
        dialog.chunks_input.setCurrentIndex(self.chunks_to_retrieve - 1)
        dialog.sbert_chunks_input.setCurrentIndex(self.sbert_chunks_to_retrieve - 1)
        dialog.context_position.setCurrentIndex(0 if self.context_position == "prepend" else 1)
        
        if dialog.exec_():
            settings = dialog.get_settings()
            self.chunks_to_retrieve = settings["chunks"]
            self.sbert_chunks_to_retrieve = settings["sbert_chunks"]
            self.context_position = settings["position"]
            
            # Save settings
            settings_obj = QSettings("AI Chat App", "MultiProviderChat")
            settings_obj.setValue("rag/chunks_to_retrieve", self.chunks_to_retrieve)
            settings_obj.setValue("rag/sbert_chunks_to_retrieve", self.sbert_chunks_to_retrieve)
            settings_obj.setValue("rag/context_position", self.context_position)
            
            self.update_context_preview()
    
    def closeEvent(self, event):
        """Handle closing event properly"""
        try:
            # Wait for any running workers
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, QThread) and attr.isRunning():
                    logger.info(f"Terminating thread: {attr_name}")
                    attr.terminate()
                    attr.wait(100)  # Wait up to 100ms for thread to finish
        except Exception as e:
            logger.error(f"Error in closeEvent: {str(e)}")
        
        # Call parent implementation
        super().closeEvent(event)
    
    def terminate_workers(self):
        """Terminate all running worker threads"""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, QThread) and attr.isRunning():
                try:
                    attr.terminate()
                    attr.wait()
                except:
                    pass
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            # Ensure all worker threads are stopped
            for attr_name in dir(self):
                try:
                    attr = getattr(self, attr_name)
                    if isinstance(attr, QThread) and attr.isRunning():
                        logger.info(f"Terminating thread in __del__: {attr_name}")
                        attr.terminate()
                        attr.wait(100)  # Wait up to 100ms
                except:
                    pass
        except:
            pass  # Ignore any errors during destruction
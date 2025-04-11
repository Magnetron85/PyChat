# knowledge_base_worker.py
from PyQt5.QtCore import QThread, pyqtSignal
import logging

class KnowledgeBaseWorker(QThread):
    """Worker thread for asynchronous knowledge base operations"""
    
    progress_updated = pyqtSignal(int, int)  # (current, total)
    operation_completed = pyqtSignal(bool, str)  # (success, message)
    vectors_loaded = pyqtSignal(object, object, object)  # (tfidf_vectors, sbert_vectors, vector_mapping)
    
    def __init__(self, rag_manager, operation, **kwargs):
        """
        Initialize the worker
        
        Args:
            rag_manager: SimpleRAGManager instance
            operation: String indicating the operation type ('load_vectors', 'add_document', etc.)
            **kwargs: Additional arguments for the specific operation
        """
        super().__init__()
        self.rag_manager = rag_manager
        self.operation = operation
        self.kwargs = kwargs
        
    def run(self):
        """Execute the requested operation"""
        try:
            if self.operation == 'load_vectors':
                kb_id = self.kwargs.get('kb_id')
                tfidf_vectors, sbert_vectors, vector_mapping = self.rag_manager._load_vectors(kb_id)
                self.vectors_loaded.emit(tfidf_vectors, sbert_vectors, vector_mapping)
                self.operation_completed.emit(True, f"Vectors loaded for knowledge base '{kb_id}'")
                
            elif self.operation == 'add_document':
                file_path = self.kwargs.get('file_path')
                doc_id = self.rag_manager.add_document(file_path)
                
                if doc_id:
                    self.operation_completed.emit(True, f"Document added successfully")
                else:
                    self.operation_completed.emit(False, "Failed to add document")
                    
            elif self.operation == 'compute_embeddings':
                chunks = self.kwargs.get('chunks', [])
                total = len(chunks)
                
                # Process in smaller batches for progress reporting
                batch_size = 5
                embeddings = []
                
                for i in range(0, total, batch_size):
                    batch = chunks[i:i + batch_size]
                    batch_embeddings = self.rag_manager.sbert_model.encode(batch, convert_to_numpy=True)
                    embeddings.append(batch_embeddings)
                    
                    # Report progress
                    current = min(i + batch_size, total)
                    self.progress_updated.emit(current, total)
                
                # Return the completed embeddings
                import numpy as np
                if embeddings:
                    result = np.vstack(embeddings)
                    self.operation_completed.emit(True, "Embeddings computed successfully")
                    return result
                
                self.operation_completed.emit(False, "No embeddings computed")
                return np.array([])
                
            elif self.operation == 'load_knowledge_base':
                kb_id = self.kwargs.get('kb_id')
                self.rag_manager._load_documents()
                self.operation_completed.emit(True, f"Knowledge base '{kb_id}' loaded")
                
        except Exception as e:
            logging.error(f"Error in KnowledgeBaseWorker: {str(e)}")
            self.operation_completed.emit(False, f"Operation failed: {str(e)}")
            
    def __del__(self):
        """Make sure the thread is stopped properly when the object is destroyed"""
        self.wait()  # Wait for the thread to finish
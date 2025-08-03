"""
Utility functions for PDF processing and text chunking
"""
import os
from typing import Dict, Any
import requests
import fitz  # PyMuPDF
from typing import List, Tuple
import tempfile
from urllib.parse import urlparse
import re

class PDFProcessor:
    def __init__(self):
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit
    
    async def extract_text_from_url(self, pdf_url: str) -> Tuple[str, str]:
        """
        Download PDF from URL and extract text
        Returns (text_content, pdf_title)
        """
        try:
            # Validate URL
            parsed_url = urlparse(pdf_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid PDF URL")
            
            # Download PDF
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(pdf_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Check file size
            if len(response.content) > self.max_file_size:
                raise ValueError(f"PDF file too large. Max size: {self.max_file_size} bytes")
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type:
                print(f"Warning: Content type is {content_type}, proceeding anyway")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            try:
                # Extract text using PyMuPDF
                text_content, pdf_title = self._extract_text_from_file(temp_path)
                return text_content, pdf_title
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except requests.RequestException as e:
            raise Exception(f"Error downloading PDF: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def _extract_text_from_file(self, file_path: str) -> Tuple[str, str]:
        """Extract text from PDF file using PyMuPDF"""
        try:
            doc = fitz.open(file_path)
            text_content = ""
            pdf_title = "Unknown Document"
            
            # Try to get title from metadata
            metadata = doc.metadata
            if metadata and metadata.get('title'):
                pdf_title = metadata['title']
            else:
                # Use filename as fallback
                pdf_title = os.path.basename(file_path)
            
            # Extract text from all pages
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    text_content += f"\n--- Page {page_num + 1} ---\n"
                    text_content += text
            
            doc.close()
            
            if not text_content.strip():
                raise ValueError("No text content found in PDF")
            
            # Clean up text
            text_content = self._clean_text(text_content)
            
            return text_content, pdf_title
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page breaks and form feeds
        text = text.replace('\f', '\n')
        text = text.replace('\r', '\n')
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()

class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Overlap between consecutive chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        """
        try:
            if not text or not text.strip():
                return []
            
            chunks = []
            start = 0
            
            while start < len(text):
                # Calculate end position
                end = start + self.chunk_size
                
                # If this is not the last chunk, try to end at a sentence boundary
                if end < len(text):
                    # Look for sentence endings within the last 100 characters
                    sentence_end = self._find_sentence_boundary(text, end - 100, end)
                    if sentence_end > start:
                        end = sentence_end
                
                # Extract chunk
                chunk = text[start:end].strip()
                
                if chunk:
                    chunks.append(chunk)
                
                # Move start position (with overlap)
                start = end - self.overlap if end > self.overlap else end
                
                # Break if we've reached the end
                if start >= len(text):
                    break
            
            print(f"Split text into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            raise Exception(f"Error splitting text: {str(e)}")
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary within a range"""
        # Look for sentence endings (., !, ?)
        sentence_endings = ['.', '!', '?']
        
        best_pos = end
        for i in range(end - 1, start - 1, -1):
            if text[i] in sentence_endings:
                # Check if this looks like a proper sentence ending
                if i + 1 < len(text) and (text[i + 1].isspace() or text[i + 1].isupper()):
                    best_pos = i + 1
                    break
        
        return best_pos

class ConfigValidator:
    """Validate environment configuration"""
    
    @staticmethod
    def validate_env_vars() -> Dict[str, Any]:
        """Validate all required environment variables"""
        required_vars = {
            "GROQ_API_KEY": "Groq API key for language model",
            "PINECONE_API_KEY": "Pinecone API key for vector database",
            "PINECONE_ENVIRONMENT": "Pinecone environment",
            "PINECONE_INDEX_NAME": "Pinecone index name"
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            if not os.getenv(var):
                missing_vars.append(f"{var} ({description})")
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return {
            "groq_api_key": os.getenv("GROQ_API_KEY"),
            "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
            "pinecone_environment": os.getenv("PINECONE_ENVIRONMENT"),
            "pinecone_index_name": os.getenv("PINECONE_INDEX_NAME")
        }
"""Document ingestion module for handling PDFs, text files, and markdown."""
import os
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a document with content and metadata."""
    content: str
    file_name: str
    file_path: str
    file_type: str  # pdf, txt, md
    page_number: int = 0


class DocumentIngester:
    """Handles ingestion of various document formats."""
    
    SUPPORTED_FORMATS = {".pdf", ".txt", ".md", ".markdown"}
    
    @staticmethod
    def ingest_directory(directory_path: str) -> List[Document]:
        """Ingest all supported documents from a directory."""
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in DocumentIngester.SUPPORTED_FORMATS:
                try:
                    docs = DocumentIngester.ingest_file(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    print(f"Warning: Could not ingest {file_path}: {e}")
        
        return documents
    
    @staticmethod
    def ingest_file(file_path: str) -> List[Document]:
        """Ingest a single document file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf":
            return DocumentIngester._ingest_pdf(file_path)
        elif suffix in {".txt", ".md", ".markdown"}:
            return DocumentIngester._ingest_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    @staticmethod
    def _ingest_pdf(file_path: Path) -> List[Document]:
        """Extract text from PDF file."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required for PDF support. Install it with: pip install pypdf")
        
        documents = []
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    doc = Document(
                        content=text,
                        file_name=file_path.name,
                        file_path=str(file_path),
                        file_type="pdf",
                        page_number=page_num
                    )
                    documents.append(doc)
        except Exception as e:
            raise ValueError(f"Error reading PDF {file_path}: {e}")
        
        return documents
    
    @staticmethod
    def _ingest_text(file_path: Path) -> List[Document]:
        """Extract text from .txt or .md files."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
        
        if not content.strip():
            return []
        
        doc = Document(
            content=content,
            file_name=file_path.name,
            file_path=str(file_path),
            file_type=file_path.suffix.lower().lstrip(".")
        )
        
        return [doc]

import re
import io
from PyPDF2 import PdfReader

def process_document(content: bytes, filename: str) -> list:
    if filename.lower().endswith('.pdf'):
        text = extract_text_from_pdf(content)
    else:
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('latin-1')  # Fallback encoding
    
    chunks = chunk_text(text)
    return chunks

def extract_text_from_pdf(content: bytes) -> str:
    pdf_file = io.BytesIO(content)
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 1000) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
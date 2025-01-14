import PyPDF2
import pdfplumber
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
import requests
import tempfile
import os
from urllib.parse import urlparse
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('pdf_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set PyPDF2 logger to WARNING level
pdf_logger = logging.getLogger('PyPDF2')
pdf_logger.setLevel(logging.WARNING)

# Set pdfplumber logger to WARNING level
plumber_logger = logging.getLogger('pdfplumber')
plumber_logger.setLevel(logging.WARNING)

class PDFReader:
    def __init__(self, pdf_path: Union[str, Path], skip_first_page: bool = True):
        """Initialize PDF reader with file path or URL"""
        self.original_path = pdf_path
        self.is_url = bool(urlparse(str(pdf_path)).scheme)
        self.temp_file = None
        self.skip_first_page = skip_first_page
        
        # Setup financial_pdfs directory
        self.pdf_dir = Path(__file__).parent.parent / 'financial_pdfs'
        self.pdf_dir.mkdir(exist_ok=True)
        
        if self.is_url:
            self.pdf_path = self._download_pdf(pdf_path)
        else:
            self.pdf_path = Path(pdf_path)
            if not self.pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.text_content = {}
        self.metadata = {}

    def _download_pdf(self, url: str) -> Path:
        """Download PDF from URL to financial_pdfs directory"""
        try:
            logger.info(f"Downloading PDF from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Extract filename from URL or use default
            filename = url.split('/')[-1]
            if not filename.lower().endswith('.pdf'):
                # Count existing financial guide files and generate new filename
                filename = 'financial_guide'
            # Create permanent path in financial_pdfs directory
            permanent_path = self.pdf_dir / filename
            
            # Download directly to permanent location
            with open(permanent_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"PDF downloaded successfully to {permanent_path}")
            return permanent_path
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            raise

    def __del__(self):
        """Cleanup temporary file if it exists"""
        if self.temp_file:
            try:
                os.unlink(self.temp_file.name)
                logger.info("Temporary PDF file cleaned up")
            except:
                pass

    def extract_with_pdfplumber(self) -> Dict[int, str]:
        """Extract text using pdfplumber (better for complex layouts)"""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                # Skip first page if specified
                start_page = 1 if self.skip_first_page else 0
                return {
                    i: page.extract_text() 
                    for i, page in enumerate(pdf.pages[start_page:], start_page + 1)
                    if page.extract_text()  # Skip empty pages
                }
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            return {}

    def extract_with_pypdf2(self) -> Dict[int, str]:
        """Extract text using PyPDF2 (faster but simpler)"""
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                self.metadata = reader.metadata
                
                # Skip first page if specified
                start_page = 1 if self.skip_first_page else 0
                return {
                    i: page.extract_text() 
                    for i, page in enumerate(reader.pages[start_page:], start_page + 1)
                    if page.extract_text()  # Skip empty pages
                }
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {e}")
            return {}

    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not text:
            return ""
            
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;?!-]', '', text)
        # Fix spacing around punctuation
        text = re.sub(r'\s*([.,;?!])', r'\1', text)
        
        return text.strip()

    def extract_terms_and_definitions(self, text: str) -> List[tuple]:
        """Extract terms and their definitions from text"""
        if not text:
            return []
        
        # Remove headers and page numbers
        text = re.sub(r'Plain English Campaign The A to Z of financial terms.*?(?=\w+\s+This is)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Remove page numbers
        
        # Split text into lines and process
        terms_and_defs = []
        current_term = None
        current_def = []
        
        # Split by newlines while preserving paragraph structure
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a term (capitalized, short, and followed by definition)
            if re.match(r'^[A-Z][A-Za-z\s\-&()]+$', line) and len(line.split()) <= 4:
                # Save previous term and definition if they exist
                if current_term and current_def:
                    terms_and_defs.append((
                        current_term,
                        ' '.join(current_def).strip()
                    ))
                
                current_term = line
                current_def = []
            
            # If we have a current term, add this line to its definition
            elif current_term and line.startswith('This'):
                current_def = [line]
            elif current_term and current_def:
                current_def.append(line)
        
        # Don't forget to add the last term
        if current_term and current_def:
            terms_and_defs.append((
                current_term,
                ' '.join(current_def).strip()
            ))
        
        # Clean up definitions
        cleaned_pairs = []
        for term, definition in terms_and_defs:
            # Remove any remaining headers or page numbers
            definition = re.sub(r'Plain English Campaign.*?(?=\w)', '', definition, flags=re.IGNORECASE)
            definition = re.sub(r'\s+', ' ', definition)  # Normalize spaces
            
            if term and definition:
                cleaned_pairs.append((term.strip(), definition.strip()))
        
        return cleaned_pairs

    def save_text_content(self, processed_content: Dict[str, any]) -> Path:
        """Save extracted text content as terms and definitions"""
        try:
            text_path = self.pdf_path.with_suffix('.txt')
            
            # Combine all text from all pages with proper spacing
            all_text = '\n\n'.join(
                page['full_text'] 
                for page in processed_content['pages'].values()
            )
            
            # Extract and format terms and definitions
            terms_and_defs = self.extract_terms_and_definitions(all_text)
            
            # Save the terms and definitions
            with open(text_path, 'w', encoding='utf-8') as f:
                if not terms_and_defs:
                    logger.warning("No terms and definitions were extracted!")
                    f.write("Warning: No terms and definitions were extracted from the PDF.\n")
                else:
                    logger.info(f"Extracted {len(terms_and_defs)} term-definition pairs")
                    for term, definition in terms_and_defs:
                        f.write(f"{term}: {definition}\n\n")
            
            return text_path
            
        except Exception as e:
            logger.error(f"Error saving text content: {e}")
            raise

def main():
    """Example usage"""
    test_paths = [
        "https://www.plainenglish.co.uk/files/financialguide.pdf",
        "https://financialliteracy.psu.edu/sites/default/files/FinLit-Glossary-of-Terms.pdf"
    ]
    
    for pdf_path in test_paths:
        try:
            logger.info(f"Processing PDF from: {pdf_path}")
            reader = PDFReader(pdf_path, skip_first_page=True)
            
            # Try both extraction methods
            plumber_text = reader.extract_with_pdfplumber()
            pypdf2_text = reader.extract_with_pypdf2()
            
            # Print some debug information
            print(f"\nResults for {pdf_path}:")
            print(f"Pages extracted with pdfplumber: {len(plumber_text)}")
            print(f"Pages extracted with PyPDF2: {len(pypdf2_text)}")
            
            # Print sample from first page
            if plumber_text:
                # Process the content
                processed_content = {
                    'metadata': reader.metadata,
                    'pages': {
                        page_num: {'full_text': text}
                        for page_num, text in plumber_text.items()
                    }
                }
                
                # Save to text file
                text_file_path = reader.save_text_content(processed_content)
                print(f"\nSaved text content to: {text_file_path}")
                
                # Rest of the debug output
                first_page = next(iter(plumber_text.values()))
                print("\nSample text from first page (pdfplumber):")
                print(first_page[:500] + "...")
                
                # Extract terms and definitions
                terms_and_defs = reader.extract_terms_and_definitions(first_page)
                print("\nExtracted terms and definitions:")
                for term, definition in terms_and_defs[:5]:
                    print(f"\nTerm: {term}")
                    print(f"Definition: {definition}")
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            continue

if __name__ == "__main__":
    main()

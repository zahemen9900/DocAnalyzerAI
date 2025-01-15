import pdfplumber
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import re
import requests
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThomasWillingGlossaryReader:
    """Reader for Thomas Willing financial history glossary PDF"""
    
    def __init__(self, source_url: str):
        """Initialize with URL"""
        self.source_url = source_url
        self.pdf_dir = Path(__file__).parent.parent / 'financial_pdfs'
        self.pdf_dir.mkdir(exist_ok=True)
        self.pdf_path = self.pdf_dir / 'Thomas-Willing-financial-history-glossary2.pdf'
        
        if not self.pdf_path.exists():
            self._download_pdf()

    def _download_pdf(self):
        """Download PDF from URL"""
        try:
            logger.info(f"Downloading PDF from {self.source_url}")
            response = requests.get(self.source_url, stream=True)
            response.raise_for_status()
            
            with open(self.pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"PDF downloaded successfully to {self.pdf_path}")
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            raise

    def extract_with_pdfplumber(self) -> dict:
        """Extract text from PDF, skipping the first page"""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                processed_pages = {}
                
                # Skip the first page (introductory text)
                for i, page in enumerate(pdf.pages[1:], 1):
                    text = page.extract_text()
                    if text:
                        processed_pages[i] = text
                
                return processed_pages
                
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            return {}

    def clean_intro_text(self, text: str) -> str:
        """Remove introductory text"""
        intro_pattern = re.compile(
            r'Glossary of Important Business, Economic, and Financial History Terms.*?sharing of creative work\.',
            re.DOTALL
        )
        return re.sub(intro_pattern, '', text).strip()

    def extract_terms_and_definitions(self, text: str) -> List[Tuple[str, str]]:
        """Extract terms and definitions separated by colons"""
        if not text:
            return []
        
        terms_and_defs = []
        current_term = None
        current_def_lines = []
        
        # Split text into lines
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line contains a colon
            if ':' in line:
                # If we have a previous term-def pair, save it
                if current_term and current_def_lines:
                    terms_and_defs.append((
                        current_term,
                        ' '.join(current_def_lines)
                    ))
                
                # Split new line at first colon
                term, definition = line.split(':', 1)
                term = re.sub(r'\s*\(n\.\)\s*|\s*\(v\.\)\s*|\s*\(adj\.\)\s*', '', term).strip()  # Remove (n.), (v.), and (adj.)
                current_term = term
                current_def_lines = [definition.strip()]
            
            # If no colon and we have a current term, append to current definition
            elif current_term:
                current_def_lines.append(line)
        
        # Don't forget the last term-def pair
        if current_term and current_def_lines:
            terms_and_defs.append((
                current_term,
                ' '.join(current_def_lines)
            ))
        
        # Clean up and validate pairs
        cleaned_pairs = []
        for term, definition in terms_and_defs:
            # Clean up term and definition
            term = re.sub(r'\s+', ' ', term).strip()
            definition = re.sub(r'\s+', ' ', definition).strip()
            
            if len(term) > 1 and len(definition) > 5:
                cleaned_pairs.append((term, definition))
        
        return cleaned_pairs

    def save_text_content(self, processed_content: dict) -> Path:
        """Save extracted content with colon format"""
        try:
            text_path = self.pdf_path.with_suffix('.txt')
            
            # Combine all terms and definitions
            all_terms_and_defs = []
            for page_num, page_data in processed_content['pages'].items():
                terms_and_defs = page_data['terms_and_definitions']
                all_terms_and_defs.extend(terms_and_defs)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_terms = []
            for term, definition in all_terms_and_defs:
                if term.lower() not in seen:
                    seen.add(term.lower())
                    unique_terms.append((term, definition))
            
            # Save with colon format
            with open(text_path, 'w', encoding='utf-8') as f:
                for term, definition in unique_terms:
                    f.write(f"{term}: {definition}\n\n")
            
            logger.info(f"Saved {len(unique_terms)} terms to {text_path}")
            return text_path
            
        except Exception as e:
            logger.error(f"Error saving text content: {e}")
            raise

    def process_pdf(self) -> Dict[str, any]:
        """Process PDF and extract terms/definitions"""
        try:
            logger.info("Starting PDF processing...")
            page_dict = self.extract_with_pdfplumber()
            
            if not page_dict:
                raise ValueError("No text content extracted from PDF")

            processed_content = {
                'metadata': {},
                'pages': {},
                'total_pages': len(page_dict)
            }

            # Process each page individually
            for page_num, page_text in page_dict.items():
                # Clean intro text only for the first content page (page 1)
                cleaned_text = self.clean_intro_text(page_text) if page_num == 1 else page_text
                
                terms_and_defs = self.extract_terms_and_definitions(cleaned_text)
                processed_content['pages'][page_num] = {
                    'full_text': cleaned_text,
                    'terms_and_definitions': terms_and_defs
                }

            # Save processed content
            self.save_text_content(processed_content)
            
            return processed_content

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

def main():
    """Example usage"""
    try:
        # URL of the PDF
        url = "https://www.augie.edu/sites/default/files/shared/Thomas-Willing-financial-history-glossary2.pdf"
        
        logger.info(f"Processing PDF from: {url}")
        reader = ThomasWillingGlossaryReader(url)
        content = reader.process_pdf()
        
        if content and content['pages']:
            terms_and_defs = []
            for page_num in content['pages']:
                page_terms = reader.extract_terms_and_definitions(content['pages'][page_num]['full_text'])
                terms_and_defs.extend(page_terms)
            
            print(f"\nExtracted {len(terms_and_defs)} terms and definitions.")
            print("\nFirst few entries:")
            for term, definition in terms_and_defs[:5]:
                print(f"\n{term}: {definition}")
    
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

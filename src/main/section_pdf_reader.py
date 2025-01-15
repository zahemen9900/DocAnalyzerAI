import pdfplumber
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import re
from pdf_reader import PDFReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SectionPDFReader(PDFReader):
    """Reader for PDFs with colon-separated terms and section letters"""
    
    def __init__(self, source_path: str, skip_pages: tuple = (0, 1, 2, -1)):
        """Initialize with skip pages option"""
        super().__init__(source_path, skip_first_page=False)
        self.skip_pages = skip_pages
    
    def extract_with_pdfplumber(self) -> dict:
        """Extract text from PDF, skipping specified pages"""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                processed_pages = {}
                
                # Convert negative indices to positive
                skip_indices = {i if i >= 0 else len(pdf.pages) + i for i in self.skip_pages}
                
                # Process only non-skipped pages
                for i, page in enumerate(pdf.pages):
                    if i not in skip_indices:
                        text = page.extract_text()
                        if text:
                            processed_pages[i] = text
                
                return processed_pages
                
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            return {}

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
            
            # Skip single letter section markers
            if re.match(r'^[A-Z]$', line):
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
                current_term = term.strip()
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
            # Skip if term is just a single letter
            if len(term.strip()) <= 1:
                continue
                
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
            for page_content in processed_content['pages'].values():
                # Access the full_text from the page_content dictionary
                text = page_content['full_text']
                terms_and_defs = self.extract_terms_and_definitions(text)
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
            raw_text = self.extract_with_pdfplumber()
            
            if not raw_text:
                raise ValueError("No text content extracted from PDF")

            # Process each page
            processed_content = {
                'metadata': self.metadata,
                'pages': {},
                'total_pages': len(raw_text)
            }

            # Process each page individually
            for page_num, page_text in raw_text.items():
                terms_and_defs = self.extract_terms_and_definitions(page_text)
                processed_content['pages'][page_num] = {
                    'full_text': page_text,
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
        # Use your test PDF path here
        test_path = "https://www.naco.org/sites/default/files/documents/Glossary%20of%20Public%20Finance%20Terms.pdf"
        
        logger.info(f"Processing PDF from: {test_path}")
        reader = SectionPDFReader(test_path)
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

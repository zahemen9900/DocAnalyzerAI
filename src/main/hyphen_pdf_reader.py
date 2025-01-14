import logging
import re
from pdf_reader import PDFReader
from typing import List, Tuple, Dict
from pathlib import Path
import pdfplumber

# Configure logging to only show INFO and above for our logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Explicitly set PyPDF2 logger to WARNING level
pdf_logger = logging.getLogger('PyPDF2')
pdf_logger.setLevel(logging.WARNING)

class HyphenPDFReader(PDFReader):
    """PDF reader specialized for hyphen-separated term definitions"""
    
    def extract_with_pdfplumber(self) -> Dict[int, str]:
        """Extract text from two-column layout PDF"""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                processed_pages = {}
                start_page = 1 if self.skip_first_page else 0
                
                for i, page in enumerate(pdf.pages[start_page:], start_page + 1):
                    # Get page dimensions
                    width = page.width
                    height = page.height
                    
                    # Extract left column (first half of page)
                    left_bbox = (0, 0, width/2, height)
                    left_text = page.crop(left_bbox).extract_text() or ""
                    
                    # Extract right column (second half of page)
                    right_bbox = (width/2, 0, width, height)
                    right_text = page.crop(right_bbox).extract_text() or ""
                    
                    # Combine columns with proper separation
                    full_text = left_text.strip() + "\n\n" + right_text.strip()
                    if full_text.strip():
                        processed_pages[i] = full_text
                
                return processed_pages
                
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            return {}

    def extract_terms_and_definitions(self, text: str) -> List[tuple]:
        """Extract terms and definitions from two-column text"""
        if not text:
            return []
        
        logger.info("Starting term extraction...")
        
        # Split text into lines and clean
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        terms_and_defs = []
        current_term = None
        current_def = []
        
        for line in lines:
            # Skip header lines
            if "Penn State" in line or "Financial and Life Skills Center" in line:
                continue
                
            # Look for term-definition patterns
            term_match = re.match(r'^([^:–-]+?)(?:[:–-])\s*(.+)$', line)
            
            if term_match:
                # If we have a previous term-def pair, save it
                if current_term and current_def:
                    terms_and_defs.append((
                        current_term.strip(),
                        ' '.join(current_def).strip()
                    ))
                
                # Start new term-def pair
                current_term = term_match.group(1)
                current_def = [term_match.group(2)]
            elif current_term and line:
                # Continue previous definition if it spans multiple lines
                # but only if the line doesn't look like the start of a new term
                if not re.match(r'^[A-Z][^:–-]+[:–-]', line):
                    current_def.append(line)
        
        # Add the last term-def pair
        if current_term and current_def:
            terms_and_defs.append((
                current_term.strip(),
                ' '.join(current_def).strip()
            ))
        
        # Clean up the extracted pairs
        cleaned_pairs = []
        for term, definition in terms_and_defs:
            # Clean up term and definition
            term = re.sub(r'\s+', ' ', term).strip()
            definition = re.sub(r'\s+', ' ', definition).strip()
            
            # Remove any partial terms or definitions
            if len(term) > 2 and len(definition) > 5 and not definition.lower().startswith(term.lower()):
                cleaned_pairs.append((term, definition))
        
        logger.info(f"Extracted {len(cleaned_pairs)} terms")
        return cleaned_pairs

    def process_pdf(self, use_plumber: bool = True) -> Dict[str, any]:
        """Main method to process PDF and return structured content"""
        try:
            logger.info("Starting PDF processing...")
            
            # Try both extraction methods and use the one that gives better results
            plumber_text = self.extract_with_pdfplumber()
            pypdf2_text = self.extract_with_pypdf2()
            
            # Use pdfplumber text if available, otherwise fallback to PyPDF2
            raw_text = plumber_text if plumber_text else pypdf2_text
            
            if not raw_text:
                raise ValueError("No text content extracted from PDF")

            # Process each page
            processed_content = {
                'metadata': self.metadata,
                'pages': {},
                'total_pages': len(raw_text)
            }

            # Combine all pages into one text for better term extraction
            all_text = '\n\n'.join(text for text in raw_text.values())
            terms_and_defs = self.extract_terms_and_definitions(all_text)
            
            # Store processed content
            processed_content['pages'][1] = {
                'full_text': all_text,
                'terms_and_definitions': terms_and_defs
            }

            # Save the content
            text_file_path = self.save_text_content(processed_content)
            logger.info(f"Content saved to {text_file_path}")
            
            return processed_content

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

    def save_text_content(self, processed_content: Dict[str, any]) -> Path:
        """Save extracted text content as terms and definitions"""
        try:
            text_path = self.pdf_path.with_suffix('.txt')
            
            # Extract terms and definitions
            terms_and_defs = []
            for page_content in processed_content['pages'].values():
                page_terms = self.extract_terms_and_definitions(page_content['full_text'])
                terms_and_defs.extend(page_terms)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_terms = []
            for term, definition in terms_and_defs:
                if term not in seen:
                    seen.add(term)
                    unique_terms.append((term, definition))
            
            # Save the terms and definitions
            with open(text_path, 'w', encoding='utf-8') as f:
                if not unique_terms:
                    logger.warning("No terms and definitions were extracted!")
                    f.write("Warning: No terms and definitions were extracted from the PDF.\n")
                else:
                    logger.info(f"Extracted {len(unique_terms)} unique term-definition pairs")
                    for term, definition in unique_terms:
                        f.write(f"{term}: {definition}\n\n")
            
            return text_path
            
        except Exception as e:
            logger.error(f"Error saving text content: {e}")
            raise

def main():
    """Example usage"""
    try:
        # Use test PDF
        test_path = "https://financialliteracy.psu.edu/sites/default/files/FinLit-Glossary-of-Terms.pdf"
        
        logger.info(f"Processing PDF from: {test_path}")
        reader = HyphenPDFReader(test_path, skip_first_page=False)
        
        # Process the PDF
        content = reader.process_pdf(use_plumber=True)
        
        if content and content['pages']:
            # Get the terms and definitions
            terms_and_defs = content['pages'][1].get('terms_and_definitions', [])
            
            if terms_and_defs:
                print("\nExtracted Terms and Definitions:")
                for term, definition in terms_and_defs[:5]:  # Show first 5 entries
                    print(f"\nTerm: {term}")
                    print(f"Definition: {definition}")
                print(f"\nTotal terms extracted: {len(terms_and_defs)}")
            else:
                print("No terms were extracted. This might indicate an issue with the PDF format.")
                
            # Print raw text sample for debugging
            print("\nRaw text sample from PDF:")
            print(content['pages'][1]['full_text'][:500] + "...")
        
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

import pdfplumber
import logging
from pathlib import Path
from typing import List, Tuple
import re
import requests
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColumnGlossaryReader:
    """Reader for columnar glossary PDFs with specific term-definition patterns"""
    
    DEFINITION_STARTERS = {
        'The', 'A', 'An', 'Any', 'This', 'That', 'These', 'Those', 'It', 'They',
        'Is', 'Are', 'Was', 'Were', 'To', 'For', 'In', 'On', 'At', 'By',
        'When', 'Where', 'Which', 'Who', 'What', 'How'
    }
    
    CONTINUATION_WORDS = {
        'It', 'This', 'That', 'These', 'Those', 'They', 'He', 'She', 'You',
        'Your', 'Their', 'Its', 'His', 'Her', 'We', 'Our', 'Such', 'Some',
        'Therefore', 'Thus', 'Consequently', 'Hence', 'So', 'As a result',
        'Accordingly', 'For this reason', 'Because of this'
    }
    
    def __init__(self, source_path: str):
        """Initialize with either URL or local path"""
        self.source_path = source_path
        self.is_url = bool(urlparse(source_path).scheme)
        
        # Setup PDF directory
        self.pdf_dir = Path(__file__).parent.parent / 'financial_pdfs'
        self.pdf_dir.mkdir(exist_ok=True)
        
        # Get PDF path
        if self.is_url:
            filename = source_path.split('/')[-1]
            self.pdf_path = self.pdf_dir / filename
            if not self.pdf_path.exists():
                self._download_pdf()
        else:
            self.pdf_path = Path(source_path)
            if not self.pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {source_path}")

    def _download_pdf(self):
        """Download PDF from URL"""
        try:
            logger.info(f"Downloading PDF from {self.source_path}")
            response = requests.get(self.source_path, stream=True)
            response.raise_for_status()
            
            with open(self.pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"PDF downloaded successfully to {self.pdf_path}")
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            raise

    def is_capitalized_word(self, word: str) -> bool:
        """Check if word follows proper capitalization pattern"""
        return (word and 
                word[0].isupper() and 
                word[1:].islower() if len(word) > 1 else True)

    def extract_term_and_definition(self, text: str) -> List[Tuple[str, str]]:
        """Extract terms and definitions using specified patterns"""
        terms_and_defs = []
        
        # Split text into blocks at periods
        blocks = re.split(r'(?<=[.])\s+(?=[A-Z])', text)
        current_term = None
        current_def_parts = []
        
        for block in blocks:
            # Skip empty blocks
            if not block.strip():
                continue
                
            # Handle section letter markers
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if not lines:
                continue
                
            # Check for lone letter section markers
            if (len(lines[0]) == 1 and lines[0].isupper() and len(lines) > 1 and 
                lines[1][0].upper() == lines[0]):
                lines = lines[1:]  # Remove the section marker
            
            # Join remaining lines
            sentence = ' '.join(lines)
            words = [w for w in sentence.split() if w.strip()]
            if len(words) < 2:  # Need at least one word for term and one for definition
                continue
            
            # Check if this is a continuation of previous definition
            if words[0] in self.CONTINUATION_WORDS and current_term:
                current_def_parts.append(sentence)
                continue
            
            # Save previous term-definition pair
            if current_term and current_def_parts:
                definition = ' '.join(current_def_parts)
                if definition.strip().endswith('.'):
                    terms_and_defs.append((current_term, definition))
            
            # Process new term
            term_words = []
            first_four = words[:4]
            
            # Rule 1: Two capitalized words followed by lowercase
            if (len(first_four) >= 3 and 
                self.is_capitalized_word(first_four[0]) and 
                self.is_capitalized_word(first_four[1]) and 
                not self.is_capitalized_word(first_four[2])):
                term_words = [first_four[0]]
            
            # Rule 2: Three capitalized words followed by lowercase
            elif (len(first_four) >= 4 and 
                  self.is_capitalized_word(first_four[0]) and 
                  self.is_capitalized_word(first_four[1]) and 
                  self.is_capitalized_word(first_four[2]) and 
                  not self.is_capitalized_word(first_four[3])):
                term_words = [first_four[0], first_four[1]]
            
            # Rule 3: Four capitalized words followed by lowercase
            elif (len(first_four) == 4 and all(self.is_capitalized_word(w) for w in first_four) and
                  len(words) > 4 and not self.is_capitalized_word(words[4])):
                term_words = [first_four[0], first_four[1], first_four[2]]
            
            # Additional check for terms with less than 3 words
            else:
                term_words = [first_four[0]]
                for i in range(1, min(3, len(first_four))):
                    if (i < len(first_four) - 1 and 
                        self.is_capitalized_word(first_four[i + 1])):
                        term_words.append(first_four[i])
                    else:
                        break
            
            # Set up new term and definition
            if term_words:
                current_term = ' '.join(term_words)
                definition_start = len(term_words)
                current_def_parts = [' '.join(words[definition_start:])]
        
        # Add last term-definition pair
        if current_term and current_def_parts:
            definition = ' '.join(current_def_parts)
            if definition.strip().endswith('.'):
                terms_and_defs.append((current_term, definition))
        
        return terms_and_defs

    def process_pdf(self) -> List[Tuple[str, str]]:
        """Process PDF and extract terms/definitions"""
        all_terms_and_defs = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    # Extract terms and definitions from page
                    terms_and_defs = self.extract_term_and_definition(text)
                    all_terms_and_defs.extend(terms_and_defs)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_terms = []
            for term, definition in all_terms_and_defs:
                if term.lower() not in seen:
                    seen.add(term.lower())
                    unique_terms.append((term, definition))
            
            logger.info(f"Extracted {len(unique_terms)} unique terms and definitions")
            return unique_terms
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def save_to_file(self, terms_and_defs: List[Tuple[str, str]], output_path: str = None):
        """Save extracted terms and definitions with new format"""
        if not output_path:
            output_path = self.pdf_path.with_suffix('.txt')
            
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for term, definition in terms_and_defs:
                    f.write(f"{term}: {definition}\n\n")
                    
            logger.info(f"Saved {len(terms_and_defs)} terms to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            raise

def main():
    """Example usage"""
    try:
        # Test with URL
        url = "https://financialaccounting.ua.edu/wp-content/uploads/References/Glossary-of-Finance-Terms-UA.pdf"
        
        logger.info(f"Processing PDF from: {url}")
        reader = ColumnGlossaryReader(url)
        terms_and_defs = reader.process_pdf()
        reader.save_to_file(terms_and_defs)
        
        # Print first few entries for verification
        print("\nFirst few extracted terms:")
        for term, definition in terms_and_defs[:5]:
            print(f"\nTerm: {term}")
            print(f"Definition: {definition}")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()

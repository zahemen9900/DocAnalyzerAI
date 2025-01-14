import pdfplumber
import logging
from pathlib import Path
from typing import List, Tuple
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColumnGlossaryReader:
    """Reader for columnar glossary PDFs with specific term-definition patterns"""
    
    PRONOUNS = {'The', 'A', 'An', 'Any', 'This', 'That', 'These', 'Those', 'It', 'They'}
    
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    def is_term_word(self, word: str) -> bool:
        """Check if a word follows term capitalization pattern"""
        return (word 
                and word[0].isupper() 
                and not all(c.isupper() for c in word[1:]))
    
    def extract_term_and_definition(self, line: str) -> Tuple[str, str]:
        """Extract term and definition based on specified patterns"""
        words = line.strip().split()
        if not words or not self.is_term_word(words[0]):
            return None, None
            
        # Get potential term words (up to first 4 words)
        potential_term = words[:4]
        remaining_words = words[4:]
        term_words = []
        
        # Process first word (always part of term if capitalized)
        term_words.append(potential_term[0])
        
        if len(potential_term) > 1:
            second_word = potential_term[1]
            
            # Check second word
            if (second_word not in self.PRONOUNS and 
                self.is_term_word(second_word) and
                (len(potential_term) < 3 or not second_word.lower() == potential_term[2].lower())):
                term_words.append(second_word)
                
                # Check third word if present
                if (len(potential_term) > 2 and 
                    self.is_term_word(potential_term[2]) and
                    (len(potential_term) < 4 or not potential_term[2].lower() == potential_term[3].lower())):
                    term_words.append(potential_term[2])
        
        # Combine remaining words as definition
        definition_start = len(term_words)
        definition_words = words[definition_start:]
        
        if not definition_words:
            return None, None
            
        term = ' '.join(term_words)
        definition = ' '.join(definition_words)
        
        return term, definition
    
    def process_pdf(self) -> List[Tuple[str, str]]:
        """Process PDF and extract terms/definitions"""
        terms_and_defs = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    # Process each line
                    lines = text.split('\n')
                    for line in lines:
                        if not line.strip():
                            continue
                            
                        term, definition = self.extract_term_and_definition(line)
                        if term and definition:
                            terms_and_defs.append((term, definition))
            
            logger.info(f"Extracted {len(terms_and_defs)} terms and definitions")
            return terms_and_defs
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def save_to_file(self, terms_and_defs: List[Tuple[str, str]], output_path: str = None):
        """Save extracted terms and definitions to file"""
        if not output_path:
            output_path = self.pdf_path.with_suffix('.txt')
            
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for term, definition in terms_and_defs:
                    f.write(f"Term: {term}\nDefinition: {definition}\n\n")
                    
            logger.info(f"Saved {len(terms_and_defs)} terms to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            raise

def main():
    """Example usage"""
    parent_path = Path(__file__).resolve().parent.parent
    parent_folder = parent_path / "financial_pdfs"
    pdf_path = parent_folder / "Glossary-of-Finance-Terms-UA.pdf"
    try:
        reader = ColumnGlossaryReader(pdf_path)
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

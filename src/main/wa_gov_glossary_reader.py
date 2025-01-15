import pdfplumber
import logging
from pathlib import Path
import re
from typing import List, Tuple
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WAGovGlossaryReader:
    """Reader for WA government glossary PDF with hyphen-separated terms"""
    
    def __init__(self, source_url: str):
        self.source_url = source_url
        self.pdf_dir = Path(__file__).parent.parent / 'financial_pdfs'
        self.pdf_dir.mkdir(exist_ok=True)
        self.pdf_path = self.pdf_dir / 'wa_gov_glossary.pdf'
        
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

    def clean_text(self, text: str) -> str:
        """Clean up text content"""
        # Remove headers and footers
        text = re.sub(r'Office of Financial Management\s+June 2018', '', text)
        # Remove GLOSSARY OF TERMS if present
        text = re.sub(r'^GLOSSARY OF TERMS\s+', '', text)
        return text.strip()

    def should_merge_with_previous(self, term: str) -> bool:
        """Check if term should be merged with previous definition"""
        first_word = term.lower().split()[0] if term else ''
        return (
            first_word in {'a', 'an', 'the', 'you', 'it', 'they', 'this', 'that', 'these', 'those',
                           'usually', 'in', 'therefore', 'when', 'can', 'since', 'also', 'if', 'however',
                           'because', 'while', 'although', 'unless', 'moreover', 'furthermore'}
            or term[0].islower() if term else False
        )

    def process_definition_colons(self, text: str) -> list:
        """Process colons within definitions to create new term-definition pairs"""
        additional_pairs = []
        sentences = text.split('. ')
        
        for sentence in sentences:
            if ': ' in sentence:
                before_colon, after_colon = sentence.split(': ', 1)
                if not self.should_merge_with_previous(before_colon):
                    additional_pairs.append((before_colon.strip(), after_colon.strip()))
                
        return additional_pairs

    def extract_terms_and_definitions(self, text: str) -> List[Tuple[str, str]]:
        """Extract terms and definitions separated by hyphens"""
        if not text:
            return []
        
        terms_and_defs = []
        # Handle both em-dash and regular hyphen, preserve numbers in definitions
        pattern = r'([A-Z][A-Za-z0-9\s\(\)/,]+?)(?:\s*[—-]\s*)([^A-Z][^—]*?)(?=(?:[A-Z][A-Za-z\s\(\)]+?\s*[—-])|$)'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        current_term = None
        current_def = []
        
        for match in matches:
            term = match.group(1).strip()
            definition = match.group(2).strip()
            
            # Clean up the definition
            definition = re.sub(r'\s+', ' ', definition)
            # Remove any trailing periods and spaces from term
            term = term.rstrip('. ')
            
            # Skip single-letter section markers
            if re.match(r'^[A-Z]$', term):
                continue
                
            # Check if this definition starts with lowercase and should be merged
            if current_term and (definition[0].islower() if definition else False):
                current_def.append(definition)
                continue
            
            # Save previous term-def pair if exists
            if current_term and current_def:
                terms_and_defs.append((current_term, ' '.join(current_def)))
            
            current_term = term
            current_def = [definition]
        
        # Add last pair
        if current_term and current_def:
            terms_and_defs.append((current_term, ' '.join(current_def)))
        
        return terms_and_defs

    def process_pdf(self) -> List[Tuple[str, str]]:
        """Process PDF and extract terms/definitions"""
        try:
            all_terms_and_defs = []
            
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        cleaned_text = self.clean_text(text)
                        # Process whole page text to maintain context
                        terms_and_defs = self.extract_terms_and_definitions(cleaned_text)
                        all_terms_and_defs.extend(terms_and_defs)
            
            # Process pairs and merge where needed
            processed_pairs = []
            i = 0
            while i < len(all_terms_and_defs):
                term, definition = all_terms_and_defs[i]
                
                if processed_pairs and self.should_merge_with_previous(term):
                    # Merge with previous definition
                    prev_term, prev_def = processed_pairs[-1]
                    processed_pairs[-1] = (prev_term, f"{prev_def} {definition}")
                else:
                    # Process colons in definition
                    additional_pairs = self.process_definition_colons(definition)
                    if additional_pairs:
                        processed_pairs.extend(additional_pairs)
                    else:
                        processed_pairs.append((term, definition))
                i += 1

            # Remove duplicates while preserving order
            seen = set()
            unique_terms = []
            for term, definition in processed_pairs:
                if term.lower() not in seen:
                    seen.add(term.lower())
                    unique_terms.append((term, definition))
            
            # Save to file
            self.save_to_file(unique_terms)
            
            return unique_terms
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

    def save_to_file(self, terms_and_defs: List[Tuple[str, str]]):
        """Save extracted terms and definitions"""
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
        url = "https://ofm.wa.gov/sites/default/files/public/legacy/budget/instructions/glossary.pdf"
        reader = WAGovGlossaryReader(url)
        terms_and_defs = reader.process_pdf()
        
        print(f"\nExtracted {len(terms_and_defs)} terms and definitions.")
        print("\nFirst few entries:")
        for term, definition in terms_and_defs[:5]:
            print(f"\n{term}: {definition}")
    
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

import logging
from pathlib import Path
from typing import List, Tuple
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinePDFReader:
    """Reader for text files with line-based term definitions"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def clean_headers_footers(self, text: str) -> str:
        """Remove headers, footers, and URLs from text"""
        patterns = [
            r'Spring \d{4}.*?RESOURCE.*?\n',
            r'Financial terms glossary.*?\n',
            r'BUILDING BLOCKS.*?RESOURCE.*?\n',
            r'\[\]\(https://.*?\)',
            r'\d+ of \d+.*?\n',
            r'Consumer Financial\s+Protection Bureau.*?\n',
            r'^[A-Z]\s*$'  # Single letter section headers like "A", "B"
        ]

        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    def extract_terms_and_definitions(self, text: str) -> List[Tuple[str, str]]:
        """Extract terms and their multiline definitions"""
        text = self.clean_headers_footers(text)

        # Split into potential term-definition blocks
        blocks = re.split(r'(?<=\.)\s*(?=[A-Z][a-z]|\b[A-Z]\b)', text)
        terms_and_defs = []

        current_term = None
        current_definition = []

        for block in blocks:
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if not lines:
                continue

            # Check if the first line is a valid term
            if re.match(r'^[A-Z][\w\s\(\)\-]+$', lines[0]):
                # Save the previous term-definition pair if valid
                if current_term and current_definition:
                    terms_and_defs.append((current_term, ' '.join(current_definition)))

                # Start a new term-definition pair
                current_term = lines[0]
                current_definition = lines[1:]
            else:
                # If it's not a term, treat it as a continuation of the current definition
                current_definition.extend(lines)

        # Add the last term-definition pair
        if current_term and current_definition:
            terms_and_defs.append((current_term, ' '.join(current_definition)))

        return terms_and_defs

    def process_file(self) -> None:
        """Process text file and rewrite with formatted terms/definitions"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            terms_and_defs = self.extract_terms_and_definitions(content)

            if not terms_and_defs:
                logger.warning("No terms and definitions were extracted!")
                return

            with open(self.file_path, 'w', encoding='utf-8') as f:
                for term, definition in terms_and_defs:
                    f.write(f"Term: {term}\nDefinition: {definition}\n\n")

            logger.info(f"Processed {len(terms_and_defs)} terms and definitions")

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise

def main():
    """Example usage"""
    file_path = "src/financial_pdfs/cfpb_building_block_activities_glossary.txt"
    reader = LinePDFReader(file_path)
    reader.process_file()

if __name__ == "__main__":
    main()

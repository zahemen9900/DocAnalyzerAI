import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlossaryReformatter:
    """Reformats glossary text files according to specified rules"""
    
    def __init__(self, input_file: str):
        self.input_path = Path(input_file)
        self.output_path = self.input_path.with_suffix('.reformatted.txt')
        
        # Words that indicate a line should be merged with previous definition
        self.merge_indicators = {
            'pronouns': {'a', 'an', 'the', 'you', 'it', 'they', 'this', 'that', 'these', 'those'},
            'connectors': {'usually', 'in', 'therefore', 'when', 'can', 'since', 'also', 'if', 'however',
                          'because', 'while', 'although', 'unless', 'moreover', 'furthermore'}
        }

    def should_merge_with_previous(self, term: str) -> bool:
        """Check if term should be merged with previous definition"""
        first_word = term.lower().split()[0] if term else ''
        return (
            first_word in self.merge_indicators['pronouns'] or
            first_word in self.merge_indicators['connectors'] or
            term[0].islower() if term else False
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

    def reformat_glossary(self) -> None:
        """Reformat the glossary file"""
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split into term-definition pairs
            pairs = []
            current_term = None
            current_def = []
            
            # First pass: collect all term-definition pairs
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    if current_term and current_def:
                        pairs.append((current_term, ' '.join(current_def)))
                        current_term = None
                        current_def = []
                    continue

                if line.startswith('Term: '):
                    if current_term and current_def:
                        pairs.append((current_term, ' '.join(current_def)))
                    current_term = line.replace('Term: ', '').strip()
                    current_def = []
                elif line.startswith('Definition: '):
                    current_def.append(line.replace('Definition: ', '').strip())
                else:
                    current_def.append(line)

            # Add last pair if exists
            if current_term and current_def:
                pairs.append((current_term, ' '.join(current_def)))

            # Process pairs and merge where needed
            processed_pairs = []
            i = 0
            while i < len(pairs):
                term, definition = pairs[i]
                merged_definition = definition

                # Look ahead for definitions that need to be merged
                j = i + 1
                while j < len(pairs):
                    next_term, next_def = pairs[j]
                    
                    # Check if next definition starts with lowercase
                    if next_def and next_def[0].islower():
                        # Remove colons and merge
                        clean_next_def = next_def.replace(':', '')
                        merged_definition = f"{merged_definition} {clean_next_def}"
                        j += 1
                    else:
                        break
                
                i = j  # Skip processed pairs
                
                if processed_pairs and self.should_merge_with_previous(term):
                    # Merge with previous definition
                    prev_term, prev_def = processed_pairs[-1]
                    processed_pairs[-1] = (prev_term, f"{prev_def} {term} {merged_definition}")
                else:
                    # Check for colons in definition
                    additional_pairs = self.process_definition_colons(merged_definition)
                    if additional_pairs:
                        processed_pairs.extend(additional_pairs)
                    else:
                        processed_pairs.append((term, merged_definition))

            # Write reformatted content
            with open(self.output_path, 'w', encoding='utf-8') as f:
                for term, definition in processed_pairs:
                    f.write(f"{term}: {definition}\n\n")
            
            logger.info(f"Reformatted glossary saved to {self.output_path}")
            
        except Exception as e:
            logger.error(f"Error reformatting glossary: {e}")
            raise

def main():
    """Example usage""" 
    try:
        input_file = "/home/zahemen/projects/dl-lib/DocAnalyzerAI/src/financial_pdfs/cfpb_building_block_activities_glossary.txt"
        reformatter = GlossaryReformatter(input_file)
        reformatter.reformat_glossary()
    except Exception as e:
        logger.error(f"Failed to reformat glossary: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

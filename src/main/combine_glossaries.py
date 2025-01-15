import logging
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlossaryCombiner:
    """Combines multiple glossary text files into one"""
    
    def __init__(self, pdf_dir: str):
        self.pdf_dir = Path(pdf_dir)
        if not self.pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_dir}")
        
        self.exclude_files = {'cfpb_building_block_activities_glossary.txt'}
        self.output_path = self.pdf_dir / 'combined_glossary.txt'

    def get_text_files(self) -> List[Path]:
        """Get all text files excluding the specified ones"""
        return [
            f for f in self.pdf_dir.glob('*.txt')
            if f.name not in self.exclude_files
        ]

    def combine_files(self) -> None:
        """Combine all text files into one"""
        try:
            text_files = self.get_text_files()
            
            if not text_files:
                logger.warning("No text files found to combine")
                return
            
            # Read and combine content from all files
            combined_content = []
            for file_path in text_files:
                logger.info(f"Processing: {file_path.name}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            combined_content.append(content)
                            combined_content.append("\n" + "="*80 + "\n")  # Separator between files
                except Exception as e:
                    logger.error(f"Error reading file {file_path.name}: {e}")
                    continue
            
            # Write combined content
            if combined_content:
                with open(self.output_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(combined_content))
                
                logger.info(f"Successfully combined {len(text_files)} files into {self.output_path}")
            else:
                logger.warning("No content found to combine")
                
        except Exception as e:
            logger.error(f"Error combining files: {e}")
            raise

def main():
    """Example usage"""
    try:
        pdf_dir = "/home/zahemen/projects/dl-lib/DocAnalyzerAI/src/financial_pdfs"
        combiner = GlossaryCombiner(pdf_dir)
        combiner.combine_files()
    
    except Exception as e:
        logger.error(f"Failed to combine glossaries: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

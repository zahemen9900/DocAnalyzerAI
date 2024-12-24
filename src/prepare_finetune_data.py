import json
import pandas as pd
from datasets import load_dataset
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataPreprocessor:
    def __init__(self, company_tickers_path: str = "../data/company_tickers.json"):
        """Initialize the preprocessor with company ticker mappings"""
        self.company_map = self._load_company_map(company_tickers_path)
        
        # Template patterns for generating conversation data
        self.query_templates = {
            'section_2': [
                "What properties and facilities does {company} own?",
                "Describe the physical assets of {company} for {year}",
                "What are the main operational facilities of {company}?",
                "List the key properties owned by {company} in {year}",
                "Can you detail {company}'s manufacturing facilities?",
            ],
            'section_7': [
                "Analyze {company}'s financial performance for {year}",
                "What are the key financial trends for {company} in {year}?",
                "How did {company} perform financially during {year}?",
                "Explain {company}'s financial condition in {year}",
                "What were the main financial challenges for {company} in {year}?",
            ]
        }

    def _load_company_map(self, filepath: str) -> Dict[str, str]:
        """Load CIK to company name mapping"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return {str(company['cik_str']): company['title'] 
                   for company in data.values()}
        except Exception as e:
            logger.error(f"Error loading company mappings: {e}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean and format text content"""
        if not text:
            return ""
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        # Truncate if too long (Blenderbot context window limitation)
        return text[:4096]  # Adjust size based on model requirements

    def _generate_conversation_pair(self, 
                                  company: str, 
                                  year: str, 
                                  section_type: str, 
                                  content: str) -> List[Dict]:
        """Generate conversation pairs from section content"""
        clean_content = self._clean_text(content)
        if not clean_content:
            return []

        # Select a random template
        template = random.choice(self.query_templates[section_type])
        query = template.format(company=company, year=year)

        # Create conversation data point
        return [{
            'personas': [
                "I am a financial analyst specializing in corporate analysis.",
                "I help people understand company financial information and operations."
            ],
            'context': 'financial_analysis',
            'previous_utterance': [],
            'free_messages': [query],
            'guided_messages': [clean_content],
            'additional_context': f"{company} {year} {section_type}"
        }]

    def process_edgar_data(self, sections: List[str] = ['section_2', 'section_7']) -> List[Dict]:
        """Process EDGAR dataset and generate conversation pairs"""
        try:
            # Load EDGAR dataset
            dataset = load_dataset("eloukas/edgar-10k")['train']
            
            conversation_pairs = []
            for item in dataset:
                cik = str(item['cik'])
                company_name = self.company_map.get(cik, f"Company_{cik}")
                year = str(item['year'])

                # Process each requested section
                for section in sections:
                    if item.get(section):
                        pairs = self._generate_conversation_pair(
                            company_name,
                            year,
                            section,
                            item[section]
                        )
                        conversation_pairs.extend(pairs)

            logger.info(f"Generated {len(conversation_pairs)} conversation pairs")
            return conversation_pairs

        except Exception as e:
            logger.error(f"Error processing EDGAR data: {e}")
            raise

    def save_dataset(self, 
                    conversation_pairs: List[Dict], 
                    output_dir: str = "../data/finetune",
                    train_split: float = 0.8) -> None:
        """Save processed data for fine-tuning"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Split into train/val
            random.shuffle(conversation_pairs)
            split_idx = int(len(conversation_pairs) * train_split)
            train_data = conversation_pairs[:split_idx]
            val_data = conversation_pairs[split_idx:]

            # Save splits
            for name, data in [("train", train_data), ("val", val_data)]:
                with open(output_path / f"{name}.json", 'w') as f:
                    json.dump(data, f, indent=2)

            logger.info(f"Saved {len(train_data)} training and {len(val_data)} validation examples")

        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise

def main():
    preprocessor = FinancialDataPreprocessor()
    conversation_pairs = preprocessor.process_edgar_data()
    preprocessor.save_dataset(conversation_pairs)

if __name__ == "__main__":
    main()

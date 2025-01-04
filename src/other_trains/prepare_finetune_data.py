import json
import pandas as pd
from datasets import load_dataset
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import random
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialDataPreprocessor:
    def __init__(self, company_tickers_path: str = "data/company_tickers.json", year = "2019"):
        """Initialize the preprocessor with company ticker mappings"""
        self.company_map = self._load_company_map(company_tickers_path)
        self.year = year
        
        # Enhanced template patterns with more variations
        self.query_templates = {
            'section_2': [
                # Original templates
                "What properties and facilities does {company} own?",
                "Describe the physical assets of {company} for {year}",
                "What are the main operational facilities of {company}?",
                "List the key properties owned by {company} in {year}",
                "Can you detail {company}'s manufacturing facilities?",
                # Additional variations
                "Tell me about {company}'s physical infrastructure as of {year}",
                "What production facilities does {company} operate?",
                "Could you summarize {company}'s property holdings?",
                "What are the major assets and facilities of {company}?",
                "Give me an overview of {company}'s operational locations",
                "Where are {company}'s main manufacturing plants located?",
                "What real estate does {company} own or lease?",
                "Describe the scale of {company}'s facilities",
                "What is the geographical distribution of {company}'s assets?",
                "Can you list {company}'s primary production sites?",
                "Tell me about {company}'s infrastructure investments",
                "What are {company}'s most significant property assets?",
                "How extensive is {company}'s facility network?",
                "Detail the manufacturing capacity of {company}",
                "What property investments has {company} made in {year}?"
            ],
            'section_7': [
                # Original templates
                "Analyze {company}'s financial performance for {year}",
                "What are the key financial trends for {company} in {year}?",
                "How did {company} perform financially during {year}?",
                "Explain {company}'s financial condition in {year}",
                "What were the main financial challenges for {company} in {year}?",
                # Additional variations
                "Break down {company}'s financial results for {year}",
                "What drove {company}'s financial performance in {year}?",
                "How profitable was {company} during {year}?",
                "Discuss {company}'s operational efficiency in {year}",
                "What were {company}'s major revenue sources in {year}?",
                "Evaluate {company}'s financial health as of {year}",
                "What financial milestones did {company} achieve in {year}?",
                "How did {company} manage its expenses in {year}?",
                "Detail {company}'s cash flow situation for {year}",
                "What were {company}'s key financial metrics in {year}?",
                "Summarize {company}'s annual financial report for {year}",
                "How did {company}'s margins evolve in {year}?",
                "What affected {company}'s bottom line in {year}?",
                "Discuss {company}'s revenue growth strategy in {year}",
                "Analyze {company}'s cost management in {year}",
                "What were {company}'s investment priorities in {year}?",
                "How did {company} allocate its capital in {year}?",
                "Explain {company}'s debt management strategy in {year}",
                "What financial risks did {company} face in {year}?",
                "How did market conditions affect {company} in {year}?"
            ]
        }

        # Add conversation starters templates
        self.conversation_starters = {
            'greetings': {
                'inputs': [
                    "Hey", "Hi", "Hello", "Good morning", "Good afternoon",
                    "Hi there", "Hey there", "Hello there", "Greetings",
                    "Hi, I need help with financial analysis",
                    "Hello, can you help me understand some financial data?",
                    "Hey, I have some questions about company finances",
                    "Hi, I'm looking for financial insights"
                ],
                'responses': [
                    "Hi! I'm your financial analysis assistant. How can I help you today?",
                    "Hello! I'm here to help you with financial analysis. What would you like to know?",
                    "Hi there! I'd be happy to help you analyze financial data. What's on your mind?",
                    "Hello! I specialize in financial analysis and can help you understand company data. What would you like to explore?",
                    "Welcome! I can help you with financial analysis and insights. What would you like to learn about?"
                ]
            },
            'transitions': {
                'inputs': [
                    "Can you help me analyze some data?",
                    "I need to understand company financials",
                    "Could you explain financial metrics?",
                    "I want to learn about company performance"
                ],
                'responses': [
                    "Of course! I'd be happy to help you analyze financial data. What specific aspects would you like to explore?",
                    "Absolutely! I can help you understand company financials. Which metrics are you interested in?",
                    "Sure thing! I can explain financial metrics and help you interpret them. What would you like to know?",
                    "I'd be glad to help you understand company performance. Would you like to start with revenue, profits, or something else?"
                ]
            },
            'financial_inquiry': {
                'inputs': [
                    "I need to analyze a company's financial statements",
                    "Could you help me understand some financial metrics?",
                    "I'm researching company performance data",
                    "How do I interpret financial ratios?",
                    "I want to learn about corporate financial analysis",
                    "Can you guide me through a company's annual report?",
                    "I need help comparing companies financially",
                    "What should I look for in financial statements?",
                    "How can I evaluate a company's financial health?",
                    "I'm studying market performance data"
                ],
                'responses': [
                    "I'd be happy to help you analyze financial statements. What specific aspects would you like to focus on?",
                    "Financial analysis can be complex. Let's break it down step by step. What company are you interested in?",
                    "I can help you understand financial metrics and their implications. Where would you like to start?",
                    "Understanding financial data is crucial. I can guide you through the key indicators. What's your main concern?",
                    "Let's explore the financial analysis together. Would you like to start with revenue, profitability, or something else?",
                    "I can help interpret financial data and trends. Which company or metric interests you most?",
                    "Financial evaluation involves multiple factors. Shall we begin with the basic financial statements?",
                    "I'll help you navigate through the financial information. What specific insights are you looking for?",
                    "We can analyze various financial aspects. Would you like to focus on profitability, efficiency, or financial health?",
                    "Understanding company financials is essential. Let's start with your specific questions or concerns."
                ]
            }
        }

        # Add response starters for more natural conversations
        self.response_starters = [
            "Here's a detailed analysis: \n",
            "Let me break that down for you: \n",
            "Based on the financial data: \n",
            "I'll analyze that for you: \n",
            "Here's what I found in the reports: \n",
            "Let me explain the details: \n",
            "According to the financial statements: \n",
            "Here's the financial breakdown: \n",
            "From my analysis of the data: \n",
            "Looking at the company's reports: \n",
            "Based on my evaluation: \n",
            "Here's what the data shows: \n",
            "Let me provide an analysis: \n",
            "From the financial perspective: \n",
            "After reviewing the documents: \n"
        ]


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
        """Clean and format text content with improved response formatting"""
        if not text:
            return ""

        # Handle different types of apostrophes and quotes
        text = text.replace('\u2019', "'")  # Replace right single quotation
        text = text.replace('\u201C', '"')  # Replace left double quotation
        text = text.replace('\u201D', '"')  # Replace right double quotation
        
        # Remove item numbers and standard headers
        text = re.sub(r'Item\s+\d+\.?\s*', '', text)
        text = re.sub(r'ITEM\s+\d+\.?\s*', '', text)
        
        # Define headers to remove with all possible variations
        base_headers = [
            "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
            "Management's Discussion and Analysis of Financial Condition and Results of Operations",
            "MANAGEMENT'S DISCUSSION AND ANALYSIS",
            "Properties",
        ]
        
        # Generate variations with different prefixes and cases
        headers_to_remove = set()
        prefixes = ['', 'Item 7. ', 'ITEM 7. ', 'Item 2. ', 'ITEM 2. ']
        
        for header in base_headers:
            for prefix in prefixes:
                # Add normal version
                headers_to_remove.add(f"{prefix}{header}")
                # Add lowercase version
                headers_to_remove.add(f"{prefix}{header}".lower())
                # Add uppercase version
                headers_to_remove.add(f"{prefix}{header}".upper())
                
                # Add versions with different types of apostrophes
                if "'" in header:
                    headers_to_remove.add(f"{prefix}{header}".replace("'", "\u2019"))
                    headers_to_remove.add(f"{prefix}{header}".replace("'", "`"))
                    headers_to_remove.add(f"{prefix}{header}".replace("'", "''"))
        
        # Add standalone items
        headers_to_remove.update([
            "Item 7.",
            "ITEM 7",
            "Item 2.",
            "ITEM 2",
            f"For the Fiscal Year Ended {self.year}",
            f"FISCAL YEAR {self.year}",
            f"Fiscal Year {self.year}",
        ])
        
        # Remove headers in a case-insensitive way
        for header in headers_to_remove:
            pattern = re.compile(re.escape(header), re.IGNORECASE)
            text = pattern.sub('', text)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        # Add a random response starter
        response_starter = random.choice(self.response_starters)
        text = f"{response_starter}{text}"
        
        # Truncate if too long
        return text[:4096]

    def _generate_conversation_pair(self, 
                                  company: str, 
                                  year: str, 
                                  section_type: str, 
                                  content: str) -> List[Dict]:
        """Generate conversation pairs matching BlenderBot format"""
        clean_content = self._clean_text(content)
        if not clean_content:
            return []

        # Select a random template
        template = random.choice(self.query_templates[section_type])
        query = template.format(company=company, year=year)

        # Create conversation data point with full BlenderBot format
        return [{
            'personas': [
                "I am a financial analyst specializing in corporate analysis.",
                "I help people understand company financial information and operations."
            ],
            'context': 'financial_analysis',
            'previous_utterance': [],  # Empty list as we're starting fresh conversations
            'free_messages': [query],
            'guided_messages': [clean_content],
            'suggestions': {  # Add empty suggestions structure
                'convai2': [],
                'empathetic_dialogues': [],
                'wizard_of_wikipedia': []
            },
            'guided_chosen_suggestions': [],  # Empty list as we're not using suggestions
            'label_candidates': [],  # Empty list as we're not using candidates
            'additional_context': f"{company} {year} {section_type}"
        }]

    def _generate_conversation_starters(self, num_samples: int) -> List[Dict]:
        """Generate conversation starters with full BlenderBot format"""
        conversation_pairs = []
        categories = ['greetings', 'transitions', 'financial_inquiry']
        
        for _ in range(num_samples):
            category = random.choice(categories)
            templates = self.conversation_starters[category]
            
            query = random.choice(templates['inputs'])
            response = random.choice(templates['responses'])
            
            conversation_pairs.append({
                'personas': [
                    "I am a financial analyst specializing in corporate analysis.",
                    "I help people understand company financial information and operations."
                ],
                'context': 'conversation_starter',
                'previous_utterance': [],
                'free_messages': [query],
                'guided_messages': [response],
                'suggestions': {
                    'convai2': [],
                    'empathetic_dialogues': [],
                    'wizard_of_wikipedia': []
                },
                'guided_chosen_suggestions': [],
                'label_candidates': [],
                'additional_context': 'general_conversation'
            })
        
        return conversation_pairs

    def process_edgar_data(self, sections: List[str] = ['section_2', 'section_7']) -> List[Dict]:
        """Process EDGAR dataset and generate conversation pairs using iterators"""
        try:
            # Load EDGAR dataset with correct source and dynamic year
            dataset = load_dataset(
                "eloukas/edgar-corpus",
                f'year_{self.year}',  # Use the year parameter from class initialization
                split='train',
                trust_remote_code=True
            )
            
            logger.info(f"Loaded dataset for year {self.year} with {len(dataset)} entries")
            
            # Create iterator for the dataset
            dataset_iter = iter(dataset)
            conversation_pairs = []
            
            # Process items using iterator
            while True:
                try:
                    item = next(dataset_iter)
                    try:
                        cik = str(item['cik'])
                        company_name = self.company_map.get(cik, f"Company_{cik}")
                        year = str(item['year'])

                        # Create section iterator
                        section_iter = iter(sections)
                        while True:
                            try:
                                section = next(section_iter)
                                if item.get(section) and item[section]:
                                    pairs = self._generate_conversation_pair(
                                        company_name,
                                        year,
                                        section,
                                        item[section]
                                    )
                                    if pairs:
                                        conversation_pairs.extend(pairs)
                                        logger.debug(
                                            f"Generated pairs for {company_name} "
                                            f"- {year} - {section}"
                                        )
                            except StopIteration:
                                break  # End of sections
                                
                    except Exception as e:
                        logger.warning(f"Skipping item due to error: {e}")
                        continue
                        
                except StopIteration:
                    break  # End of dataset
            
            # Calculate number of starters (30% of dataset)
            num_starters = max(int(len(conversation_pairs) * 0.3), 20)
            
            # Generate conversation starters
            starter_pairs = []
            num_generated = 0
            starter_iter = iter(range(num_starters))
            
            while True:
                try:
                    next(starter_iter)
                    starter_pairs.extend(self._generate_conversation_starters(1))
                    num_generated += 1
                except StopIteration:
                    break
                    
            logger.info(f"Generated {num_generated} conversation starters")
            
            # Combine and shuffle datasets
            conversation_pairs.extend(starter_pairs)
            random.shuffle(conversation_pairs)

            logger.info(
                f"Generated {len(conversation_pairs)} total conversation pairs "
                f"(including {num_generated} conversation starters)"
            )
            return conversation_pairs

        except Exception as e:
            logger.error(f"Error processing EDGAR data: {e}")
            raise

    def save_dataset(self, 
                    conversation_pairs: List[Dict], 
                    output_dir: str = "/home/zahemen/projects/dl-lib/DocAnalyzerAI/finetune_data", train_split: float = 0.8) -> None:
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
    preprocessor = FinancialDataPreprocessor(year = '2017')
    conversation_pairs = preprocessor.process_edgar_data()
    preprocessor.save_dataset(conversation_pairs)

if __name__ == "__main__":
    main()

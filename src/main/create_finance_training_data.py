import json
import random
from pathlib import Path
import logging
from typing import List, Dict, Any
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinanceTrainingDataCreator:
    """Creates conversational training data from financial glossary terms"""
    
    def __init__(self, combined_glossary_path: str, sample_size: int = None):
        self.glossary_path = Path(combined_glossary_path)
        self.output_path = self.glossary_path.parent.parent.parent / 'finetune_data/finance_training_data.json'
        self.sample_size = sample_size  # None means use all terms
        
        # Expand conversation starters
        self.conversation_starters = [
            "Can you help me understand what {term} means?",
            "I've been trying to learn about {term}, could you explain it?",
            "What does {term} mean in finance?",
            "Could you explain {term} in simple terms?",
            "I keep seeing the term {term}, what is it?",
            "Can you teach me about {term}?",
            "What's {term} all about?",
            "I'd like to learn about {term}.",
            "Help me understand {term} please.",
            # Generic starters (no term)
            "Can you help me learn about financial terms?",
            "I'd like to understand finance better.",
            "Could you explain some financial concepts?",
            "I'm trying to improve my financial literacy.",
            "Can you teach me about money and banking?",
            "What are some important financial terms to know?",
            "I need help understanding financial concepts.",
            "Could you explain some banking terms?",
            "What financial terms should I know about?",
            "Can you help me with financial terminology?"
        ]

        # Simple, answerable follow-up messages
        self.followup_messages = [
            "Thanks for explaining that!",
            "That makes sense, thank you.",
            "I understand better now.",
            "Thanks, that was helpful!",
            "Could you explain another term?",
            "That's clear, can you help me with other terms?",
            "Thanks! What other terms should I know about?",
            "That helps a lot, thank you.",
            "Great explanation, thank you!",
            "Thanks, I'd like to learn more terms."
        ]
        
        # Common personas
        self.personas = [
            ["I'm a financial advisor with 15 years of experience.", "I enjoy helping people understand money matters."],
            ["I work at a bank and love explaining financial concepts.", "I believe everyone should understand basic finance."],
            ["I'm studying finance in college.", "I like to share what I learn with others."],
            ["I'm a certified financial planner.", "I specialize in personal finance education."],
            ["I'm new to finance but learning quickly.", "I enjoy discussing financial concepts."],
            ["I'm a financial specialist with 15 years of experience.", "I'm passionate about financial literacy."],
        ]
        
        # Context variations
        self.contexts = ["financial_advice", "banking_basics", "investment_knowledge", "financial_literacy"]

        # Add greeting starters
        self.greeting_starters = [
            "Hi, I'd like to learn about finance.",
            "Hello, Could you help me understand some financial terms?",
            "Good morning! I need help understanding financial concepts.",
            "Hey there. I'm trying to learn about money and banking.",
            "Hi, I'm new to finance and need some guidance.",
            "Hello! I'm looking to improve my financial literacy.",
            "Good afternoon! Can you teach me about financial terms?",
            "Hi! I want to understand banking better.",
            "Hey! I'm interested in learning about finance.",
            "Hello, could you help explain some financial concepts?",
        ]
        
        # Add greeting responses
        self.greeting_responses = [
            "Hello! I'd be happy to help you learn about finance. What would you like to know?",
            "Hi there! Of course, I can help explain financial concepts. Where would you like to start?",
            "Welcome! I'm here to help you understand finance better. What topics interest you?",
            "Hello! I'd love to help you learn. Is there a specific financial term you're curious about?",
            "Hi! Financial literacy is important, and I'm here to help. What would you like to learn first?",
            "Good to meet you! I can explain financial concepts in simple terms. What would you like to know?",
            "Hello! Learning about finance is a great goal. Where shall we begin?",
            "Hi there! I'm happy to guide you through financial concepts. What interests you most?",
            "Welcome! I can help make finance easier to understand. What would you like to learn about?",
            "Hello! Understanding finance is important. Which topics would you like to explore?",
        ]

    def load_glossary_terms(self) -> List[tuple]:
        """Load terms and definitions from combined glossary"""
        terms_and_defs = []
        try:
            with open(self.glossary_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split on separator lines and process each section
            sections = content.split('=' * 80)
            
            for section in sections:
                # Split on double newlines to get term-definition pairs
                pairs = [p.strip() for p in section.split('\n\n') if p.strip()]
                
                for pair in pairs:
                    if ':' in pair:
                        try:
                            term, definition = pair.split(':', 1)
                            # Clean and validate the pair
                            term = term.strip()
                            definition = definition.strip()
                            if term and definition and len(term) > 1 and len(definition) > 5:
                                terms_and_defs.append((term, definition))
                        except Exception as e:
                            logger.warning(f"Skipping malformed pair: {pair[:50]}...")
                            continue
            
            return terms_and_defs
            
        except Exception as e:
            logger.error(f"Error loading glossary: {e}")
            raise

    def create_conversation_pair(self, term: str, definition: str) -> Dict[str, Any]:
        """Create a conversational exchange around a term-definition pair"""
        try:
            # Clean inputs
            term = term.strip()
            definition = definition.strip()
            
            # Decide if this should be a conversation starter (20% chance)
            has_previous = random.random() > 0.2
            
            # Select and format starter
            if '{term}' in random.choice(self.conversation_starters):
                starter = random.choice([s for s in self.conversation_starters if '{term}' in s]).format(term=term)
            else:
                starter = random.choice([s for s in self.conversation_starters if '{term}' not in s])
            
            # Create guided response
            guided_response = f"Let me explain {term}. It is {definition.lower()}."
            
            # Create conversation data
            return {
                'personas': random.choice(self.personas),
                'additional_context': term,
                'context': random.choice(self.contexts),
                'previous_utterance': [starter, guided_response] if has_previous else [],
                'free_messages': [random.choice(self.followup_messages)],
                'guided_messages': [f"I'm glad I could help explain {term}. " + random.choice([
                    "Let me know if you have any other questions.",
                    "Feel free to ask about other terms.",
                    "I'm happy to explain more concepts.",
                    "Would you like to learn about other terms?",
                    "Is there anything else you'd like to know?",
                ])],
                'suggestions': self.create_suggestions(term),
                'guided_chosen_suggestions': self.get_random_suggestions(),
                'label_candidates': []
            }
            
        except Exception as e:
            logger.error(f"Error creating conversation pair for term '{term}': {e}")
            raise

    def create_contextual_response(self, term: str, definition: str) -> str:
        """Create context-aware response"""
        return f"I'm glad I could help explain {term}. " + random.choice([
            "This concept is particularly important in personal finance.",
            "Understanding this can help you make better financial decisions.",
            "Would you like to explore related financial concepts?",
            "Is there anything specific about it you'd like to understand better?",
            "This is a key term in financial planning.",
        ])

    def create_suggestions(self, term: str) -> Dict[str, List[str]]:
        """Create realistic, knowledge-based suggestions"""
        return {
            'financial_advice': [
                "Let me know if you need any other terms explained.",
                "I can help you understand more financial concepts.",
                "Would you like to learn about related terms?",
            ],
            'banking_basics': [
                "I can explain other banking terms if you'd like.",
                "There are many other important terms to learn.",
                "Let me know if you have questions about other terms.",
            ],
            'investment_knowledge': [
                "I'm happy to explain more financial concepts.",
                "Feel free to ask about other terms.",
                "Would you like to learn about other financial terms?",
            ]
        }

    def get_random_suggestions(self) -> List[str]:
        """Generate random combination of suggestion types"""
        # All possible options including empty string
        options = ['financial_advice', 'banking_basics', 'investment_knowledge', '']
        
        # Always include at least one suggestion type
        first = random.choice(['financial_advice', '', ''])        
        # Randomly fill remaining slots
        options = ['banking_basics', 'investment_knowledge', '', '', '']
        remaining = random.sample(options, 2)
        
        return [first] + remaining

    def create_comparison_pair(self, term1: tuple, term2: tuple) -> Dict[str, Any]:
        """Create a comparison-based conversation pair"""
        term1_name, term1_def = term1
        term2_name, term2_def = term2
        
        # Format comparison response
        response = [
            (
                f"Let me explain the difference between {term1_name} and {term2_name}. "
                f"{term1_name} is {term1_def}, whereas {term2_name} is {term2_def}. "
                f"Would you like to know more about either of these concepts?"
            ),  
            (
                    f"I'll explain the difference between {term1_name} and {term2_name}. ",
                    f"{term1_name} is {term1_def}, while {term2_name} is {term2_def}. ",
                    f"Understanding these differences is crucial for making financial decisions. "
            ),
            (
                    f"Let me clarify the difference between {term1_name} and {term2_name}. ",
                    f"{term1_name} is {term1_def}, but {term2_name} is {term2_def}. ",
                    f"Would you like to explore other related financial concepts?"
            )
        ]
        
        return {
            'personas': random.choice(self.personas),
            'additional_context': f"{term1_name} vs {term2_name}",
            'context': random.choice(self.contexts),
            'previous_utterance': [],  # Comparison questions are starters
            'free_messages': [
                random.choice([
                    f"What's the difference between {term1_name} and {term2_name}?",
                    f"Can you explain how {term1_name} differs from {term2_name}?",
                    f"I'm confused about {term1_name} and {term2_name}, can you help?",
                    f"Could you compare {term1_name} and {term2_name}?",
                    f"How are {term1_name} and {term2_name} different?",
                ])
            ],
            'guided_messages': [random.choice(response)],
            'suggestions': {
                'financial_advice': [
                    f"Would you like to learn more about {term1_name}?",
                    f"I can explain more about {term2_name} if you'd like.",
                    "Let me know if you need clarification on either term.",
                ],
                'banking_basics': [
                    "I can explain other related terms.",
                    "There are several other important concepts to understand.",
                    "Would you like to explore similar financial terms?",
                ],
                'investment_knowledge': [
                    "Understanding these differences is crucial for financial decisions.",
                    "There are other related concepts we could explore.",
                    "Would you like to learn about other financial comparisons?",
                ]
            },
            'guided_chosen_suggestions': self.get_random_suggestions(),
            'label_candidates': []
        }

    def create_training_data(self) -> None:
        """Create and save training data from glossary terms"""
        try:
            terms_and_defs = self.load_glossary_terms()
            
            if self.sample_size:
                terms_and_defs = random.sample(terms_and_defs, min(self.sample_size, len(terms_and_defs)))
            
            training_data = []
            
            # Add existing conversation types (greetings, starters, follow-ups)
            # Add greeting conversations (about 10% of total)
            greeting_count = max(len(terms_and_defs) // 10, 5)
            for _ in range(greeting_count):
                greeting_pair = {
                    'personas': random.choice(self.personas),
                    'additional_context': 'financial_literacy',
                    'context': random.choice(self.contexts),
                    'previous_utterance': [],  # No previous utterance for starters
                    'free_messages': [random.choice(self.greeting_starters)],
                    'guided_messages': [random.choice(self.greeting_responses)],
                    'suggestions': self.create_suggestions('financial_literacy'),
                    'guided_chosen_suggestions': self.get_random_suggestions(),
                }
                training_data.append(greeting_pair)
            
            # For each term, create both a starter and a follow-up conversation
            for term, definition in terms_and_defs:
                # Create starter conversation (no previous utterance)
                starter_pair = {
                    'personas': random.choice(self.personas),
                    'additional_context': term,
                    'context': random.choice(self.contexts),
                    'previous_utterance': [],
                    'free_messages': [
                        random.choice([
                            f"Can you explain what {term} means?",
                            f"What is {term}?",
                            f"I keep hearing about {term}, what does it mean?",
                            f"Could you help me understand {term}?",
                            f"What does {term} mean in finance?",
                        ])
                    ],
                    'guided_messages': [
                        f"I'll explain {term}. {definition} Would you like to know more about related concepts?"
                    ],
                    'suggestions': self.create_suggestions(term),
                    'guided_chosen_suggestions': self.get_random_suggestions(),
                    'label_candidates': []
                }
                training_data.append(starter_pair)
                
                # Create follow-up conversation (with previous utterance)
                followup_pair = {
                    'personas': random.choice(self.personas),
                    'additional_context': term,
                    'context': random.choice(self.contexts),
                    'previous_utterance': [
                        random.choice([
                            f"I've heard of {term}, but I'm not sure what it means.",
                            f"Could you tell me more about {term}?",
                            f"What exactly is {term}?",
                            f"I need to understand {term} better.",
                        ]),
                        f"Of course! {term} is {definition}"
                    ],
                    'free_messages': [
                        random.choice([
                            f"Thanks! How does {term} affect my finances?",
                            f"I see. Could you give me an example of {term} in practice?",
                            f"That helps! When should I be concerned about {term}?",
                            f"Interesting! How is {term} related to other financial concepts?",
                            f"Now I understand {term} better. What else should I know about it?"
                        ])
                    ],
                    'guided_messages': [
                        f"I'm glad I could help explain {term}. " + random.choice([
                            "Would you like to learn about related financial concepts?",
                            "I can explain how this applies in different situations.",
                            "There are several important aspects to consider.",
                            "This concept is particularly important for financial planning.",
                            "Understanding this can help with making better financial decisions."
                        ])
                    ],
                    'suggestions': self.create_suggestions(term),
                    'guided_chosen_suggestions': self.get_random_suggestions(),
                    'label_candidates': []
                }
                training_data.append(followup_pair)
            
            # Add comparison-based conversations (about 20% of terms)
            comparison_count = len(terms_and_defs) // 5
            available_terms = terms_and_defs.copy()
            
            for _ in range(comparison_count):
                if len(available_terms) < 2:
                    break
                    
                # Select two random terms that might be related (based on some common words)
                term1 = random.choice(available_terms)
                related_terms = [
                    t for t in available_terms 
                    if t != term1 and (
                        any(word in t[0].lower() for word in term1[0].lower().split())
                        or any(word in t[1].lower() for word in term1[1].lower().split())
                    )
                ]
                
                if related_terms:
                    term2 = random.choice(related_terms)
                else:
                    # If no related terms found, just pick random
                    term2 = random.choice([t for t in available_terms if t != term1])
                
                # Create and add comparison pair
                comparison_pair = self.create_comparison_pair(term1, term2)
                training_data.append(comparison_pair)
                
                # Remove used terms from available pool to avoid repetition
                available_terms.remove(term1)
                if term2 in available_terms:
                    available_terms.remove(term2)
            
            # Continue with existing saving code...
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2)
            
            logger.info(f"Created {len(training_data)} conversation pairs in {self.output_path}")
            
        except Exception as e:
            logger.error(f"Error creating training data: {e}")
            raise

def main():
    """Example usage""" 
    try:
        glossary_path = "/home/zahemen/projects/dl-lib/DocAnalyzerAI/src/financial_pdfs/combined_glossary.txt" 
        # Create 1000 samples (or None for all terms) 
        creator = FinanceTrainingDataCreator(glossary_path, sample_size=4000)
        creator.create_training_data()
    except Exception as e:
        logger.error(f"Failed to create training data: {e}", exc_info=True)
        raise
                     
if __name__ == "__main__":
    main()
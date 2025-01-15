import json
import numpy as np
import random
from typing import List, Dict, Tuple
import logging
from pathlib import Path
import os
import re
from dataclasses import dataclass

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prepare_data.log')
    ]
)
logger = logging.getLogger(__name__)

CONVERSATION_STARTERS = [
   ("Hi there!", "Hello! I'm your financial AI assistant. I can help you with financial analysis, market insights, and answering questions about business and economics. What would you like to know?"),
    ("Hello!", "Hello! I'm here to assist you with any financial queries you might have. Whether it's about investments, market trends, or economic concepts, feel free to ask."),
    ("How can you help me today?", "I can assist you with financial analysis, interpret market trends, analyze company statements, explain financial concepts, and provide insights on economic topics. How may I help you today?"),
    ("What can you do?", "I can assist you with financial analysis, interpret market trends, analyze company statements, explain financial concepts, and provide insights on economic topics. How may I help you today?"),
    ("What do you know about finance?", "I have knowledge about financial analysis, market trends, company statements, financial concepts, and economic topics. Feel free to ask me any questions you have."),
    ("Tell me about yourself.", "I'm a financial AI assistant designed to help you with financial analysis, market insights, and answering questions about business and economics. How can I assist you today?"),
    ("What is your expertise?", "I specialize in financial analysis, market insights, company statements, financial concepts, and economic topics. How can I assist you today?"),
    ("How can you assist me?", "I can help you with financial analysis, interpret market trends, analyze company statements, explain financial concepts, and provide insights on economic topics. How may I help you today?"),
    ("What topics can you help with?", "I can assist you with financial analysis, interpret market trends, analyze company statements, explain financial concepts, and provide insights on economic topics. How may I help you today?"),
    ("What is your area of knowledge?", "I specialize in financial analysis, market insights, company statements, financial concepts, and economic topics. How can I assist you today?"),
    ("Hi there", "Hello! I'm your financial AI assistant. I can help you with financial analysis, market insights, and answering questions about business and economics. What would you like to know?"),
    ("Hello", "Hello! I'm here to assist you with any financial queries you might have. Whether it's about investments, market trends, or economic concepts, feel free to ask."),
    ("How can you help me today", "I can assist you with financial analysis, interpret market trends, analyze company statements, explain financial concepts, and provide insights on economic topics. How may I help you today?"),
    ("What can you do", "I can assist you with financial analysis, interpret market trends, analyze company statements, explain financial concepts, and provide insights on economic topics. How may I help you today?"),
    # Add more conversation starters here myself
]

QUESTION_STARTERS = [
    "Could you explain", 
    "I'd like to understand", 
    "Help me understand", 
    "What do you know about",
    "Tell me more about", 
    "I'm curious about",
    "What exactly is", 
    "How would you define",
    "Can you elaborate on", 
    "Could you clarify",
    "What insights can you provide about", 
    "How does one approach",
    "Could you shed light on", 
    "What can you tell me regarding",
    "How would you describe", 
    "Can you provide an overview of",
    "What is your perspective on", 
    "Could you detail",
    "What factors influence", 
    "How does one interpret",
    "What significance does", 
    "Can you walk me through",
    "What implications does", 
    "How does one assess",
    "What are the key aspects of",
    "Can you discuss",
    "What are the main features of",
    "How does one analyze",
]

RESPONSE_STYLES = {
    "analytical": [
        "Based on historical data and market analysis, {response}",
        "From a quantitative perspective, {response}",
        "Looking at the key metrics and indicators, {response}",
        "When analyzing this in detail, {response}",
        "Taking a data-driven approach, {response}",
        "Considering the financial data available, {response}"
    ],
    "educational": [
        "To understand this concept better, {response}",
        "Here's a clear explanation: {response}",
        "Let me break this down step by step: {response}",
        "The key principle to understand is that {response}",
        "In simple terms, {response}",
        "Here's a basic overview: {response}"
    ],
    "professional": [
        "From a professional financial perspective, {response}",
        "According to established financial principles, {response}",
        "In the current market environment, {response}",
        "Taking into account various factors, {response}",
        "In the context of financial analysis, {response}",
        "Considering the economic outlook, {response}"
    ]
}

FOLLOW_UP_PATTERNS = [
    "What implications does this have for {topic}?",
    "How does this relate to {topic} in practice?",
    "Could you elaborate on how {topic} affects investment decisions?",
    "What are the key risks associated with {topic}?",
    "How can investors best navigate {topic}?",
    "What strategies would you recommend for {topic}?"
]

def generate_variations(question: str, answer: str, max_variations: int = 3) -> List[Tuple[str, str]]:
    """Generate limited variations of Q&A pairs"""
    variations = [(question, answer)]
    
    # Extract core topic and clean it
    core_topic = re.sub(r'^(What is|Define|Explain|How does|Tell me about)\s+', '', question).strip('?. ')
    
    # Select a random subset of starters
    selected_starters = random.sample(QUESTION_STARTERS, min(max_variations, len(QUESTION_STARTERS)))
    
    # Generate variations with deduplication
    seen = {question.lower()}
    for starter in selected_starters:
        var_question = f"{starter} {core_topic}?"
        if var_question.lower() not in seen:
            seen.add(var_question.lower())
            variations.append((var_question, answer))
            
    return variations

def enhance_qa_variation(question: str, answer: str) -> List[Tuple[str, str]]:
    """Generate enhanced variations of QA pairs with different styles"""
    variations = []
    base_answer = answer.strip()
    
    # Add style variations
    for style, templates in RESPONSE_STYLES.items():
        styled_answer = random.choice(templates).format(response=base_answer.lower())
        variations.append((question, styled_answer))
    
    # Add contextual variations
    market_conditions = ["in a bull market", "during market volatility", 
                        "in a bear market", "during economic uncertainty"]
    context = random.choice(market_conditions)
    contextual_q = f"How does {question.rstrip('?')} apply {context}?"
    contextual_a = f"Specifically {context}, {base_answer.lower()}"
    variations.append((contextual_q, contextual_a))
    
    return variations

FINANCIAL_QA_SAMPLES = [
    ("What is ROI?", "ROI (Return on Investment) is a performance metric used to evaluate the efficiency of an investment. It's calculated by dividing the net profit by the cost of investment and expressing it as a percentage. For example, if you invest $1000 and earn $1200, your ROI is 20%."),
    ("Explain market capitalization.", "Market capitalization, or market cap, represents the total value of a company's shares in the market. It's calculated by multiplying the current share price by the total number of outstanding shares. Companies are often classified as large-cap (>$10B), mid-cap ($2-10B), or small-cap (<$2B)."),
    ("What is a balance sheet?", "A balance sheet is a financial statement that provides a snapshot of a company's financial position at a specific point in time. It shows the company's assets, liabilities, and shareholders' equity. Assets are what the company owns, liabilities are what it owes, and equity represents the shareholders' ownership."),
    ("How do you calculate EBITDA?", "EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization) is calculated by adding back interest, taxes, depreciation, and amortization to net income. It's used to assess a company's operating performance without the impact of financing decisions, accounting practices, or tax environments."),
    ("What is a P/E ratio?", "The P/E (Price-to-Earnings) ratio is a valuation metric that compares a company's current share price to its earnings per share (EPS). It indicates how much investors are willing to pay for each dollar of earnings. A high P/E ratio may suggest overvaluation, while a low P/E ratio may indicate undervaluation."),
    ("Define GDP.", "GDP (Gross Domestic Product) is the total monetary value of all goods and services produced within a country's borders in a specific period. It's used to measure the economic performance and size of an economy. GDP can be calculated using three approaches: production, income, and expenditure."),
    ("What is a bull market?", "A bull market is a financial market characterized by rising asset prices and investor optimism. It's typically associated with strong economic performance, high employment, and increasing corporate profits. Bull markets are marked by sustained periods of upward price trends."),
    ("Explain the concept of diversification.", "Diversification is an investment strategy that involves spreading your investments across different assets to reduce risk. By investing in a variety of assets, sectors, or geographic regions, you can minimize the impact of a single investment's performance on your overall portfolio."),
    ("What is inflation?", "Inflation is the rate at which the general level of prices for goods and services rises, leading to a decrease in purchasing power. It's measured by the Consumer Price Index (CPI) and can erode the value of money over time. Central banks aim to maintain low and stable inflation rates."),
    ("How do you calculate compound interest?", "Compound interest is calculated by applying the interest rate to the initial principal amount and any accumulated interest. The formula for compound interest is A = P(1 + r/n)^(nt), where A is the future value of the investment, P is the principal amount, r is the annual interest rate, n is the number of times interest is compounded per year, and t is the number of years."),
    ("What is a stock market index?", "A stock market index is a benchmark that measures the performance of a group of stocks in a particular market. It's used to track the overall performance of the market, compare investment returns, and analyze economic trends. Examples of stock market indices include the S&P 500, Dow Jones Industrial Average, and Nasdaq Composite."),
    ("Define a recession.", "A recession is a significant decline in economic activity that lasts for an extended period. It's characterized by falling GDP, rising unemployment, reduced consumer spending, and declining business investment. Recessions are typically caused by factors such as reduced consumer confidence, financial crises, or external shocks."),
    ("What is a dividend?", "A dividend is a distribution of a portion of a company's earnings to its shareholders. It's usually paid in cash or additional shares and is based on the company's profitability and dividend policy. Dividends provide investors with a source of income and can be an indicator of a company's financial health."),
    ("Explain the concept of supply and demand.", "Supply and demand is an economic model that describes the relationship between the availability of a product or service and the desire for it. When supply exceeds demand, prices tend to fall, and when demand exceeds supply, prices tend to rise. The interaction of supply and demand determines the equilibrium price and quantity in a market."),
    ("What is a 401(k) retirement plan?", "A 401(k) retirement plan is a tax-advantaged investment account offered by employers to help employees save for retirement. Employees can contribute a portion of their pre-tax income to the account, and employers may match a percentage of the contributions. 401(k) plans offer investment options such as stocks, bonds, and mutual funds."),
    ("Define a mutual fund.", "A mutual fund is an investment vehicle that pools money from multiple investors to invest in a diversified portfolio of stocks, bonds, or other securities. Mutual funds are managed by professional fund managers and offer investors access to a diversified investment portfolio without the need to select individual securities."),
    ("What is a credit score?", "A credit score is a numerical representation of an individual's creditworthiness based on their credit history. It's used by lenders to assess the risk of lending money to a borrower and determine the terms of the loan. Credit scores are calculated using factors such as payment history, credit utilization, length of credit history, new credit accounts, and credit mix."),
    ("Explain the concept of risk management.", "Risk management is the process of identifying, assessing, and prioritizing risks to minimize their impact on an organization's objectives. It involves analyzing potential risks, developing strategies to mitigate or avoid them, and monitoring the effectiveness of risk controls. Effective risk management helps organizations anticipate and respond to threats, opportunities, and uncertainties."),
    ("What is a budget?", "A budget is a financial plan that outlines an individual's or organization's income and expenses over a specific period. It helps track spending, allocate resources, and achieve financial goals. Budgets can be used for personal finance, business planning, project management, and government operations."),
    # Add hundreds of more financial Q&A samples here myself
]


def ensure_directory_exists(filepath: str):
    """Create directory if it doesn't exist"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {Path(filepath).parent}")

def truncate_text(text: str, max_words: int = 40) -> str:
    """Truncate text to a maximum number of words while maintaining coherence"""
    words = text.split()
    if len(words) <= max_words:
        return text
        
    # Try to find a good breakpoint (period or semicolon)
    truncated = ' '.join(words[:max_words])
    last_period = truncated.rfind('.')
    last_semicolon = truncated.rfind(';')
    
    break_point = max(last_period, last_semicolon)
    if break_point > len(truncated) // 2:  # Only use if break point is in latter half
        return truncated[:break_point + 1]
    
    return truncated + '.'

def clean_text(text: str) -> str:
    """Enhanced text cleaning with pattern removal"""
    # Remove problematic patterns
    patterns_to_remove = [
        r"Financial Experience is.*?[.]",
        r"Personal finance is.*?[.]",
        # r"I am a financial advisor.*?[.]",
        # r"Do you know anyone.*?[?]",
        r"I have .* saved.*?[.]",
        r"My (?:husband|wife|ex-wife).*?[.]"
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text)
    
    # Existing cleaning
    text = re.sub(r'^Assistant:\s*', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'\s*;\s*', '; ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def generate_variations(question: str, answer: str, max_variations: int = 3) -> List[Tuple[str, str]]:
    """Generate limited variations of Q&A pairs"""
    variations = [(question, answer)]
    
    # Extract core topic and clean it
    core_topic = re.sub(
        r'^(What is|Define|Explain|How does|Tell me about|Can you explain to me|Explain to me|Describe|Give me an overview of|Provide an explanation of|Could you explain|Help me understand|I want to know about|What are)\s+',
        '',
        question
    ).strip('?. ')    
    # Select a random subset of starters
    selected_starters = random.sample(QUESTION_STARTERS, min(max_variations, len(QUESTION_STARTERS)))
    
    # Generate variations with deduplication
    seen = {question.lower()}
    for starter in selected_starters:
        var_question = f"{starter} {core_topic}?"
        if var_question.lower() not in seen:
            seen.add(var_question.lower())
            variations.append((var_question, answer))
            
    return variations

def create_domain_specific_samples() -> List[Dict]:
    """Create additional domain-specific samples with enhanced variation"""
    samples = [
        {
            "question": "What is cryptocurrency mining?",
            "answer": "Cryptocurrency mining is the process of validating and adding new transactions to a blockchain using powerful computers to solve complex mathematical problems. Miners are rewarded with new coins for their work, which helps secure the network and process transactions."
        },
        {
            "question": "How does market sentiment affect stock prices?",
            "answer": "Market sentiment refers to the overall attitude or feeling that investors have toward a particular security, sector, or market. It can significantly impact stock prices through trading behavior, with positive sentiment driving prices up and negative sentiment pushing them down."
        },
        {
            "question": "What is dollar-cost averaging?",
            "answer": "Dollar-cost averaging is an investment strategy where an investor consistently buys a fixed dollar amount of a particular asset at regular intervals, regardless of its price. This approach reduces the impact of market volatility and lowers the average cost per share over time."
        },
        {
            "question": "What is an ETF (Exchange-Traded Fund)?",
            "answer": "An ETF is an investment fund traded on stock exchanges, similar to stocks. It holds assets like stocks, bonds, or commodities and offers investors diversification, liquidity, and cost efficiency."
        },
        {
            "question": "What is the difference between growth stocks and value stocks?",
            "answer": "Growth stocks are shares of companies expected to grow at a rate faster than the market average, often reinvesting profits. Value stocks are undervalued by the market and typically pay dividends, offering potential returns through stock price appreciation."
        },
        {
            "question": "What are blue-chip stocks?",
            "answer": "Blue-chip stocks are shares of large, well-established, and financially sound companies with a history of reliable performance, often paying consistent dividends."
        },
        {
            "question": "What is liquidity in finance?",
            "answer": "Liquidity refers to how quickly and easily an asset can be converted into cash without significantly affecting its price. Cash is the most liquid asset, while real estate is considered less liquid."
        },
        {
            "question": "How do interest rates affect bond prices?",
            "answer": "Interest rates and bond prices have an inverse relationship. When interest rates rise, existing bond prices fall because new bonds offer higher yields. Conversely, when rates drop, bond prices increase."
        },
        {
            "question": "What is the difference between a bull market and a bear market?",
            "answer": "A bull market refers to a period of rising stock prices and positive investor sentiment, while a bear market indicates falling stock prices and widespread pessimism among investors."
        },
        {
            "question": "What is compound interest?",
            "answer": "Compound interest is the interest calculated on both the initial principal and the accumulated interest from previous periods. It allows investments to grow exponentially over time."
        },
        {
            "question": "What is the role of a financial advisor?",
            "answer": "A financial advisor provides guidance on financial planning, investment management, retirement planning, tax strategies, and other financial matters to help clients achieve their financial goals."
        },
        {
            "question": "What is a credit score?",
            "answer": "A credit score is a numerical representation of an individual's creditworthiness based on their credit history. It's used by lenders to assess the risk of lending money to a borrower and determine the terms of the loan."
        }
        # Add hundreds of more domain-specific samples here myself
    ]
  
    enhanced_samples = []
    for sample in samples:
        # Add original sample
        enhanced_samples.append({
            "personas": ["Financial Expert"],
            "free_messages": [sample["question"]],
            "guided_messages": [sample["answer"]]
        })
        
        # Add variations with different styles
        variations = enhance_qa_variation(sample["question"], sample["answer"])
        for var_q, var_a in variations:
            enhanced_samples.append({
                "personas": ["Financial Expert"],
                "free_messages": [var_q],
                "guided_messages": [var_a]
            })
            
        # Add follow-up questions
        topic = re.sub(r'^(?:what|how|why|explain|define)\s+(?:is|are|does)\s+', '', 
                      sample["question"].lower().strip('?'))
        followup_q = random.choice(FOLLOW_UP_PATTERNS).format(topic=topic)
        followup_a = f"Regarding {topic}, {sample['answer']}"
        
        enhanced_samples.append({
            "personas": ["Financial Expert"],
            "previous_utterance": [sample["question"]],
            "free_messages": [followup_q],
            "guided_messages": [followup_a]
        })
    
    return enhanced_samples

def create_multi_turn_conversations(base_qa_pairs: List[Tuple[str, str]], max_turns: int = 3) -> List[Dict]:
    """Create multi-turn conversations with limited samples"""
    conversations = []
    
    # Limit the number of conversation chains
    max_chains = 50
    base_qa_pairs = base_qa_pairs[:max_chains * max_turns]
    
    for i in range(0, len(base_qa_pairs), max_turns):
        turns = base_qa_pairs[i:i + max_turns]
        context = []
        
        for j, (q, a) in enumerate(turns):
            # Format previous context as a clean string
            if context:
                previous_context = "\n".join([
                    f"Previous Q: {prev_q}\nPrevious A: {prev_a}"
                    for prev_q, prev_a in zip(context[::2], context[1::2])
                ])
                
                entry = {
                    "personas": ["Financial Expert"],
                    "previous_utterance": previous_context,
                    "free_messages": [q],
                    "guided_messages": [a]
                }
            else:
                entry = {
                    "personas": ["Financial Expert"],
                    "previous_utterance": [],
                    "free_messages": [q],
                    "guided_messages": [a]
                }
            
            conversations.append(entry)
            context.extend([q, a])
    
    return conversations

def generate_followup_questions(qa_pair: Tuple[str, str]) -> List[Tuple[str, str]]:
    """Generate natural followup questions based on the initial QA pair"""
    question, answer = qa_pair
    followups = []
    
    # Extract key terms for followups
    key_terms = extract_financial_terms(answer)
    
    # Generate followup patterns
    followup_templates = [
        "Can you elaborate on {term}?",
        "How does {term} relate to market performance?",
        "What are the risks associated with {term}?",
        "Could you provide an example of {term} in action?",
        "What are the best practices for managing {term}?"
    ]
    
    for term in key_terms[:2]:  # Limit to 2 followups per term
        for template in followup_templates[:2]:  # Limit templates
            followup_q = template.format(term=term)
            # Use OpenAI API or similar to generate contextual answers
            followup_a = generate_contextual_answer(followup_q, context=answer)
            followups.append((followup_q, followup_a))
    
    return followups

def extract_financial_terms(text: str) -> List[str]:
    """Extract key financial terms from text using regex and financial lexicon"""
    # Financial term patterns
    patterns = [
        r'\b(?:stock|bond|market|investment|portfolio|dividend|equity|asset|liability)\w*\b',
        r'\b(?:ROI|P/E|EPS|EBITDA|GDP|IPO)\b',
        r'\b(?:bull|bear|volatile|leverage|hedge|risk|return)\w*\b',
        r'\b(?:crypto|bitcoin|blockchain|cryptocurrency|token)\b',
        r'\b(?:interest|inflation|yield|diversification|liquidity)\w*\b',
        r'\b(?:mutual fund|ETF|index fund|401k|IRA|ROTH)\b',
        r'\b(?:credit score|credit card|credit limit|credit report)\b',
    ]
    
    terms = set()
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        terms.update(match.group() for match in matches)
    
    return list(terms)

def generate_contextual_answer(question: str, context: str) -> str:
    """Generate more focused and professional responses"""
    term = extract_financial_terms(context)
    
    # Create more professional response templates
    analysis_examples = [
        "{term} refers to {context}.",
        "In financial terms, {term} refers to {context}.",
        "{term} is {context}."
    ]

    templates = {
        "definition": "In financial terms, {term} refers to {context}.",
        "explanation": "{term} is a fundamental concept in finance that {context}.",
        "analysis": f"{random.choice(analysis_examples)}",
    }
    
    # Choose appropriate template based on question type
    if "what is" in question.lower() or "define" in question.lower():
        template = templates["definition"]
    elif "explain" in question.lower() or "how" in question.lower():
        template = templates["explanation"]
    else:
        template = templates["analysis"]
    
    term = re.sub(r'^(?:what|how|why|could you|can you|explain|define)\s+(?:is|are|does|do|the|a|an)\s+', '', 
                  question.lower().replace('?', ''))
    
    return template.format(term=term, context=context)

def augment_dataset_with_variations(data: List[Dict]) -> List[Dict]:
    """Improved dataset augmentation with enhanced variations"""
    augmented_data = []
    
    for item in data:
        end_samples = [
            "This is important considering the current economic environment and market trends.",
            "This is particularly relevant given current economic conditions.",
            "This is crucial for making informed investment decisions in today's market environment.",
            "This is essential for adapting to the changing market dynamics and making informed investment decisions.",
            "Does this clarify things for you?",
            "Would you like more information on this topic?",
            "Does this clear things up for you?",
            "", "", "", "", "", "", "", "" # Increase likelihood of no context end
            ]
        # Add original item
        augmented_data.append(item)
        
        # Add style variations
        if len(item['free_messages'][0]) > 20:
            for style, templates in RESPONSE_STYLES.items():
                styled_response = random.choice(templates).format(
                    response=item['guided_messages'][0].lower()
                )
                variation = {
                    "personas": item["personas"],
                    "previous_utterance": [],
                    "free_messages": [item['free_messages'][0]],
                    "guided_messages": [styled_response]
                }
                augmented_data.append(variation)
        
        # Add clarification requests with better formatting
        if len(item['free_messages'][0]) > 20:
            terms = extract_financial_terms(item['free_messages'][0])
            if terms:
                answer_starter = random.choice([
                    f"Let me break down {terms[0]} more clearly.",
                    f"To clarify {terms[0]},",
                    f"Here's a more detailed explanation of {terms[0]}:",
                    f"To elaborate on {terms[0]},",
                    "To clarify further,",
                    "In simpler terms,",
                    "In the context of finance,",
                    "", "", "", "", "", "", "" # Increase likelihood of no starter

                ])
                answer_end = random.choice(end_samples)

                question = f"Could you explain {terms[0]} in more detail?"
                if answer_starter and answer_end:
                    answer = f"{answer_starter} {item['guided_messages'][0]} {answer_end}"
                elif answer_starter and not answer_end:
                    answer = f"{answer_starter} {item['guided_messages'][0]}"
                elif not answer_starter and answer_end:
                    answer = f"{item['guided_messages'][0]} {answer_end}"
                else:
                    answer = f"{item['guided_messages'][0]}"
                
                clarification = {
                    "personas": item["personas"],
                    "previous_utterance": item['free_messages'],
                    "free_messages": [question],
                    "guided_messages": [answer]
                }
                augmented_data.append(clarification)
        
        # Add market context variations with improved responses
        if any(term in item['guided_messages'][0].lower() for term in ['market', 'investment', 'stock', 'bond', 'portfolio', 'asset', 'liability']):
            context_question = "How does this concept apply in current market conditions?"
            starter = random.choice([
                "Given the current market", "In today's economic landscape",
                "In today's market", "In simple terms",
                "In the context of finance,",
                "Given current market dynamics",
                "In light of recent market trends",
                "Given the current investment climate",
                "In the context of market volatility",
                "", "", "", "", "", "" # Increase likelihood of no starter
                 
            ]
            )
            context_end = random.choice(end_samples)
            if starter and context_end:
                context_answer = (
                    f"{starter}, {item['guided_messages'][0].lower()} "
                    f"{context_end}"
                )
            elif starter and not context_end:
                context_answer = f"{starter}, {item['guided_messages'][0].lower()}"
            
            elif not starter and context_end:
                context_answer = f"{item['guided_messages'][0]} {context_end}"
            else:
                context_answer = item['guided_messages'][0]
            
            context_variation = {
                "personas": item["personas"],
                "previous_utterance": [item['free_messages'][0]],
                "free_messages": [context_question],
                "guided_messages": [context_answer]
            }
            augmented_data.append(context_variation)
    
    return augmented_data

@dataclass
class ConversationTemplate:
    context: str
    possible_responses: List[str]
    followup_questions: List[str]

# Add more natural conversation flows
CONVERSATION_FLOWS = {
    "risk_assessment": ConversationTemplate(
        context="Discussion about investment risk tolerance and portfolio management",
        possible_responses=[
            "Based on what you've described, your risk tolerance appears to be {risk_level}. This suggests a portfolio with {allocation} might be suitable.",
            "Let's analyze your comfort level with market fluctuations. {explanation}",
            "Understanding your risk tolerance is crucial for building the right portfolio. {details}",
            "Your risk profile indicates a preference for {allocation} investments. Here's why:"
        ],
        followup_questions=[
            "What's your investment timeline?",
            "How would you react to a 20% market drop?",
            "What's your primary investment goal?",
            "Have you considered {alternative} for your portfolio?"
        ]
    ),
    "market_volatility": ConversationTemplate(
        context="Addressing concerns about market conditions and volatility",
        possible_responses=[
            "Market volatility is normal and can actually present opportunities. Here's why: {explanation}",
            "Let's look at historical patterns to put current market conditions in perspective. {analysis}",
            "While volatility can be concerning, maintaining a long-term perspective is key because {reason}",
            "Market corrections can be unsettling, but it's important to remember that {explanation}"
        ],
        followup_questions=[
            "What specific market sectors are you most concerned about?",
            "Have you considered diversifying into {alternative}?",
            "How has your portfolio performed during previous market corrections?",
            "What additional information would help you feel more confident?"
            "What's your primary investment goal?",
        ]
    )
}

def generate_dynamic_response(template: str, context: Dict[str, str]) -> str:
    """Generate more natural responses using templates and context"""
    replacements = {
        "risk_level": random.choice(["conservative", "moderate", "aggressive"]),
        "allocation": random.choice([
            "60% bonds and 40% stocks",
            "70% stocks and 30% bonds",
            "a balanced mix of growth and value stocks"
        ]),
        "explanation": random.choice([
            "Historical data shows that markets tend to recover over time.",
            "Diversification can help manage risk while maintaining growth potential.",
            "A well-balanced portfolio can help weather market volatility."
            "Long-term investments have historically outperformed short-term strategies."
            "Staying invested during market downturns can lead to better returns."
        ]),
        "analysis": random.choice([
            "Looking at previous market cycles...",
            "When we examine similar situations in the past...",
            "Market data indicates that...",
            "Historical trends suggest that..."
            "Based on prior performance..."
        ]),
        "alternative": random.choice([
            "defensive sectors",
            "dividend-paying stocks",
            "fixed-income securities"
        ]),
        "details": context.get("additional_info", ""),
        "reason": context.get("market_context", "")
    }
    
    return template.format(**replacements)

def create_natural_conversation(flow_type: str, context: Dict[str, str]) -> List[Dict]:
    """Create more natural conversation flows"""
    template = CONVERSATION_FLOWS[flow_type]
    conversation = []
    
    # Initial response
    initial_response = generate_dynamic_response(
        random.choice(template.possible_responses),
        context
    )
    
    conversation.append({
        "personas": ["Financial Expert"],
        "previous_utterance": [],
        "free_messages": [template.context],
        "guided_messages": [initial_response]
    })
    
    # Add natural followups
    for question in random.sample(template.followup_questions, 2):
        followup_response = generate_dynamic_response(
            random.choice(template.possible_responses),
            context
        )
        
        conversation.append({
            "personas": ["Financial Expert"],
            "previous_utterance": [conversation[-1]["free_messages"][0]],
            "free_messages": [question],
            "guided_messages": [followup_response]
        })
    
    return conversation

# Update create_enhanced_dataset function
def create_enhanced_dataset(
    output_file: str,
    conversation_ratio: float = 0.6,
    qa_ratio: float = 0.4,
    max_samples: int = 5_000,  # Increased for more samples
    max_variations: int = 7,   # Limit variations per QA pair
    max_followups: int = 3,    # Limit followup questions
    max_words_per_response: int = 40
):
    """Enhanced dataset creation with improved diversity"""
    try:
        logger.info("Starting enhanced dataset creation...")
        enhanced_data = []
        
        # Create different conversation categories
        financial_planning = []
        market_analysis = []
        investment_advice = []
        risk_management = []
        
        # Categorize conversation starters
        for starter, response in CONVERSATION_STARTERS:
            entry = {
                "personas": ["Financial Assistant"],
                "previous_utterance": [],
                "free_messages": [starter],
                "guided_messages": [response]
            }
            
            if "risk" in starter.lower() or "volatility" in starter.lower():
                risk_management.append(entry)
            elif "market" in starter.lower() or "analysis" in starter.lower():
                market_analysis.append(entry)
            elif "invest" in starter.lower() or "portfolio" in starter.lower():
                investment_advice.append(entry)
            else:
                financial_planning.append(entry)
        
        # Calculate conversation samples target (30% of max_samples)
        conv_target = int(max_samples * conversation_ratio)
        conv_per_category = conv_target // 4  # Split evenly among categories
        
        # Balance conversation categories
        for category in [financial_planning, market_analysis, investment_advice, risk_management]:
            if category:  # Check if category is not empty
                # Use modulo to cycle through samples if needed
                samples = [category[i % len(category)] for i in range(conv_per_category)]
                enhanced_data.extend(samples)
                logger.info(f"Added {len(samples)} samples from conversation category")
        
        # Process QA samples with enhanced variations
        qa_samples = []
        seen_questions = set()
        
        for question, answer in FINANCIAL_QA_SAMPLES:
            # Skip if we've reached the desired QA count
            if len(qa_samples) >= max_samples * qa_ratio:
                break
                
            clean_answer = clean_text(answer)
            truncated_answer = truncate_text(clean_answer, max_words_per_response)
            
            # Generate standard variations
            variations = generate_variations(question, truncated_answer, max_variations)
            
            # Generate enhanced variations
            enhanced_variations = enhance_qa_variation(question, truncated_answer)
            variations.extend(enhanced_variations)
            
            # Add variations with deduplication
            for var_q, var_a in variations:
                if var_q.lower() not in seen_questions:
                    seen_questions.add(var_q.lower())
                    qa_samples.append({
                        "personas": ["Financial Expert"],
                        "previous_utterance": [],
                        "free_messages": [var_q],
                        "guided_messages": [var_a]
                    })
        
        # Limit domain-specific samples
        additional_samples = create_domain_specific_samples()[:max_samples//10]
        qa_samples.extend(additional_samples)
        
        # Calculate complexity scores for sorting
        complexity_scores = {}
        for sample in qa_samples:
            question = sample["free_messages"][0]
            answer = sample["guided_messages"][0]
            complexity = len(answer.split()) + len([w for w in answer.split() if len(w) > 8])
            complexity_scores[question] = complexity
        
        # Sort QA samples by complexity
        qa_samples.sort(key=lambda x: complexity_scores[x["free_messages"][0]])
        
        # Calculate desired number of QA samples (70% of max_samples)
        desired_qa_count = min(int(max_samples * qa_ratio), len(qa_samples))
        
        # Safely sample QA examples
        if desired_qa_count > 0:
            # Use evenly spaced indices to get a representative sample
            indices = np.linspace(0, len(qa_samples)-1, desired_qa_count, dtype=int)
            selected_qa_samples = [qa_samples[i] for i in indices]
            enhanced_data.extend(selected_qa_samples)
            logger.info(f"Added {len(selected_qa_samples)} QA samples")
        
        # Add limited multi-turn conversations
        qa_pairs = [(item["free_messages"][0], item["guided_messages"][0]) 
                   for item in qa_samples[:max_samples//5]]
        multi_turn_samples = create_multi_turn_conversations(qa_pairs, max_turns=2)
        enhanced_data.extend(multi_turn_samples[:max_samples//4])
        
        # Add limited followup questions
        for qa_pair in zip(qa_samples[:max_samples//10], 
                         [sample["guided_messages"][0] for sample in qa_samples[:max_samples//10]]):
            followups = generate_followup_questions(qa_pair)[:max_followups]
            for q, a in followups:
                if len(enhanced_data) < max_samples:
                    enhanced_data.append({
                        "personas": ["Financial Expert"],
                        "previous_utterance": [qa_pair[0]],
                        "free_messages": [q],
                        "guided_messages": [a]
                    })
        
        # Augment with variations
        enhanced_data = augment_dataset_with_variations(enhanced_data)
        
        # Limit final dataset size
        if len(enhanced_data) > max_samples:
            enhanced_data = random.sample(enhanced_data, max_samples)
        
        # Shuffle the final dataset
        random.shuffle(enhanced_data)
        
        # Ensure output directory exists
        ensure_directory_exists(output_file)
        
        # Save enhanced dataset
        logger.info(f"Saving dataset to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        
        # Log dataset statistics
        categories = {
            "Financial Planning": len(financial_planning),
            "Market Analysis": len(market_analysis),
            "Investment Advice": len(investment_advice),
            "Risk Management": len(risk_management),
            "QA Samples": len(qa_samples),
            "Total Samples": len(enhanced_data)
        }
        logger.info("Dataset composition:")
        for category, count in categories.items():
            logger.info(f"{category}: {count} samples")
            
        return enhanced_data
        
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}", exc_info=True)
        raise

def main():
    try:
        # Get the absolute path to the project root
        project_root = Path(__file__).parent.parent.parent
                # Define output paths
        output_dir = project_root / "finetune_data"
        train_file = output_dir / "train.json"
        val_file = output_dir / "val.json"
        
        logger.info(f"Project root: {project_root}")
        logger.info(f"Output directory: {output_dir}")
        
        # Create the enhanced dataset
        enhanced_data = create_enhanced_dataset(str(train_file))
        
        # Split into train/val sets
        random.shuffle(enhanced_data)
        split_idx = int(len(enhanced_data) * 0.7)  # 70/30 split
        
        train_data = enhanced_data[:split_idx]
        val_data = enhanced_data[split_idx:]
        
        # Save train set
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(train_data)} training examples to {train_file}")
        
        # Save validation set
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(val_data)} validation examples to {val_file}")
        
        logger.info("Dataset preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

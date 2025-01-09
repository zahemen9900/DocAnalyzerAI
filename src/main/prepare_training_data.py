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
    ("How do you work?", "I'm trained to analyze financial data, answer questions about economics and business, and help with financial decision-making. I can examine financial statements, discuss market trends, and explain complex financial concepts in simple terms."),
    ("Good day!", "Good day! I'm here to assist you with any financial queries you might have. Whether it's about investments, market trends, or economic concepts, feel free to ask."),
    ("What can you do?", "I can assist you with financial analysis, interpret market trends, analyze company statements, explain financial concepts, and provide insights on economic topics. How may I help you today?"),
    ("How do you work?", "I'm trained to analyze financial data, answer questions about economics and business, and help with financial decision-making. I can examine financial statements, discuss market trends, and explain complex financial concepts in simple terms."),
    ("Good day!", "Good day! I'm here to assist you with any financial queries you might have. Whether it's about investments, market trends, or economic concepts, feel free to ask."),
    ("Can you help me with investments?", "Absolutely! I can provide information on various investment options, analyze potential returns, and explain associated risks to help you make informed decisions."),
    ("What's the latest in the financial markets?", "I can provide updates on recent market movements, economic indicators, and financial news to keep you informed about the latest developments."),
    ("How can I improve my financial literacy?", "I can explain financial concepts, recommend resources, and answer any questions you have to help enhance your understanding of finance and economics."),
    ("Can you assist with budgeting?", "Yes, I can offer tips on creating a budget, managing expenses, and setting financial goals to help you maintain healthy financial habits."),
    ("What are the current interest rates?", "I can provide information on current interest rates for various financial products, including savings accounts, loans, and mortgages."),
    ("How do I plan for retirement?", "I can guide you through the basics of retirement planning, including setting goals, understanding retirement accounts, and estimating the savings you'll need."),
    ("What's the difference between stocks and bonds?", "I can explain the key differences between stocks and bonds, including their risk profiles, returns, and roles in an investment portfolio."),
    ("How does inflation affect my savings?", "I can discuss how inflation impacts the purchasing power of your savings and suggest strategies to mitigate its effects."),
    ("Can you explain cryptocurrency?", "Certainly! I can provide an overview of cryptocurrencies, how they work, their potential benefits, and the risks involved."),
    ("What is diversification in investing?", "Diversification involves spreading your investments across various assets to reduce risk. I can explain how it works and why it's important."),
    ("How do I read a financial statement?", "I can guide you through the components of financial statements, such as balance sheets and income statements, to help you understand a company's financial health."),
    ("What are the tax implications of investing?", "I can provide general information on how different investments are taxed and what to consider when planning your investment strategy."),
    ("How can I save for my child's education?", "I can discuss various savings plans and investment options to help you prepare for future education expenses."),
    ("What is an emergency fund?", "An emergency fund is a savings buffer for unexpected expenses. I can advise on how much to save and strategies to build this fund."),
    ("How do credit scores work?", "I can explain what credit scores are, how they're calculated, and tips on how to improve and maintain a good credit score."),
    ("What is dollar-cost averaging?", "Dollar-cost averaging is an investment strategy where you invest fixed amounts at regular intervals, regardless of the market conditions. I can explain its benefits and considerations."),
    ("How can I protect my investments during a market downturn?", "I can provide strategies to help safeguard your investments during volatile market conditions, including diversification and asset allocation."),
    ("What are the benefits of a high-yield savings account?", "High-yield savings accounts offer higher interest rates compared to traditional savings accounts. I can discuss their advantages and what to look for when choosing one."),
    ("How does compound interest work?", "Compound interest is the interest on both the initial principal and the accumulated interest from previous periods. I can explain how it works and its impact on your savings and investments."),
    ("What should I consider before taking out a loan?", "I can outline key factors to consider, such as interest rates, repayment terms, and your financial situation, to help you make informed borrowing decisions."),
    ("How do I set financial goals?", "I can guide you through the process of setting realistic and achievable financial goals, whether they're short-term savings targets or long-term investment objectives."),
    ("What is a 401(k) plan?", "A 401(k) is a retirement savings plan offered by employers. I can explain how it works, its benefits, and considerations for maximizing your retirement savings."),
    ("How can I reduce my debt?", "I can provide strategies for managing and reducing debt, including budgeting tips, debt consolidation options, and prioritizing high-interest debts."),
    ("What are index funds?", "Index funds are investment funds that aim to replicate the performance of a specific market index. I can discuss their benefits, risks, and how they fit into an investment strategy."),
    ("How does the stock market work?", "I can explain the basics of how the stock market operates, including how stocks are traded, what influences stock prices, and the role of stock exchanges."),
    ("What is risk tolerance in investing?", "Risk tolerance refers to your ability and willingness to endure market volatility in your investment portfolio. I can help you assess your risk tolerance and its implications for your investment choices."),
    ("How do I start investing with a small amount of money?", "I can suggest investment options and strategies suitable for beginning investors with limited funds, emphasizing the importance of starting early and staying consistent."),
    ("What is a mutual fund?", "A mutual fund pools money from multiple investors to purchase a diversified portfolio of securities. I can explain how they work, their benefits, and potential drawbacks."),
    ("How can I track my expenses effectively?", "I can recommend tools and techniques for monitoring your spending habits, which is a crucial step in budgeting and achieving financial goals."),
    ("What are the advantages of online banking?", "Online banking offers convenience, accessibility, and often lower fees. I can discuss its benefits and tips for ensuring your online banking security."),
    ("How does leasing differ from buying a car?", "I can compare the financial implications of leasing versus purchasing a vehicle, including costs, benefits, and long-term considerations."),
    ("What is an emergency fund?", "An emergency fund is a savings buffer for unexpected expenses. I can advise on how much to save and strategies to build this fund."),
    ("How do credit scores work?", "I can explain what credit scores are, how they're calculated, and tips on how to improve and maintain a good credit score."),
    ("What is dollar-cost averaging?", "Dollar-cost averaging is an investment strategy where you invest fixed amounts at regular intervals, regardless of the market conditions. I can explain its benefits and considerations."),
    ("How can I protect my investments during a market downturn?", "I can provide strategies to help safeguard your investments during volatile market conditions, including diversification and asset allocation."),
    ("What are the benefits of a high-yield savings account?", "High-yield savings accounts offer higher interest rates compared to traditional savings accounts. I can discuss their advantages and what to look for when choosing one."),
    # Add more dynamic conversation starters
    ("I'm worried about the market volatility.", "I understand your concern about market volatility. Let's analyze your portfolio's risk exposure and discuss potential hedging strategies. What specific aspects of the market are you most concerned about?"),
    ("Can you explain technical analysis?", "Technical analysis involves studying price patterns and market trends using charts and indicators. Would you like me to explain some basic concepts like support/resistance levels, moving averages, or trend lines?"),
    ("I lost money in crypto.", "I'm sorry to hear about your cryptocurrency losses. Let's discuss what happened and explore risk management strategies for future investments. Would you like to start by reviewing the basic principles of crypto investing?"),
    ("What's causing the current inflation?", "Inflation is influenced by multiple factors including monetary policy, supply chain issues, and consumer demand. Let's analyze the current economic indicators and discuss how they might affect your financial planning."),
    ("Need advice on my startup's financials.", "I'd be happy to help with your startup's financial planning. Let's start by reviewing your current financial structure, cash flow projections, and funding needs. What's your most pressing concern?"),
    ("How can I improve my credit score?", "Improving your credit score involves managing debt, making timely payments, and monitoring your credit report. Let's discuss specific strategies to boost your creditworthiness. Are you looking to apply for a loan or mortgage soon?"),
    ("What's the best investment for retirement?", "Choosing the right retirement investment depends on your risk tolerance, time horizon, and financial goals. Let's explore options like 401(k) plans, IRAs, and diversified portfolios. What's your primary objective for retirement savings?"),
    ("I'm considering real estate investments.", "Real estate can be a lucrative investment, but it comes with risks and challenges. Let's discuss the key factors to consider, such as location, financing, and market trends. Are you looking to invest in rental properties or commercial real estate?"),
    ("How do I create a budget for my family?", "Creating a family budget involves tracking expenses, setting financial goals, and prioritizing spending. Let's work together to develop a customized budget plan that aligns with your family's needs and aspirations. What are your main financial priorities?"),
    ("I'm interested in sustainable investing.", "Sustainable investing focuses on environmental, social, and governance (ESG) factors. Let's explore ESG investment opportunities, impact metrics, and the potential benefits of aligning your investments with your values. What aspects of sustainability are most important to you?"),
    ("Can you explain the concept of risk management?", "Risk management involves identifying, assessing, and mitigating potential risks to protect your investments. Let's discuss risk management strategies, diversification techniques, and ways to safeguard your portfolio against market fluctuations. What level of risk are you comfortable with?"),
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
    ("Define a stock option.", "A stock option is a contract that gives the holder the right, but not the obligation, to buy or sell a specific number of shares of a stock at a predetermined price within a specified time frame. Stock options are used for investment, speculation, and employee compensation. There are two types of stock options: call options (buy) and put options (sell)."),
    ("What is a hedge fund?", "A hedge fund is an investment fund that pools capital from accredited investors and institutional investors to invest in a variety of assets and strategies. Hedge funds are managed by professional fund managers and aim to generate high returns while managing risk. They often use leverage, derivatives, and alternative investments to achieve their investment objectives."),
    ("Explain the concept of liquidity.", "Liquidity refers to the ease with which an asset can be bought or sold in the market without affecting its price. Liquid assets can be quickly converted into cash without significant price changes, while illiquid assets may take longer to sell and may incur a price discount. Liquidity is an important consideration for investors and financial institutions."),
    ("What is a capital gain?", "A capital gain is the profit realized from the sale of a capital asset, such as stocks, bonds, or real estate. It's calculated by subtracting the purchase price (cost basis) from the selling price. Capital gains can be short-term (held for one year or less) or long-term (held for more than one year) and are subject to capital gains tax."),
    ("Define a derivative.", "A derivative is a financial contract that derives its value from an underlying asset, index, or reference rate. Derivatives can be used for hedging, speculation, or arbitrage and include options, futures, forwards, and swaps. They allow investors to gain exposure to assets without owning them outright and can be highly leveraged."),
    ("What is a recession-proof industry?", "A recession-proof industry is a sector of the economy that remains stable or experiences growth during economic downturns. These industries provide essential goods or services that are in demand regardless of economic conditions. Examples of recession-proof industries include healthcare, utilities, consumer staples, and government services."),
    ("Explain the concept of time value of money.", "The time value of money is the principle that a dollar received today is worth more than a dollar received in the future due to its potential earning capacity. It's based on the premise that money can earn interest or be invested to generate returns over time. The time value of money is a fundamental concept in finance and investment analysis."),
    ("What is a credit default swap?", "A credit default swap (CDS) is a financial derivative that allows investors to hedge against the risk of default on a debt obligation, such as a bond or loan. The buyer of a CDS pays a premium to the seller in exchange for protection against credit events, such as bankruptcy or default. CDSs are used to manage credit risk and speculate on credit quality."),
    ("Define a commodity.", "A commodity is a raw material or primary agricultural product that can be bought and sold, such as gold, oil, wheat, or coffee. Commodities are standardized and interchangeable with other goods of the same type, allowing them to be traded on commodity exchanges. They are essential inputs in the production of goods and services and are subject to supply and demand dynamics."),
    ("What is a yield curve?", "A yield curve is a graphical representation of interest rates on bonds of different maturities. It shows the relationship between bond yields and time to maturity and is used to analyze economic conditions, inflation expectations, and monetary policy. The yield curve can be flat, upward-sloping (normal), or downward-sloping (inverted), each indicating different market expectations."),
    ("Explain the concept of cost of capital.", "The cost of capital is the rate of return required by investors to provide capital to a company. It represents the cost of financing a company's operations and investments and is used to evaluate the feasibility of projects and investments. The cost of capital is influenced by factors such as interest rates, risk, inflation, and market conditions."),
    ("What is a leveraged buyout?", "A leveraged buyout (LBO) is a financial transaction in which a company is acquired using a significant amount of borrowed funds or leverage. The acquiring company uses the target company's assets as collateral for the loan and aims to repay the debt with the target company's cash flow or by selling its assets. LBOs are often used to take public companies private."),
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
    ("Define a stock option.", "A stock option is a contract that gives the holder the right, but not the obligation, to buy or sell a specific number of shares of a stock at a predetermined price within a specified time frame. Stock options are used for investment, speculation, and employee compensation. There are two types of stock options: call options (buy) and put options (sell)."),
    ("What is a hedge fund?", "A hedge fund is an investment fund that pools capital from accredited investors and institutional investors to invest in a variety of assets and strategies. Hedge funds are managed by professional fund managers and aim to generate high returns while managing risk. They often use leverage, derivatives, and alternative investments to achieve their investment objectives."),
    ("Explain the concept of liquidity.", "Liquidity refers to the ease with which an asset can be bought or sold in the market without affecting its price. Liquid assets can be quickly converted into cash without significant price changes, while illiquid assets may take longer to sell and may incur a price discount. Liquidity is an important consideration for investors and financial institutions."),
    ("What is a capital gain?", "A capital gain is the profit realized from the sale of a capital asset, such as stocks, bonds, or real estate. It's calculated by subtracting the purchase price (cost basis) from the selling price. Capital gains can be short-term (held for one year or less) or long-term (held for more than one year) and are subject to capital gains tax."),
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
    ("Define a stock option.", "A stock option is a contract that gives the holder the right, but not the obligation, to buy or sell a specific number of shares of a stock at a predetermined price within a specified time frame. Stock options are used for investment, speculation, and employee compensation. There are two types of stock options: call options (buy) and put options (sell)."),
    ("What is a hedge fund?", "A hedge fund is an investment fund that pools capital from accredited investors and institutional investors to invest in a variety of assets and strategies. Hedge funds are managed by professional fund managers and aim to generate high returns while managing risk. They often use leverage, derivatives, and alternative investments to achieve their investment objectives."),
    ("Explain the concept of liquidity.", "Liquidity refers to the ease with which an asset can be bought or sold in the market without affecting its price. Liquid assets can be quickly converted into cash without significant price changes, while illiquid assets may take longer to sell and may incur a price discount. Liquidity is an important consideration for investors and financial institutions."),
    ("What is a capital gain?", "A capital gain is the profit realized from the sale of a capital asset, such as stocks, bonds, or real estate. It's calculated by subtracting the purchase price (cost basis) from the selling price. Capital gains can be short-term (held for one year or less) or long-term (held for more than one year) and are subject to capital gains tax."),
    ("Define a derivative.", "A derivative is a financial contract that derives its value from an underlying asset, index, or reference rate. Derivatives can be used for hedging, speculation, or arbitrage and include options, futures, forwards, and swaps. They allow investors to gain exposure to assets without owning them outright and can be highly leveraged."),
    ("What is a recession-proof industry?", "A recession-proof industry is a sector of the economy that remains stable or experiences growth during economic downturns. These industries provide essential goods or services that are in demand regardless of economic conditions. Examples of recession-proof industries include healthcare, utilities, consumer staples, and government services."),
    ("Explain the concept of time value of money.", "The time value of money is the principle that a dollar received today is worth more than a dollar received in the future due to its potential earning capacity. It's based on the premise that money can earn interest or be invested to generate returns over time. The time value of money is a fundamental concept in finance and investment analysis."),
    ("What is a credit default swap?", "A credit default swap (CDS) is a financial derivative that allows investors to hedge against the risk of default on a debt obligation, such as a bond or loan. The buyer of a CDS pays a premium to the seller in exchange for protection against credit events, such as bankruptcy or default. CDSs are used to manage credit risk and speculate on credit quality."),
    ("Define a commodity.", "A commodity is a raw material or primary agricultural product that can be bought and sold, such as gold, oil, wheat, or coffee. Commodities are standardized and interchangeable with other goods of the same type, allowing them to be traded on commodity exchanges. They are essential inputs in the production of goods and services and are subject to supply and demand dynamics."),
    ("What is a yield curve?", "A yield curve is a graphical representation of interest rates on bonds of different maturities. It shows the relationship between bond yields and time to maturity and is used to analyze economic conditions, inflation expectations, and monetary policy. The yield curve can be flat, upward-sloping (normal), or downward-sloping (inverted), each indicating different market expectations."),
    ("Explain the concept of cost of capital.", "The cost of capital is the rate of return required by investors to provide capital to a company. It represents the cost of financing a company's operations and investments and is used to evaluate the feasibility of projects and investments. The cost of capital is influenced by factors such as interest rates, risk, inflation, and market conditions."),
    ("What is a leveraged buyout?", "A leveraged buyout (LBO) is a financial transaction in which a company is acquired using a significant amount of borrowed funds or leverage. The acquiring company uses the target company's assets as collateral to secure the debt financing. LBOs are often used to take public companies private, restructure underperforming businesses, or finance mergers and acquisitions."),
    ("Define a capital structure.", "A capital structure is the mix of debt and equity financing used by a company to fund its operations and investments. It represents the proportion of debt, equity, and other securities in a company's capital stack. The capital structure affects a company's cost of capital, financial risk, and ability to raise funds. Companies aim to achieve an optimal capital structure that balances risk and return."),
    ("What is a corporate bond?", "A corporate bond is a debt security issued by a corporation to raise capital. It represents a loan from investors to the issuing company in exchange for periodic interest payments and the return of the principal at maturity. Corporate bonds are rated by credit agencies based on the issuer's creditworthiness and are traded in the bond market."),
    ("Explain the concept of working capital.", "Working capital is the difference between a company's current assets and current liabilities. It represents the funds available for day-to-day operations and is used to manage short-term financial obligations. Working capital is essential for maintaining liquidity, funding growth, and supporting business activities. It can be calculated as current assets minus current liabilities."),
    ("What is capital gains tax?", "Capital gains tax is a tax on the profit made from selling a capital asset, such as stocks, bonds, or real estate. It's calculated as the difference between the asset's sale price and purchase price and can be categorized as short-term or long-term based on the holding period."),
    ("What is ROI?", "Return on Investment (ROI) is a metric used to measure the profitability of an investment. It's calculated by dividing net profit by the cost of the investment and expressing the result as a percentage."),
    ("Explain market capitalization.", "Market capitalization (market cap) refers to the total value of a company's outstanding shares. It's calculated by multiplying the current stock price by the total number of outstanding shares."),
    ("What is a capital gain tax?", "A capital gains tax is a tax levied on the profit realized from the sale of a capital asset, such as stocks, bonds, or real estate. It's calculated based on the capital gain, which is the difference between the sale price and the purchase price of the asset. Capital gains can be subject to short-term or long-term tax rates, depending on the holding period of the asset."),
    ("What is ROI?", "ROI (Return on Investment) is a performance metric used to evaluate the efficiency of an investment. It's calculated by dividing the net profit by the cost of investment and expressing it as a percentage. For example, if you invest $1000 and earn $1200, your ROI is 20%."),
    ("Explain market capitalization.", "Market capitalization, or market cap, represents the total value of a company's shares in the market. It's calculated by multiplying the current share price by the total number of outstanding shares. Companies are often classified as large-cap (>$10B), mid-cap ($2-10B), or small-cap (<$2B)."),
    ("What is net worth?", "Net worth is the difference between total assets and total liabilities. It represents an individual’s or company's financial position at a given time. A positive net worth indicates more assets than liabilities, while a negative net worth means liabilities exceed assets."),
    ("Define liquidity.", "Liquidity refers to how quickly and easily an asset can be converted into cash without significantly affecting its market price. High liquidity assets, like stocks of large companies, can be sold rapidly, whereas real estate is considered less liquid."),
    ("What is a bear market?", "A bear market is a period during which stock prices fall by 20% or more from recent highs, often accompanied by widespread pessimism and negative investor sentiment."),
    ("Explain a bull market.", "A bull market is a period characterized by rising stock prices, typically by 20% or more, signaling investor confidence and expectations of strong future financial performance."),
    ("What is asset allocation?", "Asset allocation is an investment strategy that aims to balance risk and reward by distributing a portfolio's assets according to an individual's goals, risk tolerance, and investment horizon. The main asset classes are equities, fixed-income, and cash equivalents."),
    ("Define inflation.", "Inflation is the rate at which the general level of prices for goods and services rises, eroding purchasing power over time. Central banks attempt to limit inflation to maintain economic stability."),
    ("What is the time value of money?", "The time value of money is a financial concept that states a sum of money has greater value now than the same sum will have in the future due to its potential earning capacity. This principle underlies the concepts of interest rates, present value, and future value."),
    ("Explain compound interest.", "Compound interest is the interest on a loan or deposit calculated based on both the initial principal and the accumulated interest from previous periods. This effect causes wealth to grow faster over time."),
    ("What is a balance sheet?", "A balance sheet is a financial statement that provides a snapshot of a company's financial position at a specific point in time, detailing assets, liabilities, and shareholders' equity."),
    ("Define cash flow.", "Cash flow refers to the net amount of cash being transferred into and out of a business. Positive cash flow indicates that a company's liquid assets are increasing, enabling it to settle debts, reinvest, and return money to shareholders."),
    ("What is diversification?", "Diversification is an investment strategy that involves spreading investments across various financial instruments, industries, and other categories to reduce exposure to any single asset or risk."),
    ("Explain the price-to-earnings (P/E) ratio.", "The P/E ratio is a valuation metric for assessing a company's relative value. It's calculated by dividing the current market price of a stock by its earnings per share (EPS). A higher P/E suggests that investors expect higher earnings growth in the future."),
    ("What is EBITDA?", "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization. It's a measure of a company's overall financial performance and is used as an alternative to net income in some circumstances."),
    ("Define working capital.", "Working capital is the difference between a company's current assets and current liabilities. It measures a company's operational efficiency and short-term financial health."),
    ("What is the debt-to-equity ratio?", "The debt-to-equity ratio is a financial leverage ratio that compares a company's total liabilities to its shareholder equity. It's used to evaluate a company's financial leverage and risk."),
    ("Explain the concept of leverage.", "Leverage involves using borrowed capital or debt to increase the potential return of an investment. While it can amplify gains, it also increases the risk of losses."),
    ("What is a dividend?", "A dividend is a portion of a company's earnings distributed to shareholders. Dividends provide investors with a regular income stream and are typically paid quarterly."),
    ("Define capital gains.", "Capital gains are the profits realized from the sale of assets or investments, such as stocks, bonds, or real estate, when the selling price exceeds the purchase price."),
    ("What is an IPO?", "An IPO, or Initial Public Offering, is the process by which a private company offers shares to the public for the first time to raise capital. This transition allows the company to be publicly traded on stock exchanges."),
    ("Explain dollar-cost averaging.", "Dollar-cost averaging is an investment strategy where an investor divides the total amount to be invested across periodic purchases of a target asset, aiming to reduce the impact of volatility on the overall purchase."),
    ("What is a mutual fund?", "A mutual fund is an investment vehicle that pools money from multiple investors to purchase a diversified portfolio of stocks, bonds, or other securities, managed by professional fund managers."),
    ("Define expense ratio.", "The expense ratio represents the annual fee that mutual funds or exchange-traded funds charge their shareholders. It covers management, administrative, and operational expenses."),
    ("What is a stock split?", "A stock split occurs when a company increases the number of its outstanding shares to boost the stock's liquidity. Although the number of shares increases, the total dollar value of the shares remains the same, as the split doesn't add real value."),
    ("Explain the bid-ask spread.", "The bid-ask spread is the difference between the highest price a buyer is willing to pay for an asset and the lowest price a seller is willing to accept. It represents the transaction cost and liquidity of the asset."),
    ("What is a credit default swap (CDS)?", "A CDS is a financial derivative that allows an investor to 'swap' or offset their credit risk with that of another investor. For example, if a lender is concerned about a borrower defaulting, they can use a CDS to mitigate that risk."),
    ("Define quantitative easing.", "Quantitative easing is a monetary policy used by central banks to stimulate the economy by increasing money supply. It involves purchasing longer-term securities from the open market to inject liquidity and encourage lending and investment."),
    ("What is a hedge fund?", "A hedge fund is an alternative investment vehicle that employs various strategies to earn active returns for its investors. They may use leverage, derivatives, and other speculative techniques and are typically open to accredited investors."),
    ("Explain the efficient market hypothesis (EMH).", "The EMH suggests that financial markets are 'informationally efficient,' meaning that asset prices reflect all available information at any given time. Therefore, consistently achieving higher returns than the overall market is challenging without taking on additional risk."),
    ("What is a 401(k) plan?", "A 401(k) is a retirement savings plan sponsored by employers in the United States. It allows employees to save and invest a portion of their paycheck before taxes are taken."),
    ("What is a credit score?", "A credit score is a numerical representation of an individual's creditworthiness, ranging from 300 to 850. Lenders use it to assess the risk of lending money. A higher score indicates better credit history and lower risk."),
    ("Define amortization.", "Amortization is the process of spreading out a loan into a series of fixed payments over time. Each payment covers both interest and principal, reducing the loan balance gradually until it's paid off."),
    ("What is a fixed-rate mortgage?", "A fixed-rate mortgage is a home loan with an interest rate that remains constant throughout the loan's term, providing predictable monthly payments."),
    ("Explain an adjustable-rate mortgage (ARM).", "An ARM is a mortgage with an interest rate that can change periodically, typically in relation to an index. This means monthly payments can increase or decrease over time."),
    ("What is a FICO score?", "A FICO score is a type of credit score created by the Fair Isaac Corporation. It's widely used by lenders to evaluate credit risk."),
    ("Define annual percentage rate (APR).", "APR is the annual rate charged for borrowing or earned through an investment, expressed as a percentage. It includes fees or additional costs associated with the transaction."),
    ("What is a 529 plan?", "A 529 plan is a tax-advantaged savings plan designed to encourage saving for future education expenses. Contributions grow tax-free, and withdrawals for qualified education expenses are also tax-free."),
    ("Explain a Roth IRA.", "A Roth IRA is an individual retirement account allowing after-tax contributions. Qualified withdrawals during retirement are tax-free."),
    ("What is a traditional IRA?", "A traditional IRA is a retirement account where contributions may be tax-deductible, and investments grow tax-deferred until withdrawals during retirement."),
    ("Define capital expenditure (CapEx).", "CapEx refers to funds used by a company to acquire, upgrade, and maintain physical assets such as property, buildings, or equipment."),
    ("What is operating expenditure (OpEx)?", "OpEx are the ongoing costs for running a product, business, or system, including rent, utilities, and salaries."),
    ("Explain the difference between gross income and net income.", "Gross income is the total revenue earned before expenses, while net income is the profit remaining after all expenses, taxes, and costs have been deducted."),
    ("What is a mutual fund?", "A mutual fund is an investment vehicle pooling funds from multiple investors to purchase a diversified portfolio of securities managed by professionals."),
    ("Define exchange-traded fund (ETF).", "An ETF is a type of investment fund traded on stock exchanges, holding assets like stocks, commodities, or bonds, and generally operating with an arbitrage mechanism to keep trading close to its net asset value."),
    ("What is a hedge fund?", "A hedge fund is an alternative investment using pooled funds employing various strategies to earn active returns for investors, often with higher risk and less regulation."),
    ("Explain the concept of short selling.", "Short selling involves borrowing a security and selling it on the open market, planning to buy it back later at a lower price to profit from a decline in its value."),
    ("What is a margin account?", "A margin account allows investors to borrow money from a broker to purchase securities, using the account as collateral."),
    ("Define dividend yield.", "Dividend yield is a financial ratio indicating how much a company pays out in dividends each year relative to its stock price."),
    ("What is a stock buyback?", "A stock buyback occurs when a company purchases its own shares from the marketplace, reducing the number of outstanding shares."),
    ("Explain the concept of market liquidity.", "Market liquidity refers to the extent to which a market allows assets to be bought and sold at stable prices. High liquidity indicates assets can be quickly sold without affecting their price."),
    ("What is a bond?", "A bond is a fixed-income instrument representing a loan made by an investor to a borrower, typically corporate or governmental, with periodic interest payments and return of principal at maturity."),
    ("Define yield to maturity (YTM).", "YTM is the total return anticipated on a bond if held until it matures, considering all interest payments and the difference between its current price and par value."),
    ("What is a junk bond?", "A junk bond is a high-yield, high-risk security issued by companies with lower credit ratings, offering higher interest rates to attract investors."),
    ("Explain the concept of securitization.", "Securitization involves pooling various types of debt and selling them as consolidated financial instruments to investors, allowing for risk distribution."),
    ("What is a credit default swap (CDS)?", "A CDS is a financial derivative allowing an investor to 'swap' or offset their credit risk with that of another investor, functioning like insurance against default."),
    ("Define the term 'underwriting'.", "Underwriting is the process by which an individual or institution takes on financial risk for a fee, often in the context of loans, insurance, or investments."),
    ("What is an initial public offering (IPO)?", "An IPO is the process through which a private company offers shares to the public for the first time, transitioning to a publicly traded entity."),
    ("Explain the concept of dollar-cost averaging.", "Dollar-cost averaging is an investment strategy where an investor divides the total amount to be invested across periodic purchases of a target asset, aiming to reduce the impact of volatility."),
    ("What is a fiduciary?", "A fiduciary is an individual or organization legally obligated to act in the best interest of another party, such as a financial advisor managing a client's assets."),
    ("Explain the concept of market liquidity.", "Market liquidity refers to the extent to which a market allows assets to be bought and sold at stable prices. High liquidity indicates assets can be quickly sold without affecting their price."),
    ("What is a bond?", "A bond is a fixed-income instrument representing a loan made by an investor to a borrower, typically corporate or governmental, with periodic interest payments and return of principal at maturity."),
    ("Define yield to maturity (YTM).", "YTM is the total return anticipated on a bond if held until it matures, considering all interest payments and the difference between its current price and par value."),
    ("What is a junk bond?", "A junk bond is a high-yield, high-risk security issued by companies with lower credit ratings, offering higher interest rates to attract investors."),
    ("Explain the concept of securitization.", "Securitization involves pooling various types of debt and selling them as consolidated financial instruments to investors, allowing for risk distribution."),
    ("What is a credit default swap (CDS)?", "A CDS is a financial derivative allowing an investor to 'swap' or offset their credit risk with that of another investor, functioning like insurance against default."),
    ("Define the term 'underwriting'.", "Underwriting is the process by which an individual or institution takes on financial risk for a fee, often in the context of loans, insurance, or investments."),
    ("What is an initial public offering (IPO)?", "An IPO is the process through which a private company offers shares to the public for the first time, transitioning to a publicly traded entity."),
    ("Explain the concept of dollar-cost averaging.", "Dollar-cost averaging is an investment strategy where an investor divides the total amount to be invested across periodic purchases of a target asset, aiming to reduce the impact of volatility."),
    ("What is a fiduciary?", "A fiduciary is an individual or organization legally obligated to act in the best interest of another party, such as a financial advisor managing a client's assets."),
    ("Define the term 'escrow'.", "Escrow is a financial arrangement where a third party holds and regulates payment of funds required for two parties involved in a transaction, ensuring security until all conditions are met."),
    ("What is a leveraged buyout (LBO)?", "An LBO is the acquisition of a company using a significant amount of borrowed money to meet the cost of acquisition, with the assets of the company often used as collateral."),
    ("Explain the concept of quantitative easing (QE).", "QE is a monetary policy where a central bank purchases government securities or other securities to increase the money supply and encourage lending and investment."),
    ("What is a special purpose vehicle (SPV)?", "An SPV is a subsidiary created by a parent company to isolate financial risk, having its own legal status and assets/liabilities."),
    ("Define the term 'arbitrage'.", "Arbitrage involves the simultaneous purchase and sale of an asset in different markets to profit from price discrepancies."),
    ("What is a Ponzi scheme?", "A Ponzi scheme is a fraudulent investing scam promising high returns with little risk, generating returns for earlier investors through revenue paid by new investors."),
    ("Explain the concept of the time value of money.", "The time value of money is the idea that a sum of money has greater value now than the same sum in the future due to its potential earning capacity."),
    ("What is a zero-coupon bond?", "A zero-coupon bond is a debt security that doesn't pay interest but is traded at a deep discount, rendering profit at maturity when redeemed for its full face value."),
    ("Define the term 'liquidity ratio'.", "A liquidity ratio measures a company's ability to pay off short-term obligations with its liquid assets, indicating financial health and stability."),
    ("What is a credit rating?", "A credit rating is an evaluation of the creditworthiness of an individual or entity, assessing the likelihood of default on financial obligations."),
    ("Explain the concept of a stock option.", "A stock option is a contract giving the holder the right to buy or sell a specific number of shares at a predetermined price within a set time frame."),
    ("What is a dividend reinvestment plan (DRIP)?", "A DRIP is an investment strategy allowing shareholders to reinvest their dividends into additional shares of the company's stock."),
    ("Define the term 'capital gains tax'.", "Capital gains tax is a levy on the profit realized from the sale of an asset, such as stocks or real estate, calculated based on the capital gain."),
    ("What is a stock exchange?", "A stock exchange is a marketplace where securities, such as stocks and bonds, are bought and sold, providing liquidity and transparency to the financial markets."),
    ("Explain the concept of a bear market.", "A bear market is a period of declining stock prices, typically by 20% or more, often accompanied by widespread pessimism and negative investor sentiment."),
    ("What is a bull market?", "A bull market is a period of rising stock prices, typically by 20% or more, signaling investor confidence and expectations of strong future financial performance."),
    ("Define asset allocation.", "Asset allocation is an investment strategy balancing risk and reward by distributing a portfolio's assets according to an individual's goals, risk tolerance, and investment horizon."),
    # Add more specialized financial QA pairs
    ("What are ESG investments?", "ESG (Environmental, Social, and Governance) investments focus on companies meeting specific sustainability and ethical standards. These investments consider factors like carbon footprint, workplace diversity, and corporate transparency alongside financial returns."),
    ("Explain DeFi lending.", "Decentralized Finance (DeFi) lending allows users to lend and borrow cryptocurrencies directly through smart contracts, eliminating traditional intermediaries. Users can earn interest by providing liquidity to lending pools or borrow assets by providing collateral."),
    ("What is a SPAC?", "A Special Purpose Acquisition Company (SPAC) is a shell corporation listed on a stock exchange with the purpose of acquiring an existing private company, thereby making it public without going through the traditional IPO process."),
    ("How do circuit breakers work?", "Circuit breakers are automatic trading halts triggered when market indices drop by certain percentages. For example, the S&P 500 has three circuit breaker thresholds: 7% (Level 1), 13% (Level 2), and 20% (Level 3), designed to prevent panic selling."),
    ("What is tax-loss harvesting?", "Tax-loss harvesting is an investment strategy where you sell securities at a loss to offset capital gains tax liability. This involves carefully timing the sale of investments while adhering to wash-sale rules and other tax regulations."),
    ("Define the term 'escrow'.", "Escrow is a financial arrangement where a third party holds and regulates payment of funds required for two parties involved in a transaction, ensuring security until all conditions are met."),
    ("What is a leveraged buyout (LBO)?", "An LBO is the acquisition of a company using a significant amount of borrowed money to meet the cost of acquisition, with the assets of the company often used as collateral."),
    ("Explain the concept of quantitative easing (QE).", "QE is a monetary policy where a central bank purchases government securities or other securities to increase the money supply and encourage lending and investment."),
    ("What is a special purpose vehicle (SPV)?", "An SPV is a subsidiary created by a parent company to isolate financial risk, having its own legal status and assets/liabilities."),
    ("Define the term 'arbitrage'.", "Arbitrage involves the simultaneous purchase and sale of an asset in different markets to profit from price discrepancies."),
    ("What is a Ponzi scheme?", "A Ponzi scheme is a fraudulent investing scam promising high returns with little risk, generating returns for earlier investors through revenue paid by new investors."),
    ("Explain the concept of the time value of money.", "The time value of money is the idea that a sum of money has greater value now than the same sum in the future due to its potential earning capacity."),
    ("What is a zero-coupon bond?", "A zero-coupon bond is a debt security that doesn't pay interest but is traded at a deep discount, rendering profit at maturity when redeemed for its full face value."),
    ("Define the term 'liquidity ratio'.", "A liquidity ratio measures a company's ability to pay off short-term obligations with its liquid assets, indicating financial health and stability."),
    ("What is a credit rating?", "A credit rating is an evaluation of the creditworthiness of an individual or entity, assessing the likelihood of default on financial obligations."),
    ("Explain the concept of a stock option.", "A stock option is a contract giving the holder the right to buy or sell a specific number of shares at a predetermined price within a set time frame."),
    ("What is a dividend reinvestment plan (DRIP)?", "A DRIP is an investment strategy allowing shareholders to reinvest their dividends into additional shares of the company's stock."),
    ("Define the term 'capital gains tax'.", "Capital gains tax is a levy on the profit realized from the sale of an asset, such as stocks or real estate, calculated based on the capital gain."),
    ("What is a stock exchange?", "A stock exchange is a marketplace where securities, such as stocks and bonds, are bought and sold, providing liquidity and transparency to the financial markets."),
    ("Explain the concept of a bear market.", "A bear market is a period of declining stock prices, typically by 20% or more, often accompanied by widespread pessimism and negative investor sentiment."),
    ("What is a bull market?", "A bull market is a period of rising stock prices, typically by 20% or more, signaling investor confidence and expectations of strong future financial performance."),
    ("Define asset allocation.", "Asset allocation is an investment strategy balancing risk and reward by distributing a portfolio's assets according to an individual's goals, risk tolerance, and investment horizon."),
    ("What are ESG investments?", "ESG (Environmental, Social, and Governance) investments focus on companies meeting specific sustainability and ethical standards. These investments consider factors like carbon footprint, workplace diversity, and corporate transparency alongside financial returns."),
    ("Explain DeFi lending.", "Decentralized Finance (DeFi) lending allows users to lend and borrow cryptocurrencies directly through smart contracts, eliminating traditional intermediaries. Users can earn interest by providing liquidity to lending pools or borrow assets by providing collateral."),
    ("What is a SPAC?", "A Special Purpose Acquisition Company (SPAC) is a shell corporation listed on a stock exchange with the purpose of acquiring an existing private company, thereby making it public without going through the traditional IPO process."),
    ("How do circuit breakers work?", "Circuit breakers are automatic trading halts triggered when market indices drop by certain percentages. For example, the S&P 500 has three circuit breaker thresholds: 7% (Level 1), 13% (Level 2), and 20% (Level 3), designed to prevent panic selling."),
    ("What is tax-loss harvesting?", "Tax-loss harvesting is an investment strategy where you sell securities at a loss to offset capital gains tax liability. This involves carefully timing the sale of investments while adhering to wash-sale rules and other tax regulations."),
    ("Explain the concept of a reverse mortgage.", "A reverse mortgage is a loan available to homeowners aged 62 or older that allows them to convert part of their home equity into cash. The loan is repaid when the borrower moves out, sells the home, or passes away."),
    ("What is a 529 plan?", "A 529 plan is a tax-advantaged savings plan designed to encourage saving for future education expenses. These plans offer various investment options and tax benefits, such as tax-free growth and withdrawals for qualified education expenses."),
    ("What is market liquidity?", "Market liquidity measures how easily an asset can be bought or sold without causing significant price changes."),
    ("Define a bond.", "A bond is a debt security where the issuer owes the holder a debt and pays periodic interest along with the principal at maturity."),
    ("What does yield to maturity (YTM) mean?", "YTM represents the total expected return on a bond if held until its maturity date, including interest and price difference."),
    ("What are junk bonds?", "Junk bonds are high-risk, high-yield securities issued by companies with low credit ratings, offering higher returns."),
    ("Explain securitization.", "Securitization is the process of pooling financial assets, such as loans, and selling them as tradable securities."),
    ("What is a credit default swap?", "A credit default swap is a derivative contract allowing risk transfer of credit default between two parties."),
    ("What is underwriting in finance?", "Underwriting involves assessing financial risk and guaranteeing the sale of securities or insurance for a fee."),
    ("What happens during an IPO?", "An IPO is when a private company sells shares to the public for the first time to raise capital."),
    ("How does dollar-cost averaging work?", "Dollar-cost averaging involves investing a fixed amount regularly, regardless of market conditions, to reduce risk."),
    ("What is a fiduciary duty?", "Fiduciary duty is the legal obligation to act in the best interest of another party, especially in financial matters."),
    ("What is escrow?", "Escrow is a financial arrangement where a neutral third party holds funds until contract conditions are met."),
    ("Define leveraged buyout (LBO).", "An LBO is when a company is acquired using borrowed funds, with the target's assets often serving as collateral."),
    ("What is quantitative easing (QE)?", "QE is when a central bank buys financial assets to inject money into the economy and stimulate growth."),
    ("What does SPV stand for?", "A Special Purpose Vehicle (SPV) is a separate entity created to isolate financial risk."),
    ("What is arbitrage?", "Arbitrage is exploiting price differences of an asset across markets to make a profit."),
    ("How does a Ponzi scheme work?", "A Ponzi scheme uses money from new investors to pay returns to earlier investors, creating an illusion of profit."),
    ("What is the time value of money?", "The time value of money states that money available today is worth more than the same sum in the future."),
    ("What are zero-coupon bonds?", "Zero-coupon bonds are sold at a discount and pay no interest but return face value at maturity."),
    ("What is a liquidity ratio?", "A liquidity ratio measures a company's ability to meet short-term financial obligations."),
    ("What are credit ratings?", "Credit ratings evaluate the likelihood of a borrower defaulting on debt obligations."),
    ("How do stock options work?", "Stock options give the right to buy or sell shares at a set price before an expiration date."),
    ("What is a DRIP?", "A Dividend Reinvestment Plan (DRIP) allows investors to reinvest their dividends into additional shares."),
    ("What is capital gains tax?", "Capital gains tax is levied on the profit from selling assets like stocks or property."),
    ("What happens on a stock exchange?", "A stock exchange facilitates buying and selling of securities, ensuring liquidity and transparency."),
    ("What defines a bear market?", "A bear market is characterized by a prolonged decline in stock prices, typically over 20%."),
    ("What defines a bull market?", "A bull market is marked by rising stock prices and positive investor sentiment."),
    ("Explain asset allocation.", "Asset allocation distributes investments across asset classes to balance risk and reward."),
    ("What are ESG criteria?", "ESG (Environmental, Social, Governance) criteria evaluate a company's ethical and sustainability practices."),
    ("What is DeFi lending?", "DeFi lending uses smart contracts to enable direct lending and borrowing of cryptocurrencies."),
    ("How do SPACs work?", "A SPAC is a shell company created to acquire a private company and take it public."),
    ("What triggers circuit breakers in trading?", "Circuit breakers halt trading temporarily when stock indices fall by predefined percentages."),
    ("How does tax-loss harvesting work?", "Tax-loss harvesting involves selling assets at a loss to offset taxable capital gains."),
    ("What is compounding interest?", "Compounding interest earns interest on both the initial principal and previously earned interest."),
    ("Define a mutual fund.", "A mutual fund pools money from multiple investors to purchase a diversified portfolio of assets."),
    ("What is inflation?", "Inflation is the rate at which the general price level of goods and services rises."),
    ("What is deflation?", "Deflation refers to a decrease in the general price level of goods and services."),
    ("What is a credit score?", "A credit score indicates an individual's creditworthiness based on past financial behavior."),
    ("Explain diversification.", "Diversification spreads investments across different assets to reduce risk."),
    ("What is an ETF?", "An Exchange-Traded Fund (ETF) is a fund traded on stock exchanges, holding a portfolio of assets."),
    ("What are derivatives?", "Derivatives are financial contracts whose value depends on underlying assets like stocks or commodities."),
    ("What is short selling?", "Short selling is borrowing and selling assets with the intention of repurchasing them at a lower price."),
    ("What are blue-chip stocks?", "Blue-chip stocks belong to well-established, financially sound companies with a history of reliable performance."),
    ("Explain financial leverage.", "Financial leverage involves using borrowed money to increase potential returns on investments."),
    ("What is the purpose of a central bank?", "Central banks regulate monetary policy, control inflation, and ensure financial stability."),
    ("What are savings bonds?", "Savings bonds are low-risk, government-issued bonds designed for long-term savings."),
    ("What is cash flow?", "Cash flow represents the net amount of cash moving in and out of a business."),
    ("What is a balance sheet?", "A balance sheet shows a company's assets, liabilities, and equity at a specific point in time."),
    ("What is ROI?", "Return on Investment (ROI) measures the profitability of an investment relative to its cost."),
    ("Explain opportunity cost.", "Opportunity cost is the potential gain lost when choosing one option over another."),
    ("What is an emergency fund?", "An emergency fund is a savings buffer for unexpected expenses or financial hardships."),
    ("What is a budget?", "A budget is a financial plan outlining expected income and expenses over a period."),
    ("What is a mortgage?", "A mortgage is a loan used to purchase real estate, typically repaid in installments."),
    ("What are index funds?", "Index funds are mutual funds or ETFs designed to track a specific market index."),
    ("Define risk tolerance.", "Risk tolerance is an investor's ability to endure losses in their investment portfolio."),
    ("What is debt-to-equity ratio?", "Debt-to-equity ratio measures a company's financial leverage by comparing total debt to shareholders' equity."),
    ("What is equity?", "Equity represents ownership value in an asset after deducting liabilities."),
    ("What is a stock split?", "A stock split increases the number of shares outstanding while reducing the price per share."),
    ("What is a dividend yield?", "Dividend yield is the annual dividend income per share divided by the stock price."),
    ("What is a capital gain?", "A capital gain is the profit from selling an asset for more than its purchase price."),
    ("What is a capital loss?", "A capital loss occurs when an asset is sold for less than its purchase price."),
    ("What is a 401(k) plan?", "A 401(k) is an employer-sponsored retirement savings plan allowing employees to contribute a portion of their salary."),
    ("What is a Roth IRA?", "A Roth IRA is an individual retirement account where contributions are made after tax, and qualified withdrawals are tax-free."),
    ("What is a traditional IRA?", "A traditional IRA is a retirement account where contributions may be tax-deductible, and investments grow tax-deferred."),
    ("What is a credit score?", "A credit score is a numerical representation of an individual's creditworthiness."),
    ("What is a credit report?", "A credit report is a detailed record of an individual's credit history."),
    ("What is a credit freeze?", "A credit freeze restricts access to an individual's credit report, preventing new accounts from being opened."),
    ("What is a credit utilization ratio?", "Credit utilization ratio is the percentage of available credit being used."),
    ("What is a credit inquiry?", "A credit inquiry is a request to view an individual's credit report."),
    ("What is a credit limit?", "A credit limit is the maximum amount of credit extended to an individual."),
    ("What is a credit card?", "A credit card is a payment card issued to users to enable purchases on credit."),
    ("What is a debit card?", "A debit card is a payment card linked to a checking account, deducting funds directly from the account."),
    ("What is a credit union?", "A credit union is a member-owned financial cooperative offering banking services."),
    ("What is a bank?", "A bank is a financial institution that accepts deposits and provides loans."),
    ("What is a savings account?", "A savings account is a deposit account held at a financial institution, earning interest."),
    ("What is a checking account?", "A checking account is a deposit account allowing withdrawals and deposits."),
    ("What is a money market account?", "A money market account is a type of savings account offering higher interest rates."),
    ("What is a certificate of deposit (CD)?", "A CD is a time deposit with a fixed term and interest rate, typically higher than regular savings accounts."),
    ("What is compound interest?", "Compound interest is interest calculated on the initial principal and accumulated interest."),
    ("What is simple interest?", "Simple interest is interest calculated only on the principal amount."),
    ("What is a mortgage?", "A mortgage is a loan used to purchase real estate, with the property serving as collateral."),
    ("What is a down payment?", "A down payment is an initial payment made when purchasing an expensive item, often a percentage of the total cost."),
    ("What is a credit score?", "A credit score is a numerical representation of an individual's creditworthiness."),   

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
            "question": "What is diversification in investing?",
            "answer": "Diversification is an investment strategy where an investor spreads investments across various assets to reduce risk. It ensures that poor performance in one asset doesn’t drastically impact the overall portfolio."
        },
        {
            "question": "What are financial derivatives?",
            "answer": "Financial derivatives are contracts whose value is derived from an underlying asset, index, or rate. Examples include options, futures, and swaps, used for hedging risk or speculation."
        },
        {
            "question": "What is the time value of money?",
            "answer": "The time value of money is the concept that a specific amount of money is worth more today than the same amount in the future, due to its earning potential."
        },
        {
            "question": "What is a financial bubble?",
            "answer": "A financial bubble occurs when the price of an asset significantly exceeds its intrinsic value, driven by excessive speculation and investor enthusiasm, often followed by a sharp market correction."
        },
        {
            "question": "What is portfolio rebalancing?",
            "answer": "Portfolio rebalancing involves adjusting the proportions of assets in an investment portfolio to maintain the desired level of risk and return, typically after market fluctuations."
        },
        {
            "question": "What is a stop-loss order?",
            "answer": "A stop-loss order is an order placed with a broker to sell a security when it reaches a specific price, protecting investors from significant losses."
        },
        {
            "question": "What is the role of the Federal Reserve in the U.S. economy?",
            "answer": "The Federal Reserve regulates monetary policy, controls interest rates, and ensures financial system stability to promote maximum employment and stable prices."
        },
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
            "question": "What is diversification in investing?",
            "answer": "Diversification is an investment strategy where an investor spreads investments across various assets to reduce risk. It ensures that poor performance in one asset doesn’t drastically impact the overall portfolio."
        },
        {
            "question": "What are financial derivatives?",
            "answer": "Financial derivatives are contracts whose value is derived from an underlying asset, index, or rate. Examples include options, futures, and swaps, used for hedging risk or speculation."
        },
        {
            "question": "What is the time value of money?",
            "answer": "The time value of money is the concept that a specific amount of money is worth more today than the same amount in the future, due to its earning potential."
        },
        {
            "question": "What is a financial bubble?",
            "answer": "A financial bubble occurs when the price of an asset significantly exceeds its intrinsic value, driven by excessive speculation and investor enthusiasm, often followed by a sharp market correction."
        },
        {
            "question": "What is portfolio rebalancing?",
            "answer": "Portfolio rebalancing involves adjusting the proportions of assets in an investment portfolio to maintain the desired level of risk and return, typically after market fluctuations."
        },
        {
            "question": "What is a stop-loss order?",
            "answer": "A stop-loss order is an order placed with a broker to sell a security when it reaches a specific price, protecting investors from significant losses."
        },
        {
            "question": "What is the role of the Federal Reserve in the U.S. economy?",
            "answer": "The Federal Reserve regulates monetary policy, controls interest rates, and ensures financial system stability to promote maximum employment and stable prices."
        },
        {
            "question": "What is a 401(k) plan?",
            "answer": "A 401(k) is a retirement savings plan offered by employers, allowing employees to contribute a portion of their salary on a pre-tax basis, often with employer-matching contributions."
        },
        {
            "question": "What are credit default swaps (CDS)?",
            "answer": "Credit default swaps are financial derivatives that act as insurance against the default of a borrower. Investors pay a premium for protection in case of borrower default."
        },
        {
            "question": "What is a hedge fund?",
            "answer": "A hedge fund is a pooled investment fund that employs advanced strategies, such as leveraging, short-selling, and derivatives, to maximize returns for accredited investors."
        },
        {
            "question": "What is the difference between revenue and profit?",
            "answer": "Revenue is the total income a company earns from sales or services, while profit is the income left after subtracting expenses, taxes, and costs from revenue."
        },
        {
            "question": "What is an IPO (Initial Public Offering)?",
            "answer": "An IPO is the process through which a private company offers its shares to the public for the first time, raising capital and becoming publicly traded."
        },
        {
            "question": "What are the risks of investing in cryptocurrency?",
            "answer": "Cryptocurrency investments carry risks such as market volatility, regulatory uncertainty, cybersecurity threats, and lack of investor protection."
        },
        {
            "question": "What is asset allocation?",
            "answer": "Asset allocation is the strategy of dividing an investment portfolio across different asset classes, such as stocks, bonds, and real estate, to balance risk and reward."
        },
        {
            "question": "What is EBITDA?",
            "answer": "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization. It’s a measure of a company's operational profitability."
        },
        {
            "question": "What is a stock split?",
            "answer": "A stock split occurs when a company divides its existing shares into multiple shares, reducing the price per share while maintaining the overall market capitalization."
        },
        {
            "question": "What is a balance sheet?",
            "answer": "A balance sheet is a financial statement that shows a company's assets, liabilities, and equity at a specific point in time. It provides insights into the financial health and capital structure of the business."
        },
        {
            "question": "What are the different types of financial ratios?",
            "answer": "Key financial ratios include liquidity ratios (e.g., current ratio), profitability ratios (e.g., net profit margin), leverage ratios (e.g., debt-to-equity), and efficiency ratios (e.g., inventory turnover)."
        },
        {
            "question": "What is quantitative easing (QE)?",
            "answer": "Quantitative easing is a monetary policy tool used by central banks to inject liquidity into the economy by purchasing government bonds or other securities to stimulate economic activity."
        },
        {
            "question": "What is inflation?",
            "answer": "Inflation is the rate at which the general price level of goods and services rises, reducing the purchasing power of money over time."
        },
        {
            "question": "What is the difference between fiscal policy and monetary policy?",
            "answer": "Fiscal policy involves government spending and taxation to influence the economy, while monetary policy uses tools like interest rates and money supply management, typically implemented by a central bank."
        },
        {
            "question": "What is credit risk?",
            "answer": "Credit risk refers to the potential for a borrower to default on a loan or fail to meet contractual obligations, leading to financial losses for the lender."
        },
        {
            "question": "What is the difference between a stock and a bond?",
            "answer": "A stock represents ownership in a company, while a bond represents a loan made to a company or government, with periodic interest payments and repayment of the principal at maturity."
        },
        {
            "question": "What is a short sale in stock trading?",
            "answer": "A short sale involves selling borrowed shares with the intention of buying them back later at a lower price, profiting from a decline in the stock's value."
        },
        {
            "question": "What are mutual funds?",
            "answer": "Mutual funds pool money from multiple investors to invest in diversified portfolios of stocks, bonds, or other securities, managed by professional fund managers."
        },
        {
            "question": "What is a sovereign wealth fund?",
            "answer": "A sovereign wealth fund is a state-owned investment fund that manages national savings and surplus revenues, often derived from natural resources or trade surpluses."
        },
        {
            "question": "What is diversification risk?",
            "answer": "Diversification risk arises when an investment portfolio is either too concentrated in one asset or overly spread across too many assets, reducing potential returns."
        },
        {
            "question": "What is a margin call?",
            "answer": "A margin call occurs when the value of a margin account falls below the broker's required minimum, requiring the investor to deposit additional funds or sell assets to cover losses."
        },
        {
            "question": "What is the efficient market hypothesis (EMH)?",
            "answer": "The efficient market hypothesis suggests that asset prices fully reflect all available information, making it impossible to consistently outperform the market through stock-picking or market timing."
        },
        {
            "question": "What is a yield curve?",
            "answer": "A yield curve is a graph that shows the yields of bonds with different maturities. An upward-sloping curve indicates economic growth, while an inverted curve may signal a recession."
        },
        {
            "question": "What is venture capital?",
            "answer": "Venture capital is a form of private equity financing provided to startups and small businesses with high growth potential in exchange for equity."
        },
        {
            "question": "What is financial leverage?",
            "answer": "Financial leverage refers to the use of borrowed capital to increase potential returns on an investment. However, it also increases financial risk."
        },
        {
            "question": "What is liquidity risk?",
            "answer": "Liquidity risk is the risk that an asset cannot be quickly sold or converted into cash without significantly affecting its price."
        },
        {
            "question": "What is the difference between active and passive investing?",
            "answer": "Active investing involves actively managing a portfolio to outperform the market, while passive investing aims to match market returns using index funds or ETFs."
        },
        {
            "question": "What is a credit score?",
            "answer": "A credit score is a numerical representation of an individual's creditworthiness, based on their credit history, payment behavior, and outstanding debts."
        },
        {
            "question": "What is risk-adjusted return?",
            "answer": "Risk-adjusted return measures the return on an investment relative to its risk, often calculated using metrics like the Sharpe ratio or Treynor ratio."
        },
        {
            "question": "What is a recession?",
            "answer": "A recession is a period of significant economic decline, typically defined as two consecutive quarters of negative GDP growth."
        },
        {
            "question": "What are commodities in finance?",
            "answer": "Commodities are basic goods used in commerce, such as oil, gold, or agricultural products, which can be traded on commodity exchanges."
        },
        {
            "question": "What is private equity?",
            "answer": "Private equity refers to investments made directly into private companies or buyouts of public companies, with the goal of improving performance and increasing value."
        },
        {
            "question": "What is a financial audit?",
            "answer": "A financial audit is an independent examination of a company's financial statements to ensure accuracy, compliance with accounting standards, and transparency."
        },
        {
            "question": "What is a callable bond?",
            "answer": "A callable bond is a bond that the issuer can redeem before its maturity date, usually at a specified call price."
        },
        {
            "question": "What is alpha in investing?",
            "answer": "Alpha measures an investment's performance relative to a benchmark index, representing the excess return generated by active management."
        },
        {
            "question": "What is beta in finance?",
            "answer": "Beta measures an asset's volatility relative to the overall market. A beta greater than 1 indicates higher volatility, while less than 1 indicates lower volatility."
        },
        {
            "question": "What are treasury bonds?",
            "answer": "Treasury bonds are long-term debt securities issued by the government to finance public spending, considered low-risk investments."
        },
        {
            "question": "What is a financial covenant?",
            "answer": "A financial covenant is a condition set by lenders on borrowers, requiring them to maintain certain financial ratios or meet specific financial benchmarks."
        },
        # Contemporary Financial Topics (DeFi, NFTs, etc.)
        {
            "question": "What is Decentralized Finance (DeFi)?",
            "answer": "Decentralized Finance (DeFi) refers to a blockchain-based financial system that removes intermediaries like banks and brokers, allowing users to access financial services such as lending, borrowing, and trading directly through smart contracts."
        },
        {
            "question": "What are Non-Fungible Tokens (NFTs)?",
            "answer": "Non-Fungible Tokens (NFTs) are unique digital assets verified on a blockchain that represent ownership of digital or physical items, such as art, music, or real estate."
        },
        {
            "question": "How do stablecoins maintain their value?",
            "answer": "Stablecoins are cryptocurrencies pegged to stable assets like the US dollar or gold. They maintain value through collateral reserves, algorithms, or a combination of mechanisms to prevent volatility."
        },
        {
            "question": "What is yield farming in cryptocurrency?",
            "answer": "Yield farming involves lending or staking cryptocurrency assets in DeFi protocols to earn interest or rewards, often through liquidity pools."
        },
        {
            "question": "What is a DAO in finance?",
            "answer": "A Decentralized Autonomous Organization (DAO) is an organization governed by smart contracts and voting mechanisms on a blockchain, without centralized authority."
        },
        {
            "question": "What are Initial Coin Offerings (ICOs) and Initial DEX Offerings (IDOs)?",
            "answer": "ICOs are fundraising methods where tokens are sold before a project launch. IDOs occur on decentralized exchanges (DEXs) and offer immediate liquidity and decentralized listing."
        },
        {
            "question": "What are the risks associated with DeFi protocols?",
            "answer": "Key risks include smart contract vulnerabilities, lack of regulation, impermanent loss in liquidity pools, and susceptibility to hacks."
        },
        {
            "question": "What is tokenomics?",
            "answer": "Tokenomics refers to the economic structure of a cryptocurrency, including token supply, distribution, incentives, and governance mechanisms."
        },
        {
            "question": "What is crypto staking?",
            "answer": "Staking involves holding cryptocurrency in a wallet to support network operations like validating transactions, in exchange for rewards."
        },
        {
            "question": "What is a rug pull in crypto markets?",
            "answer": "A rug pull is a scam where developers of a cryptocurrency project abandon it after collecting investor funds, leaving tokens worthless."
        },

        # Market Analysis and Trading Strategies
        {
            "question": "What is technical analysis in trading?",
            "answer": "Technical analysis involves evaluating securities by analyzing statistical trends from trading activity, such as price movements and trading volumes."
        },
        {
            "question": "What is fundamental analysis?",
            "answer": "Fundamental analysis assesses a company's value based on financial statements, industry trends, and macroeconomic factors to determine if a stock is undervalued or overvalued."
        },
        {
            "question": "What is algorithmic trading?",
            "answer": "Algorithmic trading uses computer algorithms to execute trades based on predefined criteria, such as price, volume, and timing."
        },
        {
            "question": "What is a stop-loss order?",
            "answer": "A stop-loss order is an automated order to sell a security when it reaches a specific price, limiting potential losses."
        },
        {
            "question": "What is dollar-cost averaging (DCA)?",
            "answer": "Dollar-cost averaging is an investment strategy where an investor consistently invests a fixed amount in a security at regular intervals, regardless of market conditions."
        },
        {
            "question": "What is the difference between day trading and swing trading?",
            "answer": "Day trading involves buying and selling securities within the same day, while swing trading holds positions for several days or weeks to capture larger price movements."
        },
        {
            "question": "What is market liquidity?",
            "answer": "Market liquidity refers to the ease with which an asset can be bought or sold without significantly affecting its price."
        },
        {
            "question": "What is portfolio rebalancing?",
            "answer": "Portfolio rebalancing involves adjusting the allocation of assets in a portfolio to maintain a desired risk-return profile."
        },
        {
            "question": "What are contrarian investment strategies?",
            "answer": "Contrarian strategies involve going against prevailing market trends, such as buying undervalued assets when most investors are selling."
        },
        {
            "question": "What is the Sharpe ratio?",
            "answer": "The Sharpe ratio measures the risk-adjusted return of an investment by comparing its excess return over the risk-free rate to its volatility."
        },

        # Risk Management and Portfolio Theory
        {
            "question": "What is systematic risk?",
            "answer": "Systematic risk refers to market-wide risks that cannot be eliminated through diversification, such as economic downturns or political instability."
        },
        {
            "question": "What is unsystematic risk?",
            "answer": "Unsystematic risk is specific to a company or industry and can be reduced through portfolio diversification."
        },
        {
            "question": "What is Value at Risk (VaR)?",
            "answer": "Value at Risk (VaR) estimates the maximum potential loss of an investment portfolio over a given time period, under normal market conditions."
        },
        {
            "question": "What is hedging?",
            "answer": "Hedging is a strategy used to reduce financial risk by taking an offsetting position in a related asset, such as derivatives."
        },
        {
            "question": "What is modern portfolio theory (MPT)?",
            "answer": "Modern Portfolio Theory (MPT) suggests that investors can optimize returns by constructing diversified portfolios with a specific risk-return balance."
        },
        {
            "question": "What is beta risk in a portfolio?",
            "answer": "Beta risk measures a portfolio's sensitivity to market movements. A beta of 1 indicates the portfolio moves in line with the market."
        },
        {
            "question": "What are derivatives?",
            "answer": "Derivatives are financial instruments whose value derives from an underlying asset, such as stocks, bonds, or commodities."
        },
        {
            "question": "What is stress testing in finance?",
            "answer": "Stress testing evaluates how a financial institution or portfolio performs under extreme adverse market conditions."
        },
        {
            "question": "What is the Efficient Frontier?",
            "answer": "The Efficient Frontier represents portfolios offering the highest expected return for a given level of risk, according to Modern Portfolio Theory."
        },
        {
            "question": "What is credit default swap (CDS)?",
            "answer": "A CDS is a financial derivative allowing investors to hedge or speculate on the credit risk of a borrower defaulting on their debt."
        },

        # Financial Regulations and Compliance
        {
            "question": "What is Basel III?",
            "answer": "Basel III is a global regulatory framework aimed at strengthening bank capital requirements, risk management, and liquidity standards."
        },
        {
            "question": "What is Anti-Money Laundering (AML)?",
            "answer": "AML refers to a set of regulations and procedures designed to prevent illegal activities like money laundering and terrorist financing."
        },
        {
            "question": "What is Know Your Customer (KYC)?",
            "answer": "KYC is a compliance process where financial institutions verify customer identities to prevent fraud and financial crimes."
        },
        {
            "question": "What is the Dodd-Frank Act?",
            "answer": "The Dodd-Frank Act is a US law implemented to increase financial stability by improving accountability and transparency in the financial system."
        },
        {
            "question": "What is GDPR in finance?",
            "answer": "The General Data Protection Regulation (GDPR) sets guidelines for data privacy and protection for financial institutions operating in the EU."
        },

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
    key_terms = extract_financial_terms(context)
    
    # Create more professional response templates
    analysis_examples = [
        "When analyzing {term}, it's important to consider that {context}.",
        "In financial terms, {term} refers to {context}.",
        "{term} is a fundamental concept in finance that {context}."
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
    max_samples: int = 35_000,  # Increased for more samples
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

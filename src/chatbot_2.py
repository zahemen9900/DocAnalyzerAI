from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from table_analyzer import TableAnalyzer
import re
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialChatbot:
    def __init__(self, model_name="facebook/blenderbot-400M-distill"):
        """Initialize chatbot with financial analysis capabilities"""
        # Initialize base model
        self._init_model(model_name)
        
        # Initialize table analyzer
        # self.analyzer = TableAnalyzer()
        
        # State management
        self.conversation_history = []
        self.current_context = {}
        self.last_metric_discussed = None
        
        # Define financial query patterns
        self.query_patterns = {
            'growth_rate': r'growth.*(rate|percentage).*(\d{4})',
            'metric_value': r'(what|how much).*(revenue|sales|income|profit).*(\d{4})',
            'comparison': r'compare.*(revenue|sales|income|profit).*(\d{4}).*(\d{4})',
            'trend': r'trend.*(revenue|sales|income|profit)',
        }
        
        # Financial metric explanations
        self.metric_explanations = {
            'net_sales': "Net sales represents the company's total revenue from goods and services",
            'operating_income': "Operating income shows profit from core business operations",
            'net_income': "Net income is the company's total profit after all expenses and taxes",
            'research_development': "R&D expenses show investment in future innovation",
        }

    def _init_model(self, model_name: str) -> None:
        """Initialize the base conversation model"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise

    def _identify_query_type(self, query: str) -> str:
        """Identify the type of financial query"""
        query = query.lower()
        for query_type, pattern in self.query_patterns.items():
            if re.search(pattern, query):
                return query_type
        return 'general'

    def _extract_metrics_and_years(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract financial metrics and years from query"""
        metrics = re.findall(r'(revenue|sales|income|profit|research|development)', query.lower())
        years = re.findall(r'\b(20\d{2})\b', query)
        return metrics, years

    def _format_financial_response(self, value: float, metric: str) -> str:
        """Format financial values with explanations"""
        formatted_value = f"${value:,.2f}"
        explanation = self.metric_explanations.get(metric, "")
        return f"{formatted_value}. {explanation}"

    def _generate_trend_analysis(self, metric: str, data: Dict) -> str:
        """Generate trend analysis for a metric"""
        values = [float(v.strip('%')) for v in data.values()]
        trend = "increasing" if sum(values) > 0 else "decreasing"
        avg_change = sum(values) / len(values)
        return f"The {metric} shows a {trend} trend with an average change of {avg_change:.2f}% per year."

    def process_financial_query(self, query: str, df) -> str:
        """Process financial queries using TableAnalyzer"""
        try:
            query_type = self._identify_query_type(query)
            metrics, years = self._extract_metrics_and_years(query)
            
            if query_type == 'growth_rate' and metrics:
                metric = metrics[0]
                growth_rates = self.analyzer.calculate_growth_rate(df, metric)
                self.last_metric_discussed = metric
                return f"Growth rates for {metric}: {growth_rates}"
                
            elif query_type == 'metric_value':
                processed_query = self.analyzer.preprocess_query(query)
                answer = self.analyzer.query_table(df, processed_query)
                return answer
                
            elif query_type == 'trend':
                metric = metrics[0] if metrics else self.last_metric_discussed
                if metric:
                    growth_rates = self.analyzer.calculate_growth_rate(df, metric)
                    return self._generate_trend_analysis(metric, growth_rates)
                    
            return self.analyzer.query_table(df, query)
            
        except Exception as e:
            logger.error(f"Error processing financial query: {e}")
            return "I apologize, but I couldn't process that financial query. Could you rephrase it?"

    def chat(self, user_input: str, financial_data=None) -> str:
        """Main chat interface with context management"""
        try:
            # Store conversation context
            self.conversation_history.append(("user", user_input))
            
            # Check if it's a financial query
            if financial_data is not None:
                response = self.process_financial_query(user_input, financial_data)
                if response and "Error" not in response:
                    self.conversation_history.append(("bot", response))
                    return response
            
            # Fall back to general conversation
            inputs = self.tokenizer(user_input, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=100)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            self.conversation_history.append(("bot", response))
            return response
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return "I apologize, but I encountered an error. Could you try asking another way?"

    def suggest_related_queries(self, last_metric: Optional[str] = None) -> List[str]:
        """Suggest related queries based on conversation context"""
        suggestions = []
        if last_metric:
            suggestions.extend([
                f"What's the growth rate for {last_metric}?",
                f"Can you show the trend in {last_metric}?",
                f"Compare {last_metric} between different years"
            ])
        return suggestions

    def reset_conversation(self) -> None:
        """Reset conversation state"""
        self.conversation_history = []
        self.current_context = {}
        self.last_metric_discussed = None

def main():
    print("Initializing chatbot... (this may take a moment)")
    chatbot = FinancialChatbot()
    print("Chatbot is ready! Type 'quit' to exit or 'reset' to start a new conversation.")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'reset':
            chatbot.reset_conversation()
            print("Chat history has been reset.")
            continue
        
        if user_input:
            response = chatbot.chat(user_input)
            print(f"Bot: {response}")

if __name__ == "__main__":
    main()

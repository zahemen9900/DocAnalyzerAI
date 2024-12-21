from typing import Dict, List, Optional
from pathlib import Path
import logging
import torch
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialChatbot:
    """A chatbot specialized in financial document analysis and queries."""
    
    def __init__(self, huggingface_api_key: str):
        """Initialize the FinancialChatbot.
        
        Args:
            huggingface_api_key (str): Hugging Face API key for model access
        """
        # Set up Hugging Face API key
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
        
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize LLM with a lightweight model suited for text generation
        self.llm = HuggingFaceEndpoint(
            repo_id="distilgpt2",
            temperature=0.2,
            huggingfacehub_api_token=huggingface_api_key
        )
        
        # Initialize embeddings with a lightweight but effective model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                'device': self.device
            },
            encode_kwargs={
                'normalize_embeddings': True  # This helps with similarity search
            }
        )
        
        # Optimize chunk size for better memory usage
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", ","]
        )
        
        # Use windowed memory to prevent memory growth
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=5
        )
        
        self.vector_store = None
        self.qa_chain = None
        
        # Create prompt template for financial analysis with more specific instructions
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst assistant specialized in analyzing financial documents and data. 
            Focus on providing clear, data-driven insights with specific numbers and trends.
            Keep responses concise and factual."""),
            ("human", "{question}")
        ])
        
        logger.info("FinancialChatbot initialized successfully")

    def load_financial_data(self, company_dir: Path) -> None:
        """Load and process financial documents for a company."""
        try:
            documents = []
            
            # Load all text files from company directory
            for file_path in company_dir.glob("*.txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.extend(self.text_splitter.split_text(content))
            
            # Create vector store using FAISS with GPU if available
            self.vector_store = FAISS.from_texts(
                documents,
                self.embeddings,
                metadatas=[{"source": str(doc_id)} for doc_id in range(len(documents))]
            )
            
            # Set up the retrieval chain
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            )
            
            # Create the chain with optimized configuration
            self.qa_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info(f"Successfully loaded financial data from {company_dir}")
            
        except Exception as e:
            logger.error(f"Error loading financial data: {e}")
            raise

    def ask(self, question: str) -> Dict:
        """Ask the chatbot a question about the loaded financial data."""
        if not self.qa_chain:
            raise ValueError("No financial data loaded. Call load_financial_data first.")
            
        try:
            # Format the question to encourage concise analysis
            formatted_question = f"""Analyze this financial question and provide a clear, concise response with key numbers and trends: {question}"""
            
            # Get response from the chain
            response = self.qa_chain.invoke(formatted_question)
            
            # Get relevant sources
            sources = self.vector_store.similarity_search(
                formatted_question,
                k=2
            )
            
            return {
                "answer": response,
                "sources": [doc.page_content for doc in sources]
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "answer": "I encountered an error processing your question.",
                "sources": []
            }

    def analyze_financial_metrics(self, metric_type: str) -> Dict:
        """Analyze specific financial metrics from the loaded data."""
        metric_questions = {
            "revenue": "What are the key revenue trends?",
            "profit": "What are the main profit margins?",
            "cash_flow": "What are the essential cash flow metrics?",
            "expenses": "What are the major expense trends?",
            "growth": "What are the primary growth indicators?"
        }
        
        if metric_type not in metric_questions:
            raise ValueError(f"Unsupported metric type. Choose from: {list(metric_questions.keys())}")
        
        return self.ask(metric_questions[metric_type])

    def get_financial_summary(self) -> Dict:
        """Generate a concise financial summary from the loaded data."""
        try:
            summary_question = """Provide a brief financial summary covering:
            1. Revenue trends
            2. Profit margins
            3. Cash flow status
            4. Key expenses
            5. Growth metrics
            Focus on the most important numbers and trends."""
            
            return self.ask(summary_question)
            
        except Exception as e:
            logger.error(f"Error generating financial summary: {e}")
            return {
                "answer": "Failed to generate financial summary.",
                "sources": []
            }


def main():
    """Example usage of FinancialChatbot"""
    try:
        # Initialize chatbot with your Hugging Face API key
        chatbot = FinancialChatbot("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")
        
        # Load financial data for a company
        company_dir = Path.cwd() / "data" / "TSLA"
        chatbot.load_financial_data(company_dir)
        
        # Example usage of different query methods
        
        # 1. Simple question
        print("\n=== Basic Question ===")
        response = chatbot.ask("What were Tesla's main sources of revenue in 2023?")
        print(f"Answer: {response['answer']}\n")
        
        # 2. Specific metric analysis
        print("\n=== Revenue Analysis ===")
        revenue_analysis = chatbot.analyze_financial_metrics("revenue")
        print(f"Revenue Analysis: {revenue_analysis['answer']}\n")
        
        # 3. Comprehensive summary
        print("\n=== Financial Summary ===")
        summary = chatbot.get_financial_summary()
        print(f"Summary: {summary['answer']}\n")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
import gradio as gr
import torch
import logging
from pathlib import Path
from finetuned_chatbot_testing import load_model_and_tokenizer, generate_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example questions organized by category
TOPICS = {
    "Investment Basics": [
        "What's the best way to start investing with limited funds?",
        "Can you explain what a mutual fund is?",
        "What's the difference between stocks and bonds?",
        "How does compound interest work?",
    ],
    "Market Analysis": [
        "What defines a bull market vs bear market?",
        "How do interest rates affect the stock market?",
        "What is market capitalization?",
        "How do I analyze stock performance?",
    ]
}

def load_financial_model():
    """Load the financial model and tokenizer"""
    try:
        adapter_dir = "results/financial-bot-qlora/final_adapter"
        model, tokenizer = load_model_and_tokenizer(adapter_dir)
        logger.info("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def get_bot_response(message, history, model, tokenizer):
    """Generate response using the model"""
    try:
        response = generate_response(model, tokenizer, message)
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I apologize, but I encountered an error. Please try again."

def create_demo():
    """Create the Gradio interface"""
    try:
        # Load model first
        model, tokenizer = load_financial_model()
        
        # Define interface
        with gr.Blocks(
            title="Financial Advisory Assistant",
            css="#chatbot {height: 400px; overflow-y: auto;}"
        ) as demo:
            # Header
            gr.Markdown(
                """
                # 🤖 Financial Advisory Assistant
                Get expert guidance on investments, personal finance, and market analysis.
                
                <div style='padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin: 10px 0;'>
                ⚠️ This is an AI assistant for educational purposes only. 
                Please consult with qualified financial professionals for actual financial advice.
                </div>
                """
            )
            
            # Chat interface
            chatbot = gr.Chatbot(label="Chat")
            msg = gr.Textbox(label="Your message", placeholder="Type your financial question here...")
            clear = gr.Button("Clear")
            
            # Example questions with fixed click handlers
            with gr.Accordion("Example Questions", open=False):
                for topic, questions in TOPICS.items():
                    gr.Markdown(f"### {topic}")
                    for question in questions:
                        # Fixed button handler
                        def create_click_handler(q):
                            return lambda: q
                        
                        btn = gr.Button(question)
                        btn.click(
                            fn=create_click_handler(question),
                            inputs=None,
                            outputs=msg,
                        )
            
            # Handle responses
            def respond(message, chat_history):
                bot_message = get_bot_response(message, chat_history, model, tokenizer)
                chat_history.append((message, bot_message))
                return "", chat_history
            
            # Connect components
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
        
        return demo
        
    except Exception as e:
        logger.error(f"Error creating demo: {e}")
        raise

if __name__ == "__main__":
    try:
        # Create and launch the demo
        demo = create_demo()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True
        )
    except Exception as e:
        logger.error(f"Failed to launch demo: {e}")
        raise

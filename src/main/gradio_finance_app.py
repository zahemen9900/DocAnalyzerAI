import gradio as gr
from matplotlib.font_manager import font_scalings
import torch
import logging
from pathlib import Path

from websocket import send
from finetuned_chatbot_testing import load_model_and_tokenizer, generate_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced topics with emoji icons
TOPICS = {
    "💰 Investment Basics": [
        "What's the best way to start investing with limited funds?",
        "Can you explain what a mutual fund is?",
        "What's the difference between stocks and bonds?",
        "How does compound interest work?",
    ],
    "📊 Market Analysis": [
        "What defines a bull market vs bear market?",
        "How do interest rates affect the stock market?",
        "What is market capitalization?",
        "How do I analyze stock performance?",
    ],
    "💵 Personal Finance": [
        "How do I create an effective budget?",
        "What's an emergency fund and how much should I save?",
        "How can I improve my credit score?",
        "What's the best way to manage debt?",
    ],
    "🎯 Retirement Planning": [
        "How do I start planning for retirement?",
        "What's the difference between a 401(k) and IRA?",
        "How much should I save for retirement?",
        "What is dollar-cost averaging?",
    ]
}

# Enhanced CSS styling
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');

/* Base font inheritance */
:root {
    --font-sans-serif: 'Inter', sans-serif !important;
    --font-mono: 'Space Mono', monospace !important;
}

/* Global dark theme */
:root {
    color-scheme: dark !important;
}

.gradio-container {
    background-color: #1a1b1e !important;
    color: #e0e0e0 !important;
}

.dark {
    background-color: #1a1b1e !important;
}

.container {
    max-width: 800px !important;
    margin-top: 1rem !important;
    margin: auto !important;
    padding: 0 1rem !important;
    background-color: #1a1b1e !important;
    color: #171717 !important;
    font-family: 'Inter', sans-serif !important;
    border-radius: 10px !important;
}

.container-questions {
    max-width: 900px !important;
    margin: auto !important;
    padding: 0 1rem;
    background-color: #1a1b1e !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif !important;
}

.header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 1.5rem;
    background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%);
    color: white;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    font-family: 'Space Mono', monospace !important;
}

.header h1 {
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    font-size: 5em !important;
}

.header p {
    font-weight: 400 !important;
    opacity: 0.9;
}

.disclaimer {
    padding: 1rem;
    background-color: #2d2d2d;
    border-left: 4px solid #6B73FF;
    border-radius: 5px;
    margin: 1rem 0;
    margin-bottom: 1rem;
    font-size: 0.8em;
    color: #e0e0e0;
}

.feature-card {
    background: var(--background-fill-primary);
    padding: 20px !important;  /* Adjusted padding */
    margin: 10px !important;   /* Added margin */
    border-radius: 12px !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    transition: transform 0.2s;
    border: 1px solid var(--border-color-primary);
    min-height: 120px !important;  /* Set minimum height */
    display: flex !important;
    align-items: center !important;
    font-size: 1.1em !important;   /* Increased font size */
    transition: transform 0.3s ease, box-shadow 0.3s ease, background-color 0.3s ease !important;
}

.feature-card:hover {
    transform: translateY(-5px);
    background: var(--background-fill-secondary);
    box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
}

.chatbot {
    font-family: 'Inter', sans-serif !important;
    height: 450px !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    background-color: #0f0f11 !important;
    border: 1px solid #404040 !important;
    animation: fadeIn 0.5s ease-in-out;
}

.input-container {
    margin: 0.5px !important;
    background-color: #2d2d2d !important;
    padding: 0.1rem !important;  /* Reduced padding */
    border-radius: 10px;
}

.input-row {
    display: flex !important;
    gap: 0.5rem !important;
    align-items: flex-start !important;
    margin-top: 0.2rem !important;  /* Reduced margin */
}

.action-button {
    background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%) !important;
    color: white !important;
    height: 68px !important;
    border-radius: 5px !important;
    padding: 0.8rem 1rem !important;  /* Increased padding for bigger buttons */
    transition: all 0.3s ease !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 800 !important;
    letter-spacing: 0.5px !important;
    font-size: 1.1em !important;  /* Increased font size */
    position: relative;
    overflow: hidden;
}

.action-button::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(rgba(255,255,255,0.1), rgba(255,255,255,0));
    transform: translateY(-100%);
    transition: transform 0.3s ease;
}

.action-button:hover {
    opacity: 0.9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
}

.clear-button {
    background: linear-gradient(135deg, #9c9ec7 0%, #212122 100%) !important;
    color: white !important;
    border-radius: 5px !important;
    padding: 0.8rem 1rem !important;  /* Increased padding for bigger buttons */
    transition: all 0.3s ease !important;
    border: none !important;
    font-size: 0.8em !important;  /* Increased font size */
}

.clear-button:hover {
    opacity: 0.9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
}

.char-counter {
    text-align: center;
    color: #888;
    font-size: 0.8em;
    margin: 0 !important;
    padding: 0 !important;
}

.examples-header {
    color: #6B73FF;
    font-size: 1.2em;
    font-weight: 600;
    margin: 1.5rem 0 1rem 0;
    text-align: center;
    font-family: 'Space Mono', monospace;
}

.example-category {
    color: #6B73FF;
    font-weight: 600;
    margin: 1rem 0 0.5rem 0;
    font-family: 'Inter', sans-serif;
}

.example-bubble {
    background: #2d2d2d;
    border: 1px solid #404040;
    border-radius: 15px;
    padding: 0.8rem 1.2rem;
    margin: 0.4rem 0;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 0.95em;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.example-bubble:hover {
    transform: translateX(5px);
    background: #363636;
    border-color: #6B73FF;
    box-shadow: 0 2px 8px rgba(107, 115, 255, 0.2) !important;
}

.accordion-container {
    max-width: 770px !important;
    margin: auto !important;
    padding: 1rem !important;
    background-color: #2d2d2d !important;
    border-radius: 10px !important;
    border: 1px solid #404040 !important;
    margin-top: 1.5rem !important;
}

.accordion-header {
    color: #6B73FF !important;
    font-size: 1.1em !important;
    font-weight: 600 !important;
    text-align: center !important;
    font-family: 'Space Mono', monospace !important;
    padding: 1rem !important;  /* Added padding for better height */
    white-space: nowrap !important;  /* Prevent text from wrapping */
    overflow: hidden !important;  /* Prevent scroll */
    text-overflow: ellipsis !important;  /* Add ellipsis for overflow */
}

/* Fix processing text color */
.processing-text {
    color: #e0e0e0 !important;
}

/* Enhanced animations */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Smoother transitions */
* {
    transition: background-color 0.3s ease, border-color 0.3s ease !important;
}

/* Enhanced focus states */
button:focus, input:focus {
    outline: 2px solid #6B73FF !important;
    outline-offset: 2px !important;
}
"""

# Example questions organized by category
EXAMPLES = {
    "## 💰 Investment Basics": [
        "What's the best way to start investing with limited funds?",
        "Can you explain what a mutual fund is?",
        "What's the difference between stocks and bonds?",
        "How does compound interest work?",
    ],
    "## 📊 Market Analysis": [
        "What defines a bull market vs bear market?",
        "How do interest rates affect the stock market?",
        "What is market capitalization?",
        "How do I analyze stock performance?",
    ],
    "## 💵 Personal Finance": [
        "How do I create an effective budget?",
        "What's an emergency fund and how much should I save?",
        "How can I improve my credit score?",
        "What's the best way to manage debt?",
    ],
    "## Retirement Planning": [
        "How do I start planning for retirement?",
        "What's the difference between a 401(k) and IRA?",
        "How much should I save for retirement?",
        "What is dollar-cost averaging?",
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
    """Create enhanced Gradio interface"""
    try:
        model, tokenizer = load_financial_model()
        
        with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Monochrome) as demo:
            with gr.Column(elem_classes="container"):
                # Enhanced header
                gr.Markdown(
                    """
                    <div class="header">
                        <h1>🤖 FinBot AI</h1>
                        <p>Your AI companion for financial guidance and market insights</p>
                    </div>
                    
                    """
                )
                
                # Feature cards
                with gr.Row(elem_id="features-grid"):
                    with gr.Column(elem_classes=["feature-card"]):
                        gr.Markdown("##### 💼 **Investment Advice** \n##### Get expert advice on how to invest your money wisely.")
                    with gr.Column(elem_classes=["feature-card"]):
                        gr.Markdown("##### 📊 **Market Analysis** \n##### Understand market trends and make informed decisions.")
                    with gr.Column(elem_classes=["feature-card"]):
                        gr.Markdown("##### 💵 **Personal Finance, and More!** \n##### Manage your personal finances effectively, and get answers to general queries")
                gr.Markdown(
                    """
                    <div class="disclaimer">
                        <b>💡Tip:</b> To get more accurate responses, keep your questions concise and specific.
                    </div>
                    """
                )


            with gr.Column(elem_classes="container"):                
                # Enhanced chat interface
                chatbot = gr.Chatbot(
                    elem_classes="chatbot",
                    label="Financial Conversation",
                    bubble_full_width=False,
                    height=450,
                    show_label=False
                )
                
                # Simplified input area without container
                with gr.Row():
                    msg = gr.Textbox(
                        label=None,
                        container = False,
                        placeholder="Chat with me about finance, or anything else...",
                        lines=2,
                        elem_classes="input-container",
                        scale = 11
                    )
                    submit = gr.Button("Send 📤", elem_classes="action-button", scale = 1)

                
                # with gr.Row():
                #     clear = gr.Button("Clear 🗑️", elem_classes="clear-button", scale = 2)
                #     char_counter = gr.HTML(
                #         value='<p class="char-counter">0/512 characters</p>',
                #         elem_classes="char-counter"
                #     )

                    
            # Collapsible example questions section
            with gr.Accordion("Example Questions (Click to expand)", open=False, elem_classes="accordion-container"):
                gr.Markdown("## 💭 Example Questions You Can Ask:", elem_classes="accordion-header")
                
                for category, questions in EXAMPLES.items():
                    gr.Markdown(f"##{category}", elem_classes="example-category")
                    for question in questions:
                        example_btn = gr.Button(
                            question,
                            elem_classes="example-bubble"
                        )
                        example_btn.click(
                            fn=lambda x=question: x,  # Use closure to capture question
                            inputs=[],
                            outputs=msg
                        )

            with gr.Column(elem_classes="container"): 

                gr.Markdown(
                    """
                    <div class="disclaimer">
                        <em>⚠️ AI can make mistakes sometimes. Please consult with qualified financial professionals for actual financial advice.</em>
                    </div>
                    """
                )
                
            # # Character counter update function
            # def count_chars(text):
            #     return f'<p class="char-counter">{len(text)}/512 characters</p>'

            # # Update character counter on input
            # msg.change(count_chars, msg, char_counter)
            
            # Enhanced response handler with loading state
            def respond(message, chat_history):
                if not message.strip():
                    return "", chat_history
                
                try:
                    bot_message = get_bot_response(message, chat_history, model, tokenizer)
                    chat_history.append((message, bot_message))
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    bot_message = "I apologize, but I encountered an error. Please try again."
                    chat_history.append((message, bot_message))
                
                return "", chat_history

            # Connect components
            submit.click(respond, [msg, chatbot], [msg, chatbot])
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            # clear.click(lambda: None, None, chatbot, queue=False)
        
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
            server_port=7867,
            share=True
        )
    except Exception as e:
        logger.error(f"Failed to launch demo: {e}")
        raise

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
import gc
from typing import List, Tuple
import logging

# --------------------------------------------------------
# 📊 **Financial Advisory Assistant: AI-Powered Guidance**
# --------------------------------------------------------
# This AI-powered chatbot is designed to provide **general financial advice** 
# and assist users with common financial queries. 
# It leverages advanced **natural language processing (NLP)** capabilities, 
# using a fine-tuned Transformer-based language model. 
# ✅ **Purpose:** Assist users with financial planning, investment advice, and budgeting strategies.
# ✅ **Capabilities:** Real-time conversation, quick answers to financial questions, 
# and tailored suggestions.
# ✅ **Intended Audience:** Individuals seeking basic financial advice and planning assistance.
# 🛡️ *Disclaimer: Always consult a certified financial advisor for critical financial decisions.*

# --------------------------------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
BASE_MODEL = "facebook/blenderbot-400M-distill"
ADAPTER_DIR = "results/financial-bot-qlora/final_adapter"

def setup_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
        ).to(device)

        model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_DIR,
            torch_dtype=torch.float16,
        ).to(device)

        model.config.use_cache = True
        model.eval()
        logger.info(f"Model loaded on: {next(model.parameters()).device}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

model, tokenizer = setup_model()

def generate_response(message: str) -> str:
    try:
        device = next(model.parameters()).device
        prompt = f"User: {message}"
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)

        generation_config = GenerationConfig(
            max_length=256,
            min_length=32,
            num_beams=4,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        logging.error(f"Generation error: {str(e)}")
        return "I apologize, but I encountered an error. Please try again."

def chat(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """Chat function that maintains conversation history"""
    if message:
        bot_message = generate_response(message)
        history.append((message, bot_message))
    return "", history

# Create the Gradio interface with a dark theme
theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
    font=["Quicksand", "sans-serif"],
).set(
    background_fill_primary="*neutral_950",
    block_background_fill="*neutral_900",
    input_background_fill="*neutral_800",
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("""
    # 🤖 Financial Advisory AI
    Your intelligent companion for financial guidance
    """)
    
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Your message", placeholder="Type your question here...")
    clear = gr.ClearButton([msg, chatbot])
    
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    
    gr.Examples(
        examples=[
            "What's the best way to start investing with limited funds?",
            "How can I create an effective budget?",
            "What strategies do you recommend for retirement planning?",
            "Can you explain the basics of stock market investing?",
        ],
        inputs=msg,
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        share=True,
        debug=True
    )

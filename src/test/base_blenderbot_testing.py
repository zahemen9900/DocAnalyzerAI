from ast import Is
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
import logging
import gc
import sys
import time
from typing import Iterator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('base_model_testing.log')
    ]
)
logger = logging.getLogger(__name__)

def stream_text(text: str, delay: float = 0.05) -> Iterator[str]:
    """Stream text word by word with a delay"""
    words = text.split()
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)

def load_base_model(model_name: str = "facebook/blenderbot-1B-distill"):
    """Load the base BlenderBot model"""
    try:
        logger.info(f"Loading base model {model_name}...")
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Enable caching and set to eval mode
        model.config.use_cache = True
        model.eval()
        
        logger.info(f"Model loaded successfully on: {next(model.parameters()).device}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_response(
    model, 
    tokenizer, 
    input_text: str,
    max_length: int = 256,
    min_length: int = 92,
    temperature: float = 0.3,
    num_beams: int = 4
) -> str:
    """Generate a response from the base model"""
    try:
        # Prepare input
        inputs = tokenizer(
            input_text.strip(), 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length,
            padding=True
        ).to(model.device)
        
        # Setup generation config
        generation_config = GenerationConfig(
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            length_penalty=2.0,
            repetition_penalty=2.0
        )
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response.strip()
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while processing your request."

def main(IS_1B: bool = True):
    try:
        # Load model
        if IS_1B:
            model, tokenizer = load_base_model()
        else:
            model, tokenizer = load_base_model("facebook/blenderbot-3B")
        logger.info("Base model chatbot initialized successfully!")
        
        print("\n🤖 BlenderBot Base Model Chat")
        print("Type 'exit' to end the conversation\n")
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'exit':
                print("👋🏾 Bye bye!")
                break
            
            response = generate_response(model, tokenizer, user_input)
            print("Assistant: ", end="", flush=True)
            for word in stream_text(response):
                print(word, end="", flush=True)
            print()
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()

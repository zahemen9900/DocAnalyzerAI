import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel, PeftConfig
import logging
import gc
import re
import sys
import time
from typing import Iterator, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatbot_testing.log')
    ]
)
logger = logging.getLogger(__name__)

# Add response filtering patterns
DISCOURAGED_PATTERNS = [
    # r"Financial Experience is",
    # r"Personal finance is",
    # r"I am a financial advisor",
    # r"Do you know anyone",
    r"I have .* saved",
    r"My (?:husband|wife|ex-wife)",
    # r"Do you have .*\?",
    r"What do you .*\?",
    r"Personal finance (?:is|refers to|means|describes|encompasses).*?[.]",  # Match sentences starting with "Personal finance"
    r"Financial Experience (?:is|refers to|means|describes|encompasses).*?[.]",  # Match sentences starting with "Personal finance"
    r"(?:It's|It is) (?:a )?(?:very |really |quite )?interesting (?:topic|subject).*?[.]",  # Match variations of "interesting topic"
    r"This is (?:a )?(?:very |really |quite )?interesting (?:topic|subject).*?[.]",
    r"(?:This|That) makes it (?:very |really |quite )?interesting.*?[.]",
]

def filter_response(response: str) -> str:
    """Filter out unwanted patterns and improve response quality"""
    # Remove discouraged patterns
    for pattern in DISCOURAGED_PATTERNS:
        response = re.sub(pattern, "", response)
    
    # Clean up multiple spaces and punctuation
    response = re.sub(r'\s+', ' ', response)
    response = re.sub(r'\.+', '.', response)
    response = re.sub(r'\s+\.', '.', response)
    
    # Remove starting words if they begin with connecting words after filtering
    response = re.sub(r'^(?:And|But|So|Therefore|Thus|Hence|However|Moreover)\s*,?\s*', '', response.strip())
    
    return response.strip()

def stream_text(text: str, delay: float = 0.05) -> Iterator[str]:
    """Stream text word by word with a delay"""
    words = text.split()
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)

def setup_quantization_config():
    """Setup 4-bit quantization configuration"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def load_model_and_tokenizer(adapter_dir, base_model="facebook/blenderbot-400M-distill"):
    """Load the LoRA-adapted model and tokenizer"""
    try:
        logger.info(f"Loading base model {base_model}...")
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Load base model with quantization
        quant_config = setup_quantization_config()
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
        ).to(device)  # Explicitly move to device
        
        # Load LoRA adapter
        logger.info(f"Loading LoRA adapter from {adapter_dir}...")
        model = PeftModel.from_pretrained(
            base_model,
            adapter_dir,
            torch_dtype=torch.float16,
        ).to(device)  # Explicitly move to device
        
        # Set pad token and config
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Set generation config
        model.config.use_cache = True  # Enable KV cache for inference
        model.eval()
        
        # Verify device placement
        device_info = next(model.parameters()).device
        logger.info(f"Model loaded successfully on: {device_info}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_response(
    model, 
    tokenizer, 
    input_text, 
    max_length=128, 
    min_length=48,
    temperature=0.25,
    num_beams=4
):
    """Generate a response with better controls"""
    try:
        # Prepare input with explicit persona
        input_text = f"Personas: Financial Expert\nUser: {input_text.strip()}"
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length,
            padding=True
        )
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Enhanced generation configuration
        generation_config = GenerationConfig(
            max_length=max_length,
            min_length=min_length,  # Now controlled by parameter
            num_beams=num_beams,    # Now controlled by parameter
            temperature=temperature,  # Now controlled by parameter
            top_k=30,
            top_p=0.85,
            do_sample=True,
            no_repeat_ngram_size=4,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            length_penalty=2.0,
            repetition_penalty=2.0
        )
        
        # Generate with stricter settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode and clean response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace("Assistant:", "").strip()
        
        # Apply response filtering
        filtered_response = filter_response(response)
        
        return filtered_response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while processing your request."

def main():
    try:
        adapter_dir = "results/financial-bot-qlora/final_adapter"
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(adapter_dir)
        logger.info("Chatbot initialized successfully!")
        
        print("\n🤖 Financial Advisory Assistant")
        print("Type 'exit' to end the conversation\n")
        
        # Example financial questions to get started
        print("Example questions you can ask:")
        print("- What's the best way to start investing?")
        print("- Can you explain what a mutual fund is?")
        print("- How do I create a budget?")
        print("- What's the difference between stocks and bonds?\n")
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'exit':
                break

            response = generate_response(model, tokenizer, user_input)
            print("Assistant: ", end="", flush=True)
            for word in stream_text(response):
                print(word, end="", flush=True)
            print()  # New line after response
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()

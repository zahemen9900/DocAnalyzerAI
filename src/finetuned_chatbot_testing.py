import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_dir):
    """Load the trained model and tokenizer from the specified directory"""
    logger.info(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    return model, tokenizer

def generate_response(model, tokenizer, input_text, max_length=512):
    """Generate a response from the model given an input text"""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    model_dir = "results/financial-bot-gpu/final"
    model, tokenizer = load_model_and_tokenizer(model_dir)
    
    logger.info("Model and tokenizer loaded successfully.")
    
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = generate_response(model, tokenizer, user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()

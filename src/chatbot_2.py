from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import torch.cuda
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Chatbot:
    def __init__(self, model_name="facebook/blenderbot-400M-distill"):
        """Initialize chatbot with simplified memory management"""
        # GPU handling with error checking
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                # Set conservative CUDA settings
                torch.cuda.empty_cache()
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = False  # More stable, less memory
                torch.backends.cuda.matmul.allow_tf32 = False  # More stable
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device("cpu")
                logger.warning("GPU not detected. Running on CPU may be slow.")

            # Load model with conservative settings
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            raise

    def chat(self, user_input: str, temperature: float = 0.7) -> str:
        """Generate a response without maintaining conversation history"""
        try:
            # Clean GPU memory before processing
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            # Input validation
            if not user_input.strip():
                return "Please provide a valid input."

            # Conservative tokenization settings
            inputs = self.tokenizer(
                [user_input],
                return_tensors="pt",
                max_length=512,  # Reduced for stability
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                reply_ids = self.model.generate(
                    **inputs,
                    max_length=256,  # Reduced for stability
                    min_length=100,
                    num_beams=3,     # Reduced for stability
                    temperature=temperature,
                    do_sample=True,
                    no_repeat_ngram_size=2,
                    top_k=40,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(reply_ids[0], skip_special_tokens=True)

            # Clean up memory after generation
            del inputs, reply_ids
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return "I apologize, but I encountered an error processing your request."

    def reset_chat(self) -> None:
        """Reset GPU memory"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        logger.info("Memory cleared")

def main():
    print("Initializing chatbot... (this may take a moment)")
    chatbot = Chatbot()
    print("Chatbot is ready! Type 'quit' to exit or 'reset' to start a new conversation.")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'reset':
            chatbot.reset_chat()
            print("Chat history has been reset.")
            continue
        
        if user_input:
            response = chatbot.chat(user_input)
            print(f"Bot: {response}")

if __name__ == "__main__":
    main()

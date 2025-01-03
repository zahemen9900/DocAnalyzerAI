import torch
import os
import re
# Set debugging flags
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, TrainerCallback
from datasets import Dataset, DatasetDict
import json
import logging
import gc
from torch.cuda.amp import GradScaler
import torch.nn as nn
import evaluate
from torch.optim import AdamW
from transformers import get_scheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def check_gpu():
    """Enhanced GPU check with forced CUDA usage"""
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available!")
    
    # Force CUDA device selection
    torch.cuda.set_device(0)  # Use first GPU
    device = torch.device("cuda:0")
    
    # Verify CUDA is being used
    dummy_tensor = torch.tensor([1.0]).to(device)
    if not dummy_tensor.is_cuda:
        raise RuntimeError("Failed to move tensor to CUDA!")
    
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    logger.info(f"Using GPU: {gpu_properties.name}")
    logger.info(f"Total GPU Memory: {gpu_properties.total_memory / 1024**3:.2f} GB")
    
    return device

def verify_tensor_shapes(model_inputs, tokenizer):
    """Verify tensor shapes before training"""
    logger.info(f"Input IDs shape: {model_inputs['input_ids'].shape}")
    logger.info(f"Attention mask shape: {model_inputs['attention_mask'].shape}")
    if 'labels' in model_inputs:
        logger.info(f"Labels shape: {model_inputs['labels'].shape}")
    
    # Verify no dimension is 0
    for key, tensor in model_inputs.items():
        if 0 in tensor.shape:
            raise ValueError(f"Invalid shape in {key}: {tensor.shape}")

class MetricsComputer:
    """Class to handle metrics computation with access to tokenizer"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge_score = evaluate.load("rouge")
        self.bleu_score = evaluate.load("bleu")
        self.meteor = evaluate.load("meteor")
        
    def __call__(self, eval_preds):
        predictions, labels, inputs = eval_preds
        
        # Decode generated tokens
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Format references
        references = [[label] for label in labels]
        
        # Calculate metrics
        metrics = {}
        
        # ROUGE scores
        rouge_output = self.rouge_score.compute(
            predictions=predictions, 
            references=[r[0] for r in references],
            use_aggregator=True
        )
        metrics.update(rouge_output)
        
        # BLEU score
        bleu_output = self.bleu_score.compute(
            predictions=predictions,
            references=references
        )
        metrics['bleu'] = bleu_output['bleu']
        
        # METEOR score
        meteor_output = self.meteor.compute(
            predictions=predictions,
            references=[r[0] for r in references]
        )
        metrics['meteor'] = meteor_output['meteor']
        
        return metrics

def train(
    model_name: str = "facebook/blenderbot-400M-distill",
    dataset_path: str = "finetune_data/train.json",
    output_dir: str = "results/financial-bot-gpu",
    max_length: int = 128,  # Reduced from 256
    batch_size: int = 4,  # Set to 1 for debugging
):
    try:
        # Force GPU setup
        device = check_gpu()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        # Load dataset
        logger.info("Loading dataset...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process data with validation
        processed_data = []
        for item in data[:200]:  # Start with very small dataset (ABt 20% of the training data)
            if not all(key in item for key in ['personas', 'free_messages', 'guided_messages']):
                continue
                
            context = f"Personas: {' | '.join(item['personas'])}\n"
            for user_msg, bot_msg in zip(item['free_messages'], item['guided_messages']):
                if not user_msg.strip() or not bot_msg.strip():
                    continue
                    
                processed_data.append({
                    "input_text": f"{context}User: {user_msg.strip()}",
                    "target_text": f"Assistant: {bot_msg.strip()}"
                })
        
        # Create and split dataset
        dataset = Dataset.from_list(processed_data)
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        # Load model and tokenizer
        logger.info(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, 
            generation_config=None  # Disable generation config warnings
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
            
        # Move model to GPU
        model = model.to(device)
        logger.info(f"Model loaded on: {next(model.parameters()).device}")
        
        if not next(model.parameters()).is_cuda:
            raise RuntimeError("Model not on GPU!")
            
        def preprocess_function(examples):
            """Enhanced preprocessing function"""
            # Clean input text
            inputs = [text.strip() for text in examples["input_text"]]
            
            # Clean target text - remove 'Assistant:' prefix
            targets = [
                re.sub(r'^Assistant:\s*', '', text.strip())
                for text in examples["target_text"]
            ]
            
            model_inputs = tokenizer(
                inputs,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors=None
            )
            
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors=None
                )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Process datasets with smaller batch size
        logger.info("Processing datasets...")
        train_dataset = dataset["train"].map(
            preprocess_function,
            batched=True,
            batch_size=2,
            remove_columns=dataset["train"].column_names,
            desc="Processing training dataset"
        )
        
        val_dataset = dataset["test"].map(
            preprocess_function,
            batched=True,
            batch_size=2,
            remove_columns=dataset["test"].column_names,
            desc="Processing validation dataset"
        )
        
        # Update training arguments with optimized parameters
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=15,  # Increased epochs
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=16,  # Increased for stability
            learning_rate=2e-5,   # Adjusted learning rate
            weight_decay=0.05,  # Increased weight decay
            warmup_ratio=0.1,     # Added warmup
            logging_steps=2,
            eval_steps=5,
            save_steps=30,
            fp16=True,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            gradient_checkpointing=True,
            max_grad_norm=0.5,  # Reduced for better stability
            generation_max_length=128,
            predict_with_generate=True,
            generation_num_beams=4,  # Add beam search
            include_inputs_for_metrics=True,
            remove_unused_columns=True,  # Changed to True
            # Optimization settings
            lr_scheduler_type="cosine_with_restarts",
            dataloader_pin_memory=True,
            group_by_length=True,
            seed=42,
            # Added settings
            label_smoothing_factor=0.1,
            generation_config=None,  # Disable generation config warnings
        )
        
        # Update model configuration
        model.config.use_cache = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for better performance
        model.config.decoder_start_token_id = tokenizer.pad_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.max_length = max_length
        model.config.num_beams = 4

        class EnhancedGradientCallback(TrainerCallback):
            """Enhanced callback with better gradient handling and metrics"""
            def __init__(self):
                self.scaler = torch.amp.GradScaler()
                self.best_loss = float('inf')
                self.patience = 3
                self.patience_counter = 0
                self.metrics = {}
                
            def on_step_begin(self, args, state, control, **kwargs):
                if state.global_step % 50 == 0:
                    torch.cuda.empty_cache()
                    
            def on_step_end(self, args, state, control, model=None, **kwargs):
                if state.global_step > 0 and state.global_step % 100 == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=args.max_grad_norm
                    )
                    self.metrics[f"grad_norm_{state.global_step}"] = grad_norm.item()
                    
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics:
                    current_loss = metrics.get("eval_loss", float('inf'))
                    if current_loss < self.best_loss:
                        self.best_loss = current_loss
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        
                    if self.patience_counter >= self.patience:
                        logger.info("Loss plateau detected, reducing learning rate")
                        for param_group in trainer.optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        self.patience_counter = 0

        def setup_model_and_optimizer(model, args):
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() 
                              if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() 
                              if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
            
            return model, optimizer

        # Create metrics computer with tokenizer
        metrics_computer = MetricsComputer(tokenizer)

        # Initialize trainer with enhanced callback
        model, optimizer = setup_model_and_optimizer(model, training_args)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                padding=True,
                pad_to_multiple_of=8,
                label_pad_token_id=tokenizer.pad_token_id
            ),
            compute_metrics=metrics_computer,  # Use instance of MetricsComputer
            callbacks=[EnhancedGradientCallback()],
            optimizers=(optimizer, None)
        )
        
        # Verify model device before training
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"CUDA active: {torch.cuda.is_available()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        
        # Add this before training starts
        def clear_gpu_memory():
            torch.cuda.empty_cache()
            gc.collect()
            
        clear_gpu_memory()

        # Training
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model(f"{output_dir}/final")
        tokenizer.save_pretrained(f"{output_dir}/final")
        logger.info("Training completed and model saved!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        if torch.cuda.is_available():
            logger.error(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        raise
    finally:
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}", exc_info=True)
        raise
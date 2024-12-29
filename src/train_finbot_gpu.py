import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, TrainerCallback
from datasets import Dataset
import json
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available!")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")

def train(
    model_name: str = "facebook/blenderbot-400M-distill",
    dataset_path: str = "finetune_data/train.json",
    output_dir: str = "results/financial-bot-gpu",
    max_length: int = 128,
    batch_size: int = 4
):
    try:
        device = check_gpu()
        torch.backends.cudnn.benchmark = True
        
        # Load dataset
        logger.info("Loading dataset...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process data with validation
        processed_data = []
        for item in data:  # Start with very small dataset
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
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
            
        # Move model to GPU
        model = model.to(device)
        logger.info(f"Model loaded on: {next(model.parameters()).device}")
        
        if not next(model.parameters()).is_cuda:
            raise RuntimeError("Model not on GPU!")
            
        def preprocess_function(examples):
            # Batch inputs
            model_inputs = tokenizer(
                examples["input_text"],
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Batch targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples["target_text"],
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
            
            model_inputs["labels"] = labels["input_ids"]
            
            # Convert to tensors on GPU
            return {
                k: torch.tensor(v).to(device) 
                for k, v in model_inputs.items()
            }
        
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
        
        # Training arguments optimized for performance
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=3e-4,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=50,
            eval_steps=100,
            save_steps=100,
            fp16=True,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            save_total_limit=2,
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            logging_first_step=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=0,
            group_by_length=True,
            seed=42
        )
        
        # Remove the GradientHandlingCallback and directly initialize trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                padding=True,
                pad_to_multiple_of=8
            )
        )
        
        # Simple memory cleanup before training
        torch.cuda.empty_cache()
        gc.collect()

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
    train()

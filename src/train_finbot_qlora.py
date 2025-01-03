import torch
import os
import re
import gc
from datasets import Dataset, DatasetDict
import json
import logging
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    GenerationConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType,
)
import bitsandbytes as bnb
from accelerate import Accelerator
import evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_qlora.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_quantization_config():
    """Setup 4-bit quantization configuration"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def setup_lora_config():
    """Setup LoRA configuration"""
    return LoraConfig(
        r=16,  # Rank of update matrices
        lora_alpha=32,  # Alpha scaling
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        inference_mode=False,
    )

def find_all_linear_layers(model):
    """Find all linear layers for LoRA adaptation"""
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append(name)
    return linear_layers

def check_gpu():
    """Verify GPU availability and setup"""
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available! This script requires a GPU.")
    
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    gc.collect()
    
    # Print GPU info
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return device

def train(
    model_name: str = "facebook/blenderbot-400M-distill",
    dataset_path: str = "finetune_data/train.json",
    output_dir: str = "results/financial-bot-qlora",
    max_length: int = 128,
    batch_size: int = 8, 
):
    try:
        # Check GPU and clear memory
        device = check_gpu()
        
        # Load dataset with error handling
        logger.info("Loading dataset...")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process data with validation
        processed_data = []
        for item in data:  # Remove limit to process full dataset
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

        if not processed_data:
            raise ValueError("No valid data processed from the dataset")

        dataset = Dataset.from_list(processed_data)
        dataset = dataset.train_test_split(test_size=0.3, seed=42)

        # Initialize quantization config with error handling
        try:
            quant_config = setup_quantization_config()
        except Exception as e:
            logger.error("Failed to setup quantization config", exc_info=True)
            raise

        # Load tokenizer and model with error checking
        logger.info(f"Loading {model_name} with quantization...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                torch_dtype=torch.float16,
            )
            # Explicitly move model to GPU
            model = model.to(device)
            
        except Exception as e:
            logger.error("Failed to load model or tokenizer", exc_info=True)
            raise

        # Verify model is on GPU
        logger.info(f"Model device: {next(model.parameters()).device}")

        # Setup tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        # Prepare model with error handling
        try:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True
            )

            lora_config = setup_lora_config()
            model = get_peft_model(model, lora_config)
        except Exception as e:
            logger.error("Failed to prepare model for training", exc_info=True)
            raise

        # Update preprocessing function
        def preprocess_function(examples):
            # Clean inputs
            inputs = [re.sub(r'\s+', ' ', text.strip()) for text in examples["input_text"]]
            targets = [re.sub(r'\s+', ' ', text.strip()) for text in examples["target_text"]]
            
            model_inputs = tokenizer(
                inputs,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Process datasets with error handling
        logger.info("Processing datasets...")
        try:
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
        except Exception as e:
            logger.error("Failed to process datasets", exc_info=True)
            raise

        # Updated training arguments for better stability
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,  
            learning_rate=5e-6,  # Reduced learning rate
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=20,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=70,
            save_steps=70,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine_with_restarts",
            group_by_length=True,
            remove_unused_columns=False,  # Added to prevent column removal errors
            ddp_find_unused_parameters=False,  # Added for distributed training
        )

        # Initialize trainer with data collator
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                padding=True,
                label_pad_token_id=tokenizer.pad_token_id
            ),
        )

        # Train with error handling
        logger.info("Starting QLoRA training...")
        try:
            trainer.train()
        except Exception as e:
            logger.error("Training failed", exc_info=True)
            raise

        # Save with error handling
        try:
            logger.info("Saving LoRA adapter...")
            model.save_pretrained(f"{output_dir}/final_adapter")
            tokenizer.save_pretrained(f"{output_dir}/final_adapter")
            
            logger.info("Merging LoRA weights...")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(f"{output_dir}/merged_model")
        except Exception as e:
            logger.error("Failed to save model", exc_info=True)
            raise

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    # Set CUDA device explicitly
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    try:
        train()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}", exc_info=True)
        raise

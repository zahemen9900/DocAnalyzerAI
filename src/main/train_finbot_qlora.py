from altair import Padding
from numpy import full
import torch
import os
import re
import gc
from pathlib import Path
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
    GenerationConfig,
    EarlyStoppingCallback
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType,
)
import bitsandbytes as bnb
from accelerate import Accelerator
import random
import evaluate
import deepspeed
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
import nltk
from wandb import setup
import time
from transformers.trainer_utils import TrainOutput
from datetime import datetime
nltk.download('punkt')

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

def setup_deepspeed_config(training_args: TrainingArguments):
    """Setup DeepSpeed configuration using values from training arguments"""
    return {
        "train_batch_size": "auto",
        "fp16": {
            "enabled": training_args.fp16,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 2,  # Stage 2 is generally good balance of memory and speed
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "gradient_clipping": training_args.max_grad_norm,
        "steps_per_print": training_args.logging_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training_args.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": training_args.weight_decay
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": training_args.max_steps,
                "warmup_min_lr": 0,
                "warmup_max_lr": training_args.learning_rate,
                "warmup_num_steps": int(training_args.warmup_ratio * training_args.max_steps),
            }
        },
    }

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

def compute_metrics(eval_preds, tokenizer):
    """Compute ROUGE and BLEU scores"""
    predictions, labels = eval_preds
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean up predictions and labels
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Initialize metrics
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoother = SmoothingFunction().method1
    
    # Compute ROUGE scores
    rouge_scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }
    
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = rouge.score(label, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    # Compute BLEU score
    references = [[label.split()] for label in decoded_labels]
    predictions = [pred.split() for pred in decoded_preds]
    bleu_score = corpus_bleu(references, predictions, smoothing_function=smoother)
    
    # Average ROUGE scores
    results = {
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL']),
        'bleu': bleu_score
    }
    
    return results

def evaluate_model(model, tokenizer, eval_dataset, device, batch_size=8):
    """Run evaluation on test set"""
    logger.info("Running model evaluation...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    # Process dataset in batches
    for i in range(0, len(eval_dataset), batch_size):
        batch_data = eval_dataset[i:i + batch_size]
        
        # Convert batch data to tensors
        input_ids = torch.stack([torch.tensor(x) for x in batch_data['input_ids']]).to(device)
        attention_mask = torch.stack([torch.tensor(x) for x in batch_data['attention_mask']]).to(device)
        labels = torch.stack([torch.tensor(x) for x in batch_data['labels']]).to(device)
        
        # Create inputs dictionary
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        
        # Store predictions and labels
        all_predictions.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    metrics = compute_metrics((all_predictions, all_labels), tokenizer)
    
    # Log results
    logger.info("\nEvaluation Results:")
    logger.info(f"BLEU Score: {metrics['bleu']:.4f}")
    logger.info(f"ROUGE-1 Score: {metrics['rouge1']:.4f}")
    logger.info(f"ROUGE-2 Score: {metrics['rouge2']:.4f}")
    logger.info(f"ROUGE-L Score: {metrics['rougeL']:.4f}")
    
    return metrics

class PausableTrainer(Trainer):
    """Custom trainer that pauses halfway through training"""
    
    def train(
        self,
        resume_from_checkpoint = None,
        trial = None,
        ignore_keys_for_eval = None,
        **kwargs,
    ):
        """Override train to add pause"""
        # Calculate total steps
        total_steps = int(self.args.num_train_epochs * len(self.train_dataset) / (self.args.train_batch_size * self.args.gradient_accumulation_steps))
        halfway_point = total_steps // 2
        
        logger.info(f"Training will pause for 20 minutes at step {halfway_point} (halfway point)")
        
        # Start training
        train_result = super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs,
        )
        
        # Check if we're at halfway point
        if self.state.global_step >= halfway_point:
            pause_time = 20 * 60  # 20 minutes in seconds
            logger.info(f"\n\nPausing training for {pause_time//60} minutes at step {self.state.global_step}")
            logger.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("Training will resume automatically...\n")
            
            # Pause
            time.sleep(pause_time)
            
            logger.info(f"\nResuming training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return train_result

def train(
    model_name: str = "facebook/blenderbot-1B-distill",
    dataset_path: str = "finetune_data/finance_training_data.json",
    output_dir: str = "results/financial-bot-qlora",
    max_length: int = 128,
    batch_size: int = 8, 
):
    try:
        # Check GPU and clear memory
        device = check_gpu()
        
        # Load and process data
        logger.info("Loading dataset...")
        parent_dir = Path(__file__).resolve().parent.parent.parent
        dataset_path = parent_dir / dataset_path
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Properly format data for Dataset creation
        processed_data = []
        for item in data:
            # Handle list or string inputs
            free_messages = item['free_messages']
            guided_messages = item['guided_messages']
            
            # Convert to string if list
            if isinstance(free_messages, list):
                free_messages = free_messages[0] if free_messages else ""
            if isinstance(guided_messages, list):
                guided_messages = guided_messages[0] if guided_messages else ""
                
            # Convert personas list to string
            personas = ' | '.join(item['personas']) if isinstance(item['personas'], list) else str(item['personas'])
            
            # Convert previous utterances to string
            prev_utterances = item.get('previous_utterance', [])
            if isinstance(prev_utterances, list):
                prev_utterances = ' | '.join(prev_utterances)
            
            # Create processed item with all fields as strings
            processed_item = {
                'input_text': str(free_messages),
                'target_text': str(guided_messages),
                'personas': personas,
                'context': str(item.get('context', '')),
                'additional_context': str(item.get('additional_context', '')),
                'previous_utterance': str(prev_utterances),
                'guided_chosen_suggestions': str(item.get('guided_chosen_suggestions', [''])[0])
            }
            
            # Only add if we have valid input and target text
            if processed_item['input_text'] and processed_item['target_text']:
                processed_data.append(processed_item)

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
            tokenizer.padding_side = "right" #fix weird padding issue with fp16

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

        # Update preprocessing function to include context
        def preprocess_function(examples):
            """Enhanced preprocessing function with context"""
            
            # Prepare inputs with context
            inputs = []
            for idx in range(len(examples['input_text'])):
                context_parts = []
                
                # Add personas if present
                if examples['personas'][idx]:
                    context_parts.append(f"Personas: {examples['personas'][idx]}")
                
                # Add domain context
                if examples['context'][idx]:
                    context_parts.append(f"Domain: {examples['context'][idx]}")
                
                # Add specific context
                if examples['additional_context'][idx]:
                    context_parts.append(f"Topic: {examples['additional_context'][idx]}")
                
                # Add previous conversation if any
                if examples['previous_utterance'][idx]:
                    context_parts.append(f"Previous: {examples['previous_utterance'][idx]}")
                
                # Combine context with user message
                context = " [SEP] ".join(context_parts) if context_parts else ""
                input_text = f"{context} [SEP] User: {examples['input_text'][idx]}" if context else f"User: {examples['input_text'][idx]}"
                inputs.append(input_text)
            
            # Prepare targets with assistant prefix
            targets = [
                f"Assistant: {text}" if not text.startswith("Assistant:") else text 
                for text in examples['target_text']
            ]
            
            # Tokenize inputs
            model_inputs = tokenizer(
                inputs,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                padding_side='right',
                return_tensors='pt'
            )
            
            # Tokenize targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    padding_side='right',
                    return_tensors='pt'
                )
            
            model_inputs['labels'] = labels['input_ids']
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

        # Updated training arguments for DeepSpeed
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs = 50,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,
            learning_rate=5e-6,
            weight_decay=0.01,
            warmup_ratio=0.05,
            logging_steps=100,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=155,
            save_steps=155,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            gradient_checkpointing=True,
            optim="adamw_torch",  # Changed from paged_adamw_32bit for DeepSpeed compatibility
            lr_scheduler_type="cosine_with_restarts",
            group_by_length=True,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
        )
        
        # Then add DeepSpeed config using the training args
        training_args.deepspeed = setup_deepspeed_config(training_args)

        # Create early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=3,        # Number of evaluations to wait for improvement
            early_stopping_threshold=0.01,    # Minimum change to qualify as an improvement
        )

        # Initialize trainer with our custom PausableTrainer
        trainer = PausableTrainer(
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
            callbacks=[early_stopping_callback],
        )

        # Train with error handling
        logger.info("Starting QLoRA training...")
        try:
            trainer.train()
            
            # Run evaluation after training
            logger.info("Training completed. Running evaluation metrics...")
            eval_metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                eval_dataset=val_dataset,
                device=device,
                batch_size=batch_size
            )
            
            # Save metrics
            metrics_path = Path(output_dir) / "evaluation_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(eval_metrics, f, indent=2)
            
            logger.info(f"Evaluation metrics saved to {metrics_path}")
            
        except Exception as e:
            logger.error("Training or evaluation failed", exc_info=True)
            raise

        # Save with error handling
        try:
            logger.info("Saving LoRA adapter...")
            full_output_dir = Path(__file__).resolve().parent.parent.parent / output_dir
            model.save_pretrained(full_output_dir / "final_adapter")
            tokenizer.save_pretrained(full_output_dir / "final_adapter")

            
            logger.info("Merging LoRA weights...")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(full_output_dir / "final_model")
            logger.info("Model saved successfully!")
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
    # Set environment variables for DeepSpeed
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MASTER_PORT"] = str(29500)  # Default DeepSpeed port
    
    try:
        train()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}", exc_info=True)
        raise

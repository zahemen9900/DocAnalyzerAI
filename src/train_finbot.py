import json
import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from transformers import BitsAndBytesConfig
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType,
)
import bitsandbytes as bnb
from dataclasses import dataclass
from typing import Dict, List, Optional
import evaluate
import numpy as np
import wandb
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FinancialDataConfig:
    max_source_length: int = 512
    max_target_length: int = 256
    train_file: str = "finetune_data/train.json"
    val_file: str = "finetune_data/val.json"
    
class FinancialDataset:
    def __init__(self, config: FinancialDataConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
    def load_and_process_data(self, filepath: str) -> Dataset:
        """Load and process the financial conversation data"""
        try:
            # Load the JSON data
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Convert to format expected by model
            processed_data = []
            for item in data:
                # Combine context information
                context = (
                    f"Personas: {' '.join(item['personas'])}\n"
                    f"Context: {item['additional_context']}\n"
                )
                
                # Add previous utterances if they exist
                if item['previous_utterance']:
                    context += f"Previous: {' | '.join(item['previous_utterance'])}\n"
                
                # Process each conversation turn
                for user_msg, bot_msg in zip(item['free_messages'], item['guided_messages']):
                    processed_data.append({
                        'context': context,
                        'input': user_msg,
                        'response': bot_msg
                    })
            
            # Convert to Dataset
            dataset = Dataset.from_list(processed_data)
            
            # Tokenize
            return dataset.map(
                self._tokenize_function,
                remove_columns=dataset.column_names,
                batch_size=1000,
                num_proc=4
            )
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise

    def _tokenize_function(self, examples):
        """Tokenize the inputs and targets"""
        # Combine context and input
        full_input = f"{examples['context']}\nUser: {examples['input']}\nAssistant:"
        
        model_inputs = self.tokenizer(
            full_input,
            max_length=self.config.max_source_length,
            padding="max_length",
            truncation=True,
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            examples['response'],
            max_length=self.config.max_target_length,
            padding="max_length",
            truncation=True,
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

def compute_metrics(eval_preds):
    """Compute evaluation metrics"""
    global tokenizer  # Access the global tokenizer
    rouge_score = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    
    predictions, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE scores
    rouge_results = rouge_score.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    # BERTScore
    bert_results = bertscore.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        lang="en"
    )
    
    # Combine metrics
    results = {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bertscore_f1": np.mean(bert_results["f1"])
    }
    
    return results

def train():
    global tokenizer
    wandb.init(project="financial-chatbot")
    
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Detect available devices
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = torch.device("cuda:0")  # Use first GPU by default
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        n_gpus = 0
        device = torch.device("cpu")
        logger.info("No GPU available, using CPU")
    
    # Load model with manual device placement
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": device} if n_gpus > 0 else None,  # Manual device mapping
    )
    
    # Move model to device if not using device_map
    if n_gpus == 0:
        model = model.to(device)
    
    # Print model device info
    logger.info(f"Model device: {next(model.parameters()).device}")
    
    # Prepare model for k-bit training with device awareness
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA with device awareness
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Verify model placement
    logger.info(f"Model parameters device after LoRA: {next(model.parameters()).device}")
    
    # Modified training arguments with device awareness
    training_args = Seq2SeqTrainingArguments(
        output_dir="checkpoints/financial-chatbot",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        warmup_ratio=0.03,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="bertscore_f1",
        greater_is_better=True,
        fp16=True if n_gpus > 0 else False,  # Only use fp16 if GPU available
        optim="adamw_torch",
        max_grad_norm=0.3,
        weight_decay=0.01,
        report_to="wandb",
        no_cuda=n_gpus == 0,  # Explicitly set no_cuda based on GPU availability
    )
    
    # Load and process data
    data_config = FinancialDataConfig()
    dataset_processor = FinancialDataset(data_config, tokenizer)
    
    train_dataset = dataset_processor.load_and_process_data(data_config.train_file)
    eval_dataset = dataset_processor.load_and_process_data(data_config.val_file)
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        ),
        compute_metrics=compute_metrics,
    )
    
    # Train model
    try:
        trainer.train()
        
        # Save final model
        trainer.save_model("models/financial-chatbot-final")
        
        # End wandb run
        wandb.finish()
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        wandb.finish()
        raise

if __name__ == "__main__": 
    train()

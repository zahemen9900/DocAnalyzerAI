import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer, 
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, DatasetDict
import evaluate
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_metrics(eval_preds):
    """Compute ROUGE and BLEU metrics"""
    rouge_score = evaluate.load("rouge")
    bleu_score = evaluate.load("bleu")
    
    predictions, labels = eval_preds
    # Decode predictions and labels
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE scores
    rouge_output = rouge_score.compute(predictions=predictions, references=labels)
    
    # Compute BLEU score
    bleu_output = bleu_score.compute(predictions=predictions, references=labels)
    
    return {
        'rouge1': rouge_output['rouge1'],
        'rouge2': rouge_output['rouge2'],
        'rougeL': rouge_output['rougeL'],
        'bleu': bleu_output['bleu']
    }

def load_dataset(filepath):
    """Load and preprocess the dataset with train/val split"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    processed_data = []
    for item in data:
        # Format context info
        context = (
            f"Background: Financial Analysis\n"
            f"Personas: {' | '.join(item['personas'])}\n"
        )
        
        if item['previous_utterance']:
            context += f"Previous: {' | '.join(item['previous_utterance'])}\n"
            
        for user_msg, bot_msg in zip(item['free_messages'], item['guided_messages']):
            bot_msg = bot_msg.replace("[UPDATE]", "").strip()
            # Truncate bot_msg to first 512 tokens for CPU testing
            bot_msg = ' '.join(bot_msg.split()[:512])
            
            processed_data.append({
                "input_text": f"{context}\nUser: {user_msg}",
                "target_text": f"Assistant: {bot_msg}"
            })
    
    # Create dataset from processed data
    full_dataset = Dataset.from_list(processed_data)
    
    # Split into train/validation sets (90/10 split)
    split_dataset = full_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    
    return DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })

def train():
    # Use smaller model for CPU
    model_name = "facebook/blenderbot-400M-distill"
    dataset_path = "finetune_data/train.json"
    output_dir = "results/financial-bot"
    
    logger.info("Loading dataset...")
    dataset = load_dataset(dataset_path)
    
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        
        model_inputs = tokenizer(
            inputs,
            max_length=128,  # Reduced for CPU
            padding="max_length",
            truncation=True,
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=128,  # Reduced for CPU
                padding="max_length",
                truncation=True,
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    logger.info("Processing datasets...")    
    processed_datasets = DatasetDict({
        'train': dataset['train'].map(
            preprocess_function,
            batched=True,
            batch_size=4,
            remove_columns=dataset['train'].column_names,
        ),
        'validation': dataset['validation'].map(
            preprocess_function,
            batched=True,
            batch_size=4,
            remove_columns=dataset['validation'].column_names,
        )
    })

    # Minimal training args for CPU
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,     # Changed to 2 epochs
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        save_total_limit=2,
        prediction_loss_only=False
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets['train'],
        eval_dataset=processed_datasets['validation'],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True
        ),
        compute_metrics=compute_metrics,
    )
    
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully")
        
        # Run final evaluation
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")
        
        # Save the model
        output_path = f"{output_dir}/final"
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train()

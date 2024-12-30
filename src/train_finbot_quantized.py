import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments
from datasets import Dataset
import json
import logging
import gc
import evaluate
import numpy as np
from torch.quantization import quantize_dynamic
import torch.nn as nn
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq
from torch.optim import AdamW
import copy
import re
from typing import Dict, List, Union
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_quantized.log')
    ]
)
logger = logging.getLogger(__name__)

class FinancialMetricsComputer:
    """Enhanced metrics computer for financial domain"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.meteor = evaluate.load("meteor")
        self.bertscore = evaluate.load("bertscore")
        
    def compute_domain_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute domain-specific metrics"""
        # Financial term accuracy
        financial_terms = ['stock', 'bond', 'market', 'investment', 'risk', 'return', 'portfolio']
        pred_terms = sum(1 for term in financial_terms if term in prediction.lower())
        ref_terms = sum(1 for term in financial_terms if term in reference.lower())
        financial_term_accuracy = pred_terms / max(ref_terms, 1)
        
        # Numerical consistency
        pred_numbers = len(re.findall(r'\d+(?:\.\d+)?%?', prediction))
        ref_numbers = len(re.findall(r'\d+(?:\.\d+)?%?', reference))
        numerical_consistency = 1.0 if pred_numbers == ref_numbers else 0.0
        
        return {
            'financial_term_accuracy': financial_term_accuracy,
            'numerical_consistency': numerical_consistency
        }
    
    def __call__(self, eval_preds):
        predictions, labels = eval_preds
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean predictions and labels
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Initialize metrics
        metrics = {}
        
        # Calculate ROUGE scores
        for pred, label in zip(decoded_preds, decoded_labels):
            rouge_scores = self.rouge_scorer.score(label, pred)
            for key, score in rouge_scores.items():
                metrics[f'{key}_f'] = metrics.get(f'{key}_f', 0) + score.fmeasure
        
        # Average ROUGE scores
        for key in list(metrics.keys()):
            metrics[key] /= len(decoded_preds)
        
        # Calculate METEOR score
        meteor_score = self.meteor.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )
        metrics['meteor'] = meteor_score['meteor']
        
        # Calculate BERTScore
        bertscore = self.bertscore.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            lang="en"
        )
        metrics['bertscore_f1'] = sum(bertscore['f1']) / len(bertscore['f1'])
        
        # Calculate domain-specific metrics
        domain_metrics = {
            'financial_term_accuracy': 0.0,
            'numerical_consistency': 0.0
        }
        
        for pred, label in zip(decoded_preds, decoded_labels):
            scores = self.compute_domain_metrics(pred, label)
            for key, value in scores.items():
                domain_metrics[key] += value
        
        # Average domain metrics
        for key in domain_metrics:
            domain_metrics[key] /= len(decoded_preds)
            metrics[key] = domain_metrics[key]
        
        return metrics

class EnhancedQuantizedTrainer(Seq2SeqTrainer):
    """Enhanced trainer with quantization and domain-specific features"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_metric = float('inf')
        self.metrics_history = []
    
    def quantize_model(self):
        """Apply dynamic quantization to the model"""
        logger.info("Starting model quantization...")
        try:
            # Create a copy of the model for quantization
            qmodel = copy.deepcopy(self.model)
            qmodel.eval()
            
            # Specify modules to quantize
            modules_to_quantize = {
                'decoder.layers': nn.Linear,
                'encoder.layers': nn.Linear,
                'shared': nn.Embedding
            }
            
            # Apply dynamic quantization
            qmodel_quantized = quantize_dynamic(
                qmodel,
                modules_to_quantize,
                dtype=torch.qint8
            )
            
            logger.info("Model quantization completed successfully")
            return qmodel_quantized
            
        except Exception as e:
            logger.error(f"Quantization failed: {str(e)}")
            raise

def train_quantized(
    model_name: str = "facebook/blenderbot-400M-distill",
    dataset_path: str = "finetune_data/train.json",
    output_dir: str = "results/financial-bot-quantized",
    max_length: int = 128,
    batch_size: int = 16,
    num_epochs: int = 5
):
    try:
        # Force CPU for quantization
        device = torch.device("cpu")
        logger.info(f"Using device: {device}")
        
        # Set quantization backend
        torch.backends.quantized.engine = 'fbgemm'
        
        # Load dataset
        logger.info("Loading dataset...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process data
        processed_data = []
        for item in data:
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
        
        # Create dataset
        dataset = Dataset.from_list(processed_data)
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        # Load model and tokenizer
        logger.info(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Set up tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Move model to CPU for quantization
        model = model.to(device)
        model.eval()
        
        # Define preprocessing function
        def preprocess_function(examples):
            model_inputs = tokenizer(
                examples["input_text"],
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors=None
            )
            
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples["target_text"],
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors=None
                )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Process datasets
        logger.info("Processing datasets...")
        train_dataset = dataset["train"].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        val_dataset = dataset["test"].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["test"].column_names
        )
        
        # Enhanced training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="bertscore_f1",  # Changed to BERTScore
            greater_is_better=True,
            fp16=False,
            dataloader_pin_memory=False,
            seed=42,
            generation_max_length=max_length,
            predict_with_generate=True,
            include_inputs_for_metrics=True,
            optim="adamw_torch",  # Use PyTorch's AdamW
            lr_scheduler_type="cosine",
            warmup_ratio=0.1
        )
        
        # Initialize metrics computer
        metrics_computer = FinancialMetricsComputer(tokenizer)
        
        # Initialize enhanced quantized trainer
        trainer = EnhancedQuantizedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                padding=True,
                pad_to_multiple_of=8
            ),
            compute_metrics=metrics_computer
        )
        
        # Apply quantization
        logger.info("Starting quantization process...")
        try:
            quantized_model = trainer.quantize_model()
            trainer.model = quantized_model
            logger.info("Quantization successful")
        except Exception as e:
            logger.error(f"Quantization failed, falling back to original model: {str(e)}")
        
        # Training
        logger.info("Starting training...")
        trainer.train()
        
        # Save model and configuration
        logger.info("Saving quantized model...")
        trainer.save_model(f"{output_dir}/final_quantized")
        tokenizer.save_pretrained(f"{output_dir}/final_quantized")
        
        with open(f"{output_dir}/quantization_config.json", 'w') as f:
            json.dump({
                'backend': 'fbgemm',
                'device': 'cpu',
                'quantized_modules': list(modules_to_quantize.keys())
            }, f)
        
        # Save metrics history
        metrics_path = f"{output_dir}/evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(trainer.metrics_history, f, indent=2)
        
        logger.info("Training completed with enhanced metrics!")
        
    except Exception as e:
        logger.error(f"Enhanced training failed: {str(e)}", exc_info=True)
        raise
    finally:
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    try:
        train_quantized()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}", exc_info=True)
        raise

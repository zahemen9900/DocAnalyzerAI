import torch
from transformers import (
    BlenderbotTokenizer, 
    BlenderbotForConditionalGeneration,
    Trainer, 
    TrainingArguments
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import load_dataset
import logging
from pathlib import Path
from huggingface_hub import notebook_login, HfApi
from torch.utils.tensorboard import SummaryWriter
import bitsandbytes as bnb
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset(data_dir: str):
    """Load and prepare the dataset for training"""
    # Load the processed datasets
    dataset = load_dataset(
        'json', 
        data_files={
            'train': f'{data_dir}/train.json',
            'validation': f'{data_dir}/val.json'
        }
    )
    
    return dataset

def prepare_qlora_model(model_name: str):
    """Prepare model for QLoRA training"""
    # Load model in 4-bit precision
    model = BlenderbotForConditionalGeneration.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # Alpha scaling
        target_modules=["q_proj", "v_proj"],  # Target attention layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    return model

def setup_tensorboard(output_dir: str):
    """Setup TensorBoard logging"""
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    return SummaryWriter(f'{output_dir}/runs/{current_time}')

def train_model(
    model_name: str = "facebook/blenderbot-400M-distill",
    data_dir: str = "../data/finetune",
    output_dir: str = "../models/financial_chatbot",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    push_to_hub: bool = True,
    hub_model_id: str = None
):
    """Fine-tune BlenderBot on financial data using QLoRA"""
    try:
        # Initialize tensorboard
        writer = setup_tensorboard(output_dir)
        
        # Load tokenizer and model
        tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        model = prepare_qlora_model(model_name)
        
        # Load dataset
        dataset = prepare_dataset(data_dir)
        
        def preprocess_function(examples):
            # Combine context and query
            inputs = [
                f"{ctx} {msg}" 
                for ctx, msg in zip(
                    examples['additional_context'], 
                    examples['free_messages']
                )
            ]
            targets = examples['guided_messages']
            
            model_inputs = tokenizer(
                inputs, 
                max_length=512,
                padding='max_length',
                truncation=True
            )
            
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=512,
                    padding='max_length',
                    truncation=True
                )
            
            model_inputs['labels'] = labels['input_ids']
            return model_inputs
        
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        
        # Setup training arguments with WandB integration
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            report_to=["tensorboard"],
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id
        )
        
        class CustomTrainer(Trainer):
            def log(self, logs: dict) -> None:
                """Custom logging to TensorBoard"""
                super().log(logs)
                if self.state.global_step % self.args.logging_steps == 0:
                    for key, value in logs.items():
                        writer.add_scalar(key, value, self.state.global_step)
        
        # Initialize trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_datasets['train'],
            eval_dataset=processed_datasets['validation'],
        )
        
        # Login to HuggingFace if pushing to hub
        if push_to_hub:
            notebook_login()
        
        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model and tokenizer
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Push to HuggingFace Hub if specified
        if push_to_hub and hub_model_id:
            logger.info(f"Pushing model to HuggingFace Hub: {hub_model_id}")
            api = HfApi()
            api.upload_folder(
                folder_path=output_dir,
                repo_id=hub_model_id,
                repo_type="model"
            )
        
        # Close tensorboard writer
        writer.close()
        logger.info(f"Model saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

def main():
    # Set your HuggingFace Hub model ID
    hub_model_id = "your-username/financial-chatbot"
    
    train_model(
        push_to_hub=True,
        hub_model_id=hub_model_id,
        num_epochs=5,  # Adjust as needed
        batch_size=2   # Reduced batch size for 4-bit training
    )

if __name__ == "__main__":
    main()

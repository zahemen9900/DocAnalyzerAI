import torch
from pathlib import Path
import logging
import gc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from optimum.exporters import TasksManager
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import onnxruntime as ort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_to_onnx(
    model_id: str = "facebook/blenderbot-3B",
    output_dir: str = "onnx_models/blenderbot-3B"
):
    """Export BlenderBot model to ONNX format"""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting {model_id} to ONNX format...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Export using ORTModelForSeq2SeqLM
        logger.info("Converting to ONNX format...")
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(
            model_id,
            export=True,
            provider="CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        )
        
        # Configure ONNX Runtime
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        if torch.cuda.is_available():
            sess_options.intra_op_num_threads = 1
        else:
            sess_options.intra_op_num_threads = 4
        
        # Save model and tokenizer
        logger.info("Saving ONNX model and tokenizer...")
        ort_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model exported successfully to {output_dir}")
        
        # Clean up
        del model, ort_model
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        raise

def main():
    try:
        # Export BlenderBot 3B
        export_to_onnx()
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise

if __name__ == "__main__":
    main()

# DocAnalyzerAI Financial Assistant

An AI-powered financial assistant trained on comprehensive financial domain knowledge using QLoRA fine-tuning. The model provides expert guidance on investments, market analysis, and financial planning.

## Features

- 🤖 **Advanced Language Model**: Fine-tuned on the BlenderBot base model using QLoRA for efficient adaptation
- 💼 **Financial Expertise**: Trained on diverse financial topics including:
  - Investment strategies and portfolio management
  - Market analysis and trading concepts
  - Personal finance and retirement planning
  - Risk management and diversification
  - Contemporary topics (DeFi, crypto, ESG investing)
- 🗣️ **Natural Conversations**: 
  - Dynamic multi-turn dialogues
  - Contextual followup questions
  - Professional yet accessible responses
- 📊 **Comprehensive Training Data**:
  - +30,000 curated financial Q&A pairs
  - Real-world market scenarios
  - Technical and fundamental concepts
- 🎯 **Optimized Performance**:
  - 4-bit quantization for efficient inference
  - LoRA adaptation for parameter-efficient fine-tuning
  - Enhanced token handling and response generation

## Installation

```bash
git clone https://github.com/yourusername/DocAnalyzerAI.git
cd DocAnalyzerAI
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Prepare training data:
```bash
python src/main/prepare_training_data.py
```

2. Fine-tune using QLoRA:
```bash
python src/main/train_finbot_qlora.py
```

### Running the Chatbot Interface

Launch the Gradio web interface:
```bash
python src/main/gradio_finance_app.py
```

Access the chatbot at `http://localhost:7867`

## Project Structure

```
DocAnalyzerAI/
├── src/
│   ├── main/
│   │   ├── prepare_training_data.py   # Training data generation
│   │   ├── train_finbot_qlora.py      # QLoRA fine-tuning
│   │   └── gradio_finance_app.py      # Web interface
│   └── other_trains/                   # Additional training scripts
├── finetune_data/                      # Generated training datasets
├── results/                            # Model checkpoints and outputs
└── requirements.txt                    # Dependencies
```

## Technical Details

### Model Architecture
- Base model: facebook/blenderbot-400M-distill
- Quantization: 4-bit precision using bitsandbytes
- Adaptation: LoRA with rank=16, alpha=32
- Training: QLoRA fine-tuning with cosine learning rate scheduling

### Training Data
- Structured Q&A pairs with variations
- Multi-turn conversations
- Dynamic response generation
- Professional financial context

### Performance Optimizations
- Gradient checkpointing
- Mixed precision training
- Efficient parameter updates
- Memory optimization

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace Transformers library
- QLoRA paper and implementation
- Financial domain experts and resources

## Disclaimer

This AI assistant is for educational and informational purposes only. Always consult qualified financial professionals for actual financial advice.


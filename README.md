# _**DocAnalyzerAI**_ - AI-Powered Financial Document Chatbot
---

## Project Overview

### Context
This project, developed as part of the GenAI consulting team at Boston Consulting Group (BCG), focuses on creating an AI-powered chatbot that analyzes financial documents. The initiative is at the intersection of finance and generative AI (GenAI), showcasing innovative applications of AI in the financial domain.

### Goals
1. **Data Extraction and Analysis**: Extract and analyze key financial data from 10-K and 10-Q documents.
2. **Chatbot Development**: Develop an interactive chatbot capable of:
   - Understanding and interpreting financial data.
   - Providing insights into financial trends.
   - Communicating complex financial information effectively.
3. **Strategic Consulting**: Leverage AI-driven insights to deliver strategic recommendations to clients.

### Deliverables
- **Processed Financial Data**: Key financial trends and insights extracted from client documents.
- **AI Chatbot**: A generative AI-powered chatbot designed for financial analysis and user interaction.
- **Strategic Recommendations**: Use insights to provide value-driven consulting solutions.

## Repository Structure
```
├── data/             # Financial documents and preprocessed datasets
├── models/           # Trained LLMs and model weights
├── notebooks/        # Jupyter notebooks for EDA and prototyping
├── src/              # Core source code for the chatbot and data processing
│   ├── data_prep/    # Data extraction and cleaning scripts
│   ├── chatbot/      # Chatbot architecture and logic
│   └── utils/        # Utility functions
├── tests/            # Unit tests and integration tests
├── docs/             # Documentation and project reports
├── .gitignore        # Files to be ignored by Git
├── requirements.txt  # Python dependencies
└── README.md         # Project description and usage
```

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/ai-financial-chatbot.git
   cd ai-financial-chatbot
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Preprocess Financial Data
Run the preprocessing pipeline to extract and clean data from 10-K and 10-Q documents:
```bash
python src/data_prep/preprocess.py --input data/raw --output data/processed
```

### 2. Train or Load the Model
Train the chatbot model or load pre-trained weights:
```bash
python src/chatbot/train.py --config configs/train_config.json
```

### 3. Start the Chatbot
Launch the chatbot interface:
```bash
python src/chatbot/app.py
```

## Key Features
- **Natural Language Processing (NLP)**: Leverages cutting-edge language models (e.g., GPT, LLaMA, or custom LLMs).
- **Interactive Interface**: Provides user-friendly interaction for financial data queries.
- **Financial Analytics**: Capable of identifying trends, anomalies, and insights from financial documents.

## Tech Stack
- **Languages**: Python
- **Libraries**: TensorFlow, PyTorch, Hugging Face Transformers, Pandas
- **Infrastructure**: AWS for CI/CD, Docker for containerization
- **Tools**: Power BI for visualization, SQL for database queries

## Contribution
Contributions are welcome! Please adhere to the following process:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit changes and open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- **Boston Consulting Group (BCG)**: For the opportunity to work on this exciting project.
- **GenAI Consulting Team**: For their guidance and collaboration.
- **OpenAI & Hugging Face**: For providing state-of-the-art NLP tools.

## Contact
For inquiries or further collaboration, contact David at david.bcgintern@bcg.com.


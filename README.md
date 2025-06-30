# Legal RAG QA System (CLI Only)

A Retrieval-Augmented Generation (RAG) pipeline for legal question answering, built with Python and designed to provide accurate, grounded answers based on legal case passages. This version is CLI-only and includes synthetic queries for performance measurement.

## Features

- **Semantic Search**: Uses sentence transformers and FAISS for efficient passage retrieval
- **AI-Powered Answers**: Generates contextual answers using OpenAI's GPT models
- **CLI Interface**: Simple command-line usage
- **Synthetic Queries**: Built-in example questions for easy testing and benchmarking
- **Legal Context**: Grounds answers in actual legal case passages and citations
- **Performance Metrics**: Reports retrieval and generation times

## Quick Start

### Prerequisites

- Python 3.8–3.10 (recommended for best compatibility)
- OpenAI API key (for answer generation)
- The following data files in your project directory:
  - `top_10000_data.csv` (extracted from `top_10000_data.csv.gz`)
  - `passage_dict.json`

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**:
   - Open (or create) a file named `.env` in your project directory.
   - Add your OpenAI API key (get it from https://platform.openai.com/account/api-keys):
     ```
     OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
     ```
   - Save the file.

4. **Ensure data files are in place**:
   - `top_10000_data.csv` (extracted from `top_10000_data.csv.gz`)
   - `passage_dict.json`

## Usage

### 1. Answer a Single Legal Question

```bash
python rag_qa.py ask --query "What is a violent crime as it pertains to career offender status?"
```

- You can change the query to any legal question you want.
- The system will print the answer, relevant passages, and performance metrics.

### 2. Run Synthetic Queries (Performance Demo)

```bash
python rag_qa.py synthetic
```

- This will run a batch of example legal questions and print answers, retrieval/generation times, and averages.

## Performance Metrics
- **Retrieval time**: Time taken to find the most relevant passages.
- **Generation time**: Time taken by the LLM to generate an answer.

## Troubleshooting

- **OpenAI API key error**: If you see an error about an invalid API key, make sure you have set your real key in the `.env` file.
- **Module not found**: Make sure you installed all dependencies in the same Python environment you are running the code from.
- **Python version**: For best compatibility, use Python 3.8–3.10. Some packages may not work with Python 3.12+ on Windows.

## Example Synthetic Queries
- What is a violent crime as it pertains to career offender status?
- How do courts define 'violent felony' under the ACCA?
- What constitutes a crime of violence for sentencing purposes?
- How do courts interpret the 'residual clause' of the ACCA?
- What is a common law marriage?

## File Structure

```
Thomson_challenge/
├── rag_qa.py              # Main RAG QA system (CLI only)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── top_10000_data.csv     # Legal case data (extracted)
├── passage_dict.json      # Legal passage text
├── .env                   # Environment variables (create this)
```

## License

This project is for educational and research purposes. Please ensure compliance with data usage agreements and API terms of service.

## Contact

For questions or collaboration opportunities, please reach out through the project repository. 
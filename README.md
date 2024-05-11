# Resume Evaluation Assistant

## Description

The Resume Evaluation Assistant is a Streamlit web application designed to help users improve their resumes by providing comprehensive feedback. It leverages Natural Language Processing (NLP) and Machine Learning (ML) techniques, including spaCy for keyword extraction and OpenAI's GPT for advanced feedback. Users can upload their resumes and input job descriptions to receive personalized suggestions for improvement.

## Features

- **Text Extraction**: Supports extracting text from PDF and DOCX resume files.
- **Keyword Extraction**: Utilizes spaCy's Named Entity Recognition (NER) to identify key terms related to skills, organizations, etc.
- **Semantic Similarity**: Calculates the semantic similarity between the resume text and the job description using the Sentence Transformer library.
- **AI-Driven Feedback**: Provides detailed feedback by comparing the resume against the job description through OpenAI's GPT model, offering scores and suggestions for enhancement.

## Installation

### Prerequisites

- Python 3.7+
- Pip

### Libraries Installation

Run the following command to install all required libraries:

```bash
pip install streamlit pandas spacy nltk sentence_transformers pdfplumber python-docx openai python-dotenv
```

### Additional Setup

- Download the necessary spaCy model with:

```bash
python -m spacy download en_core_web_sm
```

- Ensure you have an OpenAI API key and set it in a `.env` file as follows:

```env
OPENAI_API_KEY='your_api_key_here'
```

## Usage

To run the application:

1. Navigate to the project directory.
2. Run the application using Streamlit:

```bash
streamlit run main.py
```

3. Open your web browser and go to the provided localhost URL, typically `http://localhost:8501`.

## How It Works

1. **Choose Feature**: Select "Evaluate Resume" from the dropdown menu.
2. **Upload Resume**: Upload your resume in PDF or DOCX format.
3. **Enter Job Description**: Paste the relevant job description into the text area.
4. **Receive Feedback**: Get immediate feedback on how well your resume matches the job description, including AI-generated suggestions for improvement.

## Contributing

Contributions to the Resume Evaluation Assistant are welcome. Please ensure to follow best practices for code changes and pull requests.

## License

Specify your project's license here, e.g., MIT, GPL-3.0, etc.


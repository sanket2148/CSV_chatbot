import streamlit as st
import os
import re
import pandas as pd
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import spacy
from collections import Counter

# Additional Imports for PDF and DOCX extraction
import pdfplumber
from docx import Document
import io

# Enhanced Error Handling and User Feedback
@st.cache_data(show_spinner=False)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


# Enhanced Resume Text Extraction
def extract_text_from_resume(file):
    text = ""
    try:
        if file.type == "application/pdf":
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text()
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(io.BytesIO(file.read()))
            for para in doc.paragraphs:
                text += para.text
    except Exception as e:
        st.error(f"Error processing file: {e}")
    return text
def extract_keywords_ner(text):
    # Use Spacy's NER to enhance keyword extraction
    doc = nlp(text)
    keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
    entities = [ent.text for ent in doc.ents if ent.label_ in ['SKILL', 'ORG', 'PRODUCT', 'GPE', 'NORP', 'PERSON']]
    all_terms = keywords + entities
    return Counter(all_terms)
# Enhanced Security and Validation
def sanitize_input(user_input):
    return re.sub(r'[^a-zA-Z0-9\s.,?!]', '', user_input)

def validate_query(user_query):
    """Check if the user query contains allowed terms and no blocked terms."""
    # Example terms, adjust according to your requirements
    # Extended list of allowed terms based on the dataset's context
    allowed_terms = [
        "average", "mean", "median", "sum", "count", "max", "min", "maximum", "minimum",
        "standard deviation", "variance", "total", "compare", "difference", "percentage",
        "proportion", "increase", "decrease", "growth", "rate", "trend", "correlation",
        "how many", "number of", "quantity", "volume", "frequency", "distribution",
        "show me", "tell me", "give me", "list", "find", "search for", "detail", "report",
        "analysis", "analyze", "review", "summary", "summarize", "overview", "breakdown",
        "evaluate", "evaluation", "compare", "comparison", "contrast", "rate", "ranking",
        "rank", "position", "categorize", "category", "classify", "classification",
        "group", "grouping", "segment", "segmentation", "range", "scope", "scale",
        "pattern", "patterns", "predict", "prediction", "forecast", "forecasting",
        "estimate", "estimation", "calculate", "calculation", "compute", "computation",
        "identify", "determine", "detect", "recognition", "recognize", "reveal", "expose",
        "explore", "exploration", "investigate", "investigation", "query", "inquire",
        "assessment", "assess", "measure", "measurement", "dimension", "scope", "scale",
        "insight", "insights", "understand", "understanding", "learn", "learning",
        "visualize", "visualization", "chart", "graph", "diagram", "plot", "map", "mapping",
        "relationship", "relationships", "association", "associations", "link", "links"
    ]

    blocked_terms = [
        "delete", "drop", "alter", "inject", "script", "<script>", "exec", "execute",
        "create", "update", "remove", "truncate", "modify", "merge", "grant", "revoke",
        "link", "unlink", "shell", "import", "export", "login", "logout", "register",
        "administrator", "sysadmin", "cmd", "command", "bash", "powershell", "sh",
        "`", "~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "+", "=",
        "{", "}", "[", "]", "|", "\\", ":", ";", "\"", "'", "<", ">", ",", ".", "?",
        "/tmp", "/bin", "/etc", "/usr", "sudo", "root", "passwd", "password", "credentials",
        "auth", "authentication", "token", "api_key", "apikey", "session", "cookie",
        "hack", "exploit", "overflow", "buffer", "attack", "vulnerability", "security",
        "encode", "encrypt", "decrypt", "cipher", "decipher", "ssl", "tls", "https",
        "ftp", "sftp", "scp", "wget", "curl", "http", "javascript", "python", "ruby",
        "perl", "php", "java", ".exe", ".sh", ".bat", ".js", ".py", ".pl", ".php",
        ".jar", ".dll", ".so", ".tmp", ".log", ".ini", ".conf", ".cfg", "config",
        "setup", "install", "uninstall", "archive", "backup", "copy", "paste"
    ]


    if any(term in user_query.lower() for term in blocked_terms):
        return False
    return any(term in user_query.lower() for term in allowed_terms)

# Enhanced NLP and Feedback Mechanism
nlp = spacy.load("en_core_web_sm")

def provide_feedback(resume_text, job_desc_text):
    model = load_model()
    resume_keywords = extract_keywords_ner(resume_text)
    job_desc_keywords = extract_keywords_ner(job_desc_text)
    suggestions = []
    # Example feedback generation
    common_terms = resume_keywords & job_desc_keywords
    if 'Python' not in common_terms:
        suggestions.append("Consider highlighting your experience with Python.")
    # Advanced semantic similarity based feedback
    similarity_score = calculate_semantic_similarity(resume_text, job_desc_text, model)
    suggestions.append(f"Resume and job description similarity score: {similarity_score:.2f}%")
    return suggestions

def calculate_semantic_similarity(text1, text2, model):
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_scores.item() * 100

# Main Application Flow Adjustments
def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("OpenAI API key is not set.")
        return
    
    st.set_page_config(page_title="Career Assistant")
    st.title("Career Assistant")
    feature_choice = st.selectbox("Choose a feature:", ["Query CSV", "Evaluate Resume"])

    if feature_choice == "Query CSV":
        handle_csv_feature(OPENAI_API_KEY)
    elif feature_choice == "Evaluate Resume":
        handle_resume_feature()

# Placeholder for CSV handling feature (unchanged)
def handle_csv_feature(api_key):
    st.subheader("Query CSV Data")
    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file:
        user_question = st.text_input("Ask a question about your CSV:")
        if user_question:
            sanitized_question = sanitize_input(user_question)
            if not validate_query(sanitized_question):
                st.error("Your question contains terms or patterns that are not allowed. Please try again.")
            else:
                llm = OpenAI(api_key=api_key, temperature=0)
                agent = create_csv_agent(llm, csv_file, verbose=True)
                with st.spinner("Analyzing..."):
                    response = agent.run(sanitized_question)
                    st.write(response)

# Adjusted Resume Feature with Text Extraction
def handle_resume_feature():
    st.subheader("Evaluate Your Resume")
    resume_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])
    job_desc_text = st.text_area("Paste the job description here:")
    if resume_file and job_desc_text:
        resume_text = extract_text_from_resume(resume_file)
        feedback = provide_feedback(resume_text, job_desc_text)
        for item in feedback:
            st.write(item)
            
            
import os
from openai import OpenAI

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def analyze_resume_with_llm(resume_text, job_desc_text):
    """
    Uses an LLM to analyze the resume in the context of the job description.
    Returns feedback and suggestions for the resume based on OpenAI's GPT.
    """
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant who evaluates resumes."},
            {"role": "user", "content": f"Job description:\n{job_desc_text}\n\nResume:\n{resume_text}\n\nEvaluate how well the resume matches the job description. Provide a score out of 10 for how well the resume matches job requirements, and suggest improvements or missing keywords."},
        ]
        
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo",  # Adjust the model based on your requirements
        )

        # Extracting and returning the feedback from the response
        feedback = response.choices[0].message["content"]
        return feedback.strip()
    except Exception as e:
        # Handle any errors that occur during the API call
        print(f"LLM Feedback Generation Error: {e}")
        return None

# Example usage
resume_text = "Your resume text here..."
job_desc_text = "Your job description text here..."
feedback = analyze_resume_with_llm(resume_text, job_desc_text)
print(feedback)





def handle_resume_feature():
    st.subheader("Evaluate Your Resume")
    resume_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])
    job_desc_text = st.text_area("Paste the job description here:")
    
    if resume_file and job_desc_text:
        resume_text = extract_text_from_resume(resume_file)
        
        # Existing feedback mechanism
        feedback = provide_feedback(resume_text, job_desc_text)
        for item in feedback:
            st.write(item)
        
        # New: Generate and display LLM-based feedback
        llm_feedback = analyze_resume_with_llm(resume_text, job_desc_text)
        if llm_feedback:
            st.subheader("LLM-based Feedback")
            st.write(llm_feedback)


if __name__ == "__main__":
    main()

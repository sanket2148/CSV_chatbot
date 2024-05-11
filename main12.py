# from langchain_experimental.agents import create_csv_agent
# from langchain.llms import OpenAI
# from dotenv import load_dotenv
# import os
# import streamlit as st


# def main():
#     load_dotenv()

#     # Load the OpenAI API key from the environment variable
#     # if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "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":
#     #     print("OPENAI_API_KEY is not set")
#     #     exit(1)
#     # else:
#     #     print("OPENAI_API_KEY is set")

#     st.set_page_config(page_title="Ask your CSV")
#     st.header("Ask your CSV 📈")

#     csv_file = st.file_uploader("Upload a CSV file", type="csv")
#     if csv_file is not None:
#         llm =OpenAI(temperature=0)
#         agent = create_csv_agent(
#            llm  , csv_file, verbose=True)

#         user_question = st.text_input("Ask a question about your CSV: ")

#         if user_question is not None and user_question != "":
#             with st.spinner(text="In progress..."):
#                 st.write(agent.run(user_question))


# if __name__ == "__main__":
#     main()



from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import re
import streamlit as st

# Add a simple sanitization function
def sanitize_input(user_input):
    # Remove potentially dangerous characters or patterns
    sanitized = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', user_input)
    return sanitized

# Add a validation function for user queries
def validate_query(user_query):
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

    
    # Check if the query contains blocked terms
    if any(blocked_term in user_query.lower() for blocked_term in blocked_terms):
        return False

    # Check if the query contains at least one term from the allowed list
    if any(term in user_query.lower() for term in allowed_terms):
        return True
    return False


def main():
    load_dotenv()

    # Validate the OpenAI API key environment variable
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        st.stop()

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0)
        agent = create_csv_agent(llm, csv_file, verbose=True)

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question:
            # Sanitize the user's input
            sanitized_question = sanitize_input(user_question)
            
            # Validate the sanitized input
            if not validate_query(sanitized_question):
                st.error("Your question contains terms or patterns that are not allowed. Please try again.")
                st.stop()
            else:
                with st.spinner(text="In progress..."):
                    try:
                        response = agent.run(sanitized_question)
                        st.write(response)
                    except Exception as e:
                        st.error(f"An error occurred while processing your question: {e}")
                        st.stop()

if __name__ == "__main__":
    main()



from collections import Counter
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import numpy as np

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_keywords(text):
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return Counter(filtered_words)

def calculate_semantic_similarity(text1, text2):
    # Convert sentences to embeddings
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    
    # Calculate cosine similarity
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    
    return cosine_scores.item()

def calculate_ats_score(resume_text, job_desc_text):
    # Basic keyword match score
    resume_keywords = extract_keywords(resume_text)
    job_desc_keywords = extract_keywords(job_desc_text)
    common_terms = sum((resume_keywords & job_desc_keywords).values())
    total_terms = sum(job_desc_keywords.values())
    keyword_match_score = (common_terms / total_terms) * 100 if total_terms > 0 else 0
    
    # Advanced semantic similarity score
    semantic_similarity_score = calculate_semantic_similarity(resume_text, job_desc_text) * 100
    
    # Combine scores (adjust weighting as necessary)
    combined_score = (keyword_match_score * 0.5) + (semantic_similarity_score * 0.5)
    
    return combined_score

# Example usage
resume_text = "Your resume text goes here..."
job_desc_text = "Job description text goes here..."
ats_score = calculate_ats_score(resume_text, job_desc_text)
print(f"Advanced ATS Score: {ats_score:.2f}%")



def extract_requirements(text):
    """
    Extracts key requirements from the job description.
    This function can be expanded or modified to use NLP techniques
    for more advanced extraction, such as identifying skills, qualifications, and experiences.
    """
    doc = nlp(text)
    # Extracting nouns and verbs as potential skill indicators
    requirements = [token.lemma_.lower() for token in doc if not token.is_stop and token.pos_ in ['NOUN', 'VERB']]
    return Counter(requirements)

def provide_feedback(resume_text, job_desc_text):
    resume_keywords = extract_keywords_ner(resume_text)
    job_desc_requirements = extract_requirements(job_desc_text)
    
    # Finding missing requirements in the resume
    missing_requirements = job_desc_requirements - resume_keywords
    suggestions = []
    
    # Simplified feedback based on missing key requirements
    for requirement, _ in missing_requirements.items():
        suggestions.append(f"Consider highlighting your experience or skills related to '{requirement}'.")

    # This is a simple approach and can be expanded to include feedback on formatting, structure, etc.
    return suggestions

# Example usage
resume_text = "Your resume text goes here..."
job_desc_text = "The job description text goes here..."
feedback = provide_feedback(resume_text, job_desc_text)

print("Feedback for improving your resume:")
for suggestion in feedback:
    print(f"- {suggestion}")
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import spacy

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")  # Assuming English text; use an appropriate model for your needs
stop_words = set(stopwords.words('english'))
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_keywords_ner(text):
    # Use Spacy's NER to enhance keyword extraction
    doc = nlp(text)
    keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
    entities = [ent.text for ent in doc.ents if ent.label_ in ['SKILL', 'ORG', 'PRODUCT', 'GPE', 'NORP', 'PERSON']]
    all_terms = keywords + entities
    return Counter(all_terms)

def calculate_ats_score(resume_text, job_desc_text):
    resume_keywords = extract_keywords_ner(resume_text)
    job_desc_keywords = extract_keywords_ner(job_desc_text)
    common_terms = sum((resume_keywords & job_desc_keywords).values())
    total_terms = sum(job_desc_keywords.values())
    keyword_match_score = (common_terms / total_terms) * 100 if total_terms > 0 else 0
    semantic_similarity_score = calculate_semantic_similarity(resume_text, job_desc_text) * 100
    combined_score = (keyword_match_score * 0.4) + (semantic_similarity_score * 0.6)  # Adjusted weighting
    
    return combined_score, common_terms

def provide_feedback(resume_text, job_desc_text):
    _, common_terms = calculate_ats_score(resume_text, job_desc_text)
    suggestions = []
    
    # Example feedback generation (simplified)
    if 'Python' not in common_terms:
        suggestions.append("Consider highlighting your experience with Python.")
    
    # Add more feedback logic based on job_desc_text analysis and common_terms
    
    return suggestions

# Example usage
resume_text = "Your resume text goes here..."
job_desc_text = "Job description text goes here..."
ats_score, _ = calculate_ats_score(resume_text, job_desc_text)
feedback = provide_feedback(resume_text, job_desc_text)

print(f"Advanced ATS Score: {ats_score:.2f}%")
for suggestion in feedback:
    print(f"- {suggestion}")
from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Assuming you have the functions defined as discussed:
# - sanitize_input
# - validate_query
# - extract_keywords_ner
# - calculate_semantic_similarity
# - calculate_ats_score
# - provide_feedback

# Load environment variables and NLP models
load_dotenv()
stop_words = set(stopwords.words('english'))
model = SentenceTransformer('all-MiniLM-L6-v2')

def main():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        return

    st.set_page_config(page_title="Career Assistant")
    st.header("Career Assistant")

    # Feature selection
    feature_choice = st.radio("Choose a feature:", ("Query CSV", "Evaluate Resume"))

    if feature_choice == "Query CSV":
        handle_csv_feature(OPENAI_API_KEY)
    elif feature_choice == "Evaluate Resume":
        handle_resume_feature()

def handle_csv_feature(api_key):
    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        llm = OpenAI(api_key=api_key, temperature=0)
        agent = create_csv_agent(llm, csv_file, verbose=True)
        user_question = st.text_input("Ask a question about your CSV:")
        if user_question:
            sanitized_question = sanitize_input(user_question)
            if not validate_query(sanitized_question):
                st.error("Your question contains terms or patterns that are not allowed. Please try again.")
            else:
                with st.spinner("In progress..."):
                    response = agent.run(sanitized_question)
                    st.write(response)

def handle_resume_feature():
    resume_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])
    job_desc_text = st.text_area("Paste the job description here:")
    if resume_file and job_desc_text:
        resume_text = "Extracted resume text goes here..."  # Implement actual extraction
        ats_score = calculate_ats_score(resume_text, job_desc_text)
        feedback = provide_feedback(resume_text, job_desc_text)
        st.success(f"ATS Score: {ats_score:.2f}%")
        st.subheader("Suggestions for Improvement:")
        for suggestion in feedback:
            st.text(suggestion)

if __name__ == "__main__":
    main()
################################################33
#########################################33
##########################################3
##################################

import streamlit as st
import os
import re
import pandas as pd
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain_experimental.agents import create_csv_agent
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import spacy

# Download necessary NLTK data and load Spacy model
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

# Initialize global variables
stop_words = set(stopwords.words('english'))
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define utility functions
def sanitize_input(user_input):
    """Remove potentially dangerous characters or patterns from user input."""
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

def extract_keywords_ner(text):
    # Use Spacy's NER to enhance keyword extraction
    doc = nlp(text)
    keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
    entities = [ent.text for ent in doc.ents if ent.label_ in ['SKILL', 'ORG', 'PRODUCT', 'GPE', 'NORP', 'PERSON']]
    all_terms = keywords + entities
    return Counter(all_terms)

def calculate_semantic_similarity(text1, text2):
    # Convert sentences to embeddings
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    
    # Calculate cosine similarity
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    
    return cosine_scores.item()

def calculate_ats_score(resume_text, job_desc_text):
    resume_keywords = extract_keywords_ner(resume_text)
    job_desc_keywords = extract_keywords_ner(job_desc_text)
    common_terms = sum((resume_keywords & job_desc_keywords).values())
    total_terms = sum(job_desc_keywords.values())
    keyword_match_score = (common_terms / total_terms) * 100 if total_terms > 0 else 0
    semantic_similarity_score = calculate_semantic_similarity(resume_text, job_desc_text) * 100
    combined_score = (keyword_match_score * 0.4) + (semantic_similarity_score * 0.6)  # Adjusted weighting
    
    return combined_score, common_terms

def extract_requirements(text):
    """
    Extracts key requirements from the job description.
    This function can be expanded or modified to use NLP techniques
    for more advanced extraction, such as identifying skills, qualifications, and experiences.
    """
    doc = nlp(text)
    # Extracting nouns and verbs as potential skill indicators
    requirements = [token.lemma_.lower() for token in doc if not token.is_stop and token.pos_ in ['NOUN', 'VERB']]
    return Counter(requirements)

def provide_feedback(resume_text, job_desc_text):
    resume_keywords = extract_keywords_ner(resume_text)
    job_desc_requirements = extract_requirements(job_desc_text)
    
    # Finding missing requirements in the resume
    missing_requirements = job_desc_requirements - resume_keywords
    suggestions = []
    
    # Simplified feedback based on missing key requirements
    for requirement, _ in missing_requirements.items():
        suggestions.append(f"Consider highlighting your experience or skills related to '{requirement}'.")

    # This is a simple approach and can be expanded to include feedback on formatting, structure, etc.
    return suggestions

# Define Streamlit app structure
def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        return

    st.set_page_config(page_title="Career Assistant")
    st.title("Career Assistant")

    feature_choice = st.selectbox("Choose a feature:", ["Query CSV", "Evaluate Resume"])

    if feature_choice == "Query CSV":
        handle_csv_feature(OPENAI_API_KEY)
    elif feature_choice == "Evaluate Resume":
        handle_resume_feature()

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

def handle_resume_feature():
    st.subheader("Evaluate Your Resume")
    resume_text = st.text_area("Paste your resume text here:")
    job_desc_text = st.text_area("Paste the job description here:")
    if st.button("Evaluate"):
        if resume_text and job_desc_text:
            feedback = provide_feedback(resume_text, job_desc_text)
            for item in feedback:
                st.write(item)

if __name__ == "__main__":
    main()


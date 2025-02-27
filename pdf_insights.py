import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from heapq import nlargest
import streamlit as st

# Download required NLTK data
nltk.download('punk')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    """Preprocess the extracted text"""
    # Tokenize into sentences
    sentences = sent_tokenize(text)

    # Convert to lowercase and remove special characters
    clean_sentences = [s.lower() for s in sentences]
    clean_sentences = [''.join(c for c in s if c not in punctuation) for s in clean_sentences]

    return sentences, clean_sentences

def calculate_word_frequency(text):
    """Calculate word frequency excluding stopwords"""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())

    # Remove stopwords and punctuation
    word_freq = {}
    for word in words:
        if word not in stop_words and word not in punctuation:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1

    return word_freq

def score_sentences(sentences, clean_sentences, word_freq):
    """Score sentences based on word frequency"""
    sentence_scores = {}

    for i, sentence in enumerate(clean_sentences):
        words = word_tokenize(sentence)
        score = sum(word_freq.get(word, 0) for word in words)
        # Normalize by sentence length
        sentence_scores[sentences[i]] = score / max(len(words), 1)

    return sentence_scores

def extract_key_points(text, num_points=5):
    """Extract key points from text"""
    # Preprocess text
    sentences, clean_sentences = preprocess_text(text)

    # Calculate word frequency
    word_freq = calculate_word_frequency(text)

    # Score sentences
    sentence_scores = score_sentences(sentences, clean_sentences, word_freq)

    # Extract top sentences as key points
    key_points = nlargest(num_points, sentence_scores, key=sentence_scores.get)

    return key_points

# Streamlit UI
def main():
    st.title("PDF Key Points Extractor")
    st.write("Upload a PDF file to extract key points")

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Number of key points slider
        num_points = st.slider("Number of key points to extract", 3, 10, 5)

        if st.button("Extract Key Points"):
            try:
                # Extract text from PDF
                text = extract_text_from_pdf(uploaded_file)

                # Extract key points
                key_points = extract_key_points(text, num_points)

                # Display results
                st.subheader("Key Points:")
                for i, point in enumerate(key_points, 1):
                    st.write(f"{i}. {point}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
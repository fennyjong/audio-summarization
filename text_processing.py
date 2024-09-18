# text_processing.py
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def setup_nltk():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

setup_nltk()

def preprocess_text(text):
    text = text.lower()
    sentences = sent_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    preprocessed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
        preprocessed_sentences.append(' '.join(words))
    
    return preprocessed_sentences

def textrank_summarize(text, num_sentences=5):
    sentences = sent_tokenize(text)
    preprocessed_sentences = preprocess_text(text)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    scores = np.array([sum(similarity_matrix[i]) for i in range(len(sentences))])
    ranking = scores.argsort()[::-1]
    
    top_sentences = sorted(ranking[:num_sentences])
    summary = ' '.join([sentences[i] for i in top_sentences])
    
    return summary
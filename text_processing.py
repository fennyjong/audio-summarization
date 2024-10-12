import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import networkx as nx

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
    stop_words = set(stopwords.words('indonesian'))
    
    preprocessed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
        preprocessed_sentences.append(words)  # Store as list of words instead of string
    
    return sentences, preprocessed_sentences

def calculate_similarity(filtered_sentences):
    num_sentences = len(filtered_sentences)
    similarity_matrix = np.zeros((num_sentences, num_sentences))

    for i in range(num_sentences):
        for j in range(num_sentences):
            if i != j:
                intersection_count = len(set(filtered_sentences[i]) & set(filtered_sentences[j]))
                log_i = np.log(len(filtered_sentences[i])) if len(filtered_sentences[i]) > 0 else 1
                log_j = np.log(len(filtered_sentences[j])) if len(filtered_sentences[j]) > 0 else 1
                similarity_matrix[i][j] = intersection_count / (log_i + log_j)

    return similarity_matrix

def textrank_summarize(text, num_sentences=3):
    sentences, preprocessed_sentences = preprocess_text(text)
    similarity_matrix = calculate_similarity(preprocessed_sentences)

    # Step 4: Calculate WS(Si) with PageRank
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    # Step 5: Display and Save Results
    sentence_labels = [f"S{i+1}" for i in range(len(sentences))]
    similarity_df = pd.DataFrame(similarity_matrix, columns=sentence_labels, index=sentence_labels)
    
    print("Matriks Kemiripan (Similarity Matrix):")
    print(similarity_df)

    # Create DataFrame for scores
    ws_scores_df = pd.DataFrame(list(scores.items()), columns=['Sentence Index', 'WS Score'])
    ws_scores_df['Sentence Index'] = ws_scores_df['Sentence Index'].apply(lambda x: f"S{x+1}")
    
    # Sort by WS Score to get the top sentences
    ws_scores_df = ws_scores_df.sort_values(by='WS Score', ascending=False)
    top_sentences_indices = ws_scores_df.head(num_sentences)['Sentence Index'].str.extract('(\d+)')[0].astype(int) - 1
    
    # Sort the indices of the top sentences to maintain original order
    sorted_top_sentences_indices = sorted(top_sentences_indices)
    
    # Create summary from sorted top sentences
    summary = ' '.join([sentences[index] for index in sorted_top_sentences_indices])
    
    return summary

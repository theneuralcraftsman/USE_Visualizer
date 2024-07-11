import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import plotly.graph_objs as go
import pandas as pd

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

@st.cache_resource
def load_model():
    with st.spinner("Loading Universal Sentence Encoder..."):
        return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

embed = load_model()

def similarity_matrix(sentences):
    embeddings = embed(sentences)
    similarity_matrix = np.inner(embeddings, embeddings)
    return similarity_matrix

def visualize_similarity_matrix():
    st.header("Sentence Similarity Matrix")
    
    st.write("""
    This tool uses the Universal Sentence Encoder to compare the similarity between sentences.
    Similarity scores range from 0 to 1, where:
    - 1 (bright yellow) indicates identical or very similar meaning
    - 0 (dark blue) indicates completely unrelated sentences
    Higher scores (closer to 1, and yellower in the heatmap) suggest more similar meanings.
    """)
    
    default_sentences = (
        "The cat sat on the mat.\n"
        "A dog chased the cat.\n"
        "The mat was comfortable.\n"
        "Cats are independent animals.\n"
        "Dogs are loyal companions.\n"
        "The sun was shining brightly.\n"
        "It was a rainy day outside."
    )
    
    sentences = st.text_area("Enter sentences (one per line):", default_sentences).split('\n')
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    if len(sentences) < 2:
        st.warning("Please enter at least 2 sentences for comparison.")
        return
    
    sim_matrix = similarity_matrix(sentences)
    
    st.subheader("Similarity Heatmap")
    st.write("""
    This heatmap visualizes the similarity between all pairs of sentences.
    Lighter colors (yellow) indicate higher similarity, while darker colors (dark blue) represent lower similarity.
    The diagonal is always brightest yellow as each sentence is identical to itself (similarity of 1).
    Similarity scores range from 0 (dark blue, completely different) to 1 (bright yellow, identical meaning).
    """)
    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix, 
        x=sentences, 
        y=sentences, 
        colorscale='Viridis',
        zmin=0, zmax=1,
        colorbar=dict(title="Similarity Score")
    ))
    fig.update_layout(
        title='Sentence Similarity Heatmap',
        height=800,
        width=800,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig)
    
    st.subheader("Similarity Scores Table")
    st.write("""
    This table shows the exact similarity scores between all pairs of sentences.
    Scores closer to 1 indicate higher similarity.
    """)
    df = pd.DataFrame(sim_matrix, columns=sentences, index=sentences)
    st.dataframe(df)
    
    st.subheader("Most Similar Sentence Pairs")
    st.write("""
    Here are the top 5 most similar pairs of sentences, excluding comparisons of a sentence with itself.
    These pairs have the highest similarity scores, indicating the strongest semantic relationships.
    """)
    pairs = []
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            pairs.append((sentences[i], sentences[j], sim_matrix[i][j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    for pair in pairs[:5]:  # Show top 5 most similar pairs
        st.write(f"Similarity: {pair[2]:.4f}")
        st.write(f"1: {pair[0]}")
        st.write(f"2: {pair[1]}")
        st.write("---")
    
    st.subheader("Sentence Analysis")
    st.write("""
    Select a sentence to see which other sentences are most similar to it.
    This can help identify sentences with related meanings or themes.
    """)
    selected_sentence = st.selectbox("Select a sentence to analyze:", sentences)
    sentence_index = sentences.index(selected_sentence)
    similarities = sim_matrix[sentence_index]
    sorted_indices = np.argsort(similarities)[::-1]
    
    st.write("Most similar sentences:")
    for i in sorted_indices[1:4]:  # Skip the first one as it's the sentence itself
        st.write(f"{similarities[i]:.4f}: {sentences[i]}")

def main():
    st.title("Sentence Similarity Analyzer")
    st.write("""
    This tool uses the Universal Sentence Encoder to analyze and compare the semantic similarity between sentences.
    Enter your sentences below and explore how they relate to each other!
    """)
    visualize_similarity_matrix()

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import ollama
import io

nltk.download('punkt')

# Load SentenceTransformer model for semantic similarity
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to get response from Ollama
def get_ollama_response(prompt, model_name="llama3"):
    try:
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    except Exception as e:
        return f"Error: {e}"

# Evaluation metrics
def compute_bleu(reference, candidate):
    ref_tokens = nltk.word_tokenize(reference)
    cand_tokens = nltk.word_tokenize(candidate)
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)

def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, candidate)
    return score['rougeL'].fmeasure

def compute_semantic_similarity(ref, cand):
    embeddings = st_model.encode([ref, cand])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

def compute_exact_match(ref, cand):
    return int(ref.strip().lower() == cand.strip().lower())

# Evaluation function
def evaluate_responses(df):
    results = []
    for _, row in df.iterrows():
        prompt = row['Prompt']
        expected = row['Expected']
        predicted = row['Predicted']

        bleu = compute_bleu(expected, predicted)
        rouge = compute_rouge(expected, predicted)
        semantic = compute_semantic_similarity(expected, predicted)
        exact = compute_exact_match(expected, predicted)

        results.append({
            'Prompt': prompt,
            'Expected': expected,
            'Predicted': predicted,
            'BLEU': bleu,
            'ROUGE-L': rouge,
            'Semantic Similarity': semantic,
            'Exact Match': exact
        })
    return pd.DataFrame(results)

# Streamlit UI
st.set_page_config(page_title="LLM Response Evaluator - Ollama", layout="wide")
st.title("🔍 LLM Response Evaluation Dashboard (Ollama)")

uploaded_file = st.file_uploader("Upload CSV file with columns: Prompt, Expected", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)

    if 'Prompt' not in df_input.columns or 'Expected' not in df_input.columns:
        st.error("CSV must contain 'Prompt' and 'Expected' columns.")
    else:
        st.success("File uploaded successfully!")
        st.dataframe(df_input)

        if st.button("Generate LLM Responses via Ollama"):
            with st.spinner("Generating responses..."):
                df_input['Predicted'] = df_input['Prompt'].apply(get_ollama_response)
            st.success("LLM responses generated.")

        if 'Predicted' in df_input.columns:
            if st.button("Evaluate Responses"):
                df_result = evaluate_responses(df_input)
                st.success("Evaluation complete!")
                st.dataframe(df_result)

                # Plot Metrics
                st.subheader("📊 Metrics Visualization")
                fig = px.box(df_result, y=["BLEU", "ROUGE-L", "Semantic Similarity"], title="Evaluation Metrics")
                st.plotly_chart(fig, use_container_width=True)

                # Download results
                csv = df_result.to_csv(index=False).encode()
                st.download_button("Download Evaluation Results as CSV", data=csv, file_name="evaluation_results.csv", mime="text/csv")

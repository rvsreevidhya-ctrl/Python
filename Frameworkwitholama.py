import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import requests

# ----------------------------
# Ollama configuration
# ----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"  # You can change this to the model you’ve pulled with `ollama pull`

def query_ollama(prompt, model=OLLAMA_MODEL):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"Error querying Ollama: {str(e)}"

# ----------------------------
# Load and preprocess data
# ----------------------------
def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
    except:
        df = pd.read_csv(uploaded_file)
    return df

# ----------------------------
# Evaluation Functions
# ----------------------------
def evaluate_predictions(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

def generate_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    return df_cm

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🧠 LLM Evaluation Dashboard using Ollama")

uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(df)

    # Assumes the data has a 'question' and 'expected_answer' column
    if 'question' in df.columns and 'expected_answer' in df.columns:
        st.subheader("🤖 Generating Responses using Ollama...")
        responses = []

        with st.spinner("Generating responses..."):
            for q in df['question']:
                answer = query_ollama(q)
                responses.append(answer)

        df['ollama_response'] = responses

        st.subheader("📝 Responses")
        st.dataframe(df[['question', 'expected_answer', 'ollama_response']])

        # You can map or classify responses for metric comparison if categories exist
        if 'category' in df.columns:  # optional column for classification
            y_true = df['category']
            y_pred = df['ollama_response'].apply(lambda x: x.split()[0])  # simple label extractor

            acc, prec, rec, f1 = evaluate_predictions(y_true, y_pred)

            st.subheader("📊 Evaluation Metrics")
            st.metric("Accuracy", f"{acc:.2f}")
            st.metric("Precision", f"{prec:.2f}")
            st.metric("Recall", f"{rec:.2f}")
            st.metric("F1 Score", f"{f1:.2f}")

            st.subheader("🔍 Confusion Matrix")
            cm_df = generate_confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Data must contain 'question' and 'expected_answer' columns.")

import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import io

# Configure Gemini API
genai.configure(api_key="AIzaSyB3jT5RkcHW7YYsM30NiO42jcBlmhEg8iE")
model = genai.GenerativeModel("gemini-2.0-flash")

# ---- GEMINI EVALUATION FUNCTION ----
def evaluate_with_gemini(prediction, reference):
    prompt = f"""
You are evaluating a model's answer compared to the ground truth.

Model's Answer:
{prediction}

Ground Truth:
{reference}

Determine whether the model's answer is correct **based on semantic and factual alignment**, not just exact match. Reply with:
- `Correct` or `Incorrect`
- A short justification in 1-2 lines

Format:
Verdict: Correct/Incorrect
Reason: [your brief explanation]
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Basic parsing logic
        verdict = "incorrect"
        reason = "Unable to parse Gemini response."

        for line in text.lower().split("\n"):
            if "correct" in line:
                verdict = "correct" if "incorrect" not in line else "incorrect"
            if "reason" in line:
                reason = line.split(":", 1)[-1].strip().capitalize()

        return 1 if verdict == "correct" else 0, reason

    except Exception as e:
        st.error(f"Gemini evaluation error: {e}")
        return 0, "Gemini evaluation failed."


# ---- STREAMLIT APP ----
def main():
    st.set_page_config(page_title="Gemini LLM Evaluator", layout="wide")
    st.title("📊 Gemini LLM Evaluation Dashboard")
    st.markdown("Upload an Excel file with **`Sftresponse`** and **`actual value`** columns for evaluation.")

    uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        if 'Sftresponse' not in df.columns or 'actual value' not in df.columns:
            st.error("❌ The Excel file must contain columns named `Sftresponse` and `actual value`.")
            return

        with st.spinner("🤖 Evaluating responses using Gemini..."):
            gemini_results = df.apply(
                lambda row: evaluate_with_gemini(row['Sftresponse'], row['actual value']), axis=1
            )

            df['gemini_eval'] = gemini_results.apply(lambda x: x[0])
            df['gemini_explanation'] = gemini_results.apply(lambda x: x[1])

            # Ground truth: Exact string match
            df['actual_binary'] = df.apply(
                lambda row: 1 if row['Sftresponse'].strip().lower() == row['actual value'].strip().lower() else 0, axis=1
            )

        # ---- METRICS ----
        accuracy = accuracy_score(df['actual_binary'], df['gemini_eval'])
        precision = precision_score(df['actual_binary'], df['gemini_eval'], zero_division=0)
        recall = recall_score(df['actual_binary'], df['gemini_eval'], zero_division=0)
        f1 = f1_score(df['actual_binary'], df['gemini_eval'], zero_division=0)

        st.subheader("📈 Evaluation Metrics")
        st.metric("Accuracy", f"{accuracy:.2f}")
        st.metric("Precision", f"{precision:.2f}")
        st.metric("Recall", f"{recall:.2f}")
        st.metric("F1 Score", f"{f1:.2f}")

        # ---- VISUALIZATION ----
        st.subheader("📊 Metric Visualization")
        metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}
        fig, ax = plt.subplots()
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        st.pyplot(fig)

        # ---- EXPANDED TABLE ----
        st.subheader("🧾 Evaluation Table")
        st.dataframe(df[['Sftresponse', 'actual value', 'gemini_eval', 'gemini_explanation']])

        # ---- DOWNLOAD RESULTS ----
        st.subheader("⬇️ Download Evaluated Results")
        output = io.BytesIO()
        df.to_excel(output, index=False)
        st.download_button("Download Excel", output.getvalue(), file_name="evaluated_results.xlsx")


if __name__ == "__main__":
    main()

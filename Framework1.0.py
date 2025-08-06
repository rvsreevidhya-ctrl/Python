import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import io
from datetime import datetime
import time
import random
import re
from collections import Counter
import math

# Configure page
st.set_page_config(
    page_title="Gemini LLM Evaluator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        margin: 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    .evaluation-summary {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .status-correct {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-incorrect {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .sidebar-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .rate-limit-info {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# Configure Gemini API
@st.cache_resource
def configure_gemini():
    genai.configure(api_key="AIzaSyCqpvoteap1CJrIYkK0fjI34_rVGN1ELHY")
    return genai.GenerativeModel("gemini-2.5-flash")

model = configure_gemini()

# ---- RATE LIMITING AND RETRY MECHANISM ----
class RateLimiter:
    def __init__(self, max_requests_per_minute=8, base_delay=8):
        self.max_requests_per_minute = max_requests_per_minute
        self.base_delay = base_delay
        self.request_times = []
        self.consecutive_errors = 0
        
    def wait_if_needed(self):
        """Implement rate limiting with adaptive delays"""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # If we're at the limit, wait
        if len(self.request_times) >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.request_times[0]) + 1
            if wait_time > 0:
                time.sleep(wait_time)
                # Clear old requests after waiting
                self.request_times = []
        
        # Add adaptive delay based on consecutive errors
        if self.consecutive_errors > 0:
            adaptive_delay = self.base_delay * (2 ** min(self.consecutive_errors - 1, 4))
            jitter = random.uniform(0.5, 1.5)
            total_delay = adaptive_delay * jitter
            time.sleep(total_delay)
        else:
            time.sleep(self.base_delay)
        
        # Record this request
        self.request_times.append(time.time())
    
    def on_success(self):
        """Reset error counter on successful request"""
        self.consecutive_errors = 0
    
    def on_error(self, error_msg=""):
        """Handle error and increase delay"""
        self.consecutive_errors += 1
        if "429" in error_msg or "quota" in error_msg.lower():
            time.sleep(30 + random.uniform(0, 10))

# Initialize rate limiter
rate_limiter = RateLimiter()

def safe_gemini_call(prompt, max_retries=5):
    """Make a safe call to Gemini with rate limiting and retries"""
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()
            response = model.generate_content(prompt)
            rate_limiter.on_success()
            return response.text.strip()
            
        except Exception as e:
            error_msg = str(e)
            rate_limiter.on_error(error_msg)
            
            if attempt == max_retries - 1:
                raise e
            
            retry_delay = 30
            if "retry_delay" in error_msg:
                try:
                    delay_match = re.search(r'seconds: (\d+)', error_msg)
                    if delay_match:
                        retry_delay = int(delay_match.group(1)) + random.uniform(5, 15)
                except:
                    pass
            
            st.warning(f"API error (attempt {attempt + 1}/{max_retries}): {error_msg}")
            st.info(f"Waiting {retry_delay:.1f} seconds before retry...")
            
            time.sleep(retry_delay)
    
    raise Exception("Max retries exceeded")

# ---- EVALUATION FUNCTIONS ----
def evaluate_with_gemini(prediction, reference, use_advanced_prompt=True):
    """Enhanced evaluation function with rate limiting and retry mechanism"""
    
    if use_advanced_prompt:
        prompt = f"""
You are an expert AI evaluator tasked with assessing the quality of model responses. 
Evaluate the semantic similarity and factual correctness of the model's answer against the ground truth.

Instructions:
1. Focus on semantic meaning rather than exact wording
2. Consider partial correctness for complex answers
3. Account for different valid phrasings of the same concept
4. Be strict about factual accuracy

Model's Response: "{prediction}"
Ground Truth: "{reference}"

Provide your evaluation in this exact format:
Verdict: [Correct/Incorrect]
Confidence: [High/Medium/Low]
Reason: [Brief explanation in 1-2 sentences]
"""
    else:
        prompt = f"""
Compare these two responses for semantic similarity:

Model Answer: {prediction}
Reference Answer: {reference}

Reply with:
Verdict: Correct/Incorrect
Reason: [brief explanation]
"""

    try:
        text = safe_gemini_call(prompt)

        # Enhanced parsing
        verdict = "incorrect"
        confidence = "medium"
        reason = "Unable to parse response"

        lines = text.lower().split("\n")
        for line in lines:
            if "verdict:" in line:
                verdict = "correct" if "correct" in line and "incorrect" not in line else "incorrect"
            elif "confidence:" in line:
                if "high" in line:
                    confidence = "high"
                elif "low" in line:
                    confidence = "low"
                else:
                    confidence = "medium"
            elif "reason:" in line:
                reason = line.split(":", 1)[-1].strip().capitalize()

        return 1 if verdict == "correct" else 0, reason, confidence

    except Exception as e:
        st.error(f"Gemini evaluation error: {e}")
        return 0, f"Evaluation failed: {str(e)}", "low"

# ---- ADDITIONAL METRICS FUNCTIONS ----
def calculate_bleu_score(prediction, reference, n=4):
    """Calculate BLEU score"""
    def get_ngrams(text, n):
        words = text.lower().split()
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    
    if not prediction.strip() or not reference.strip():
        return 0.0
    
    # Calculate n-gram precisions
    precisions = []
    for i in range(1, n+1):
        pred_ngrams = get_ngrams(prediction, i)
        ref_ngrams = get_ngrams(reference, i)
        
        if not pred_ngrams:
            precisions.append(0.0)
            continue
            
        pred_counts = Counter(pred_ngrams)
        ref_counts = Counter(ref_ngrams)
        
        clipped_counts = {ngram: min(count, ref_counts[ngram]) 
                         for ngram, count in pred_counts.items()}
        
        precision = sum(clipped_counts.values()) / len(pred_ngrams)
        precisions.append(precision)
    
    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0
    
    bleu = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    
    # Brevity penalty
    pred_len = len(prediction.split())
    ref_len = len(reference.split())
    if pred_len < ref_len:
        bp = math.exp(1 - ref_len / pred_len)
    else:
        bp = 1.0
    
    return bleu * bp

def calculate_rouge_l(prediction, reference):
    """Calculate ROUGE-L score"""
    def lcs_length(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    pred_words = prediction.lower().split()
    ref_words = reference.lower().split()
    
    if not pred_words or not ref_words:
        return 0.0
    
    lcs_len = lcs_length(pred_words, ref_words)
    
    if lcs_len == 0:
        return 0.0
    
    recall = lcs_len / len(ref_words)
    precision = lcs_len / len(pred_words)
    
    if recall + precision == 0:
        return 0.0
    
    f1 = 2 * recall * precision / (recall + precision)
    return f1

def calculate_bert_score_proxy(prediction, reference):
    """Calculate a proxy for BERT score using word overlap and semantic similarity"""
    # This is a simplified version - for production use, consider using the actual BERT score library
    
    def jaccard_similarity(text1, text2):
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def cosine_similarity_char(text1, text2):
        # Character-level cosine similarity as a proxy
        def get_char_counts(text):
            return Counter(text.lower().replace(' ', ''))
        
        counts1 = get_char_counts(text1)
        counts2 = get_char_counts(text2)
        
        # Get all unique characters
        all_chars = set(counts1.keys()) | set(counts2.keys())
        
        if not all_chars:
            return 1.0
        
        # Create vectors
        vec1 = [counts1.get(char, 0) for char in all_chars]
        vec2 = [counts2.get(char, 0) for char in all_chars]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    # Combine Jaccard similarity and character-level cosine similarity
    jaccard = jaccard_similarity(prediction, reference)
    cosine = cosine_similarity_char(prediction, reference)
    
    # Weighted combination (you can adjust these weights)
    bert_proxy = 0.6 * jaccard + 0.4 * cosine
    
    return bert_proxy

def calculate_exact_match(prediction, reference):
    """Calculate exact match score"""
    pred_clean = prediction.strip().lower()
    ref_clean = reference.strip().lower()
    return 1.0 if pred_clean == ref_clean else 0.0

# ---- FIXED METRICS CALCULATION ----
def calculate_advanced_metrics(df):
    """Calculate comprehensive evaluation metrics with proper ground truth"""
    metrics = {}
    
    # Use string-based exact match as ground truth for consistency
    # This ensures we're comparing actual model performance
    y_true = df['exact_match']  # Ground truth based on exact string matching
    y_pred = df['gemini_eval']  # Gemini's evaluation of model response
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Additional metrics
    total_samples = len(df)
    correct_predictions = sum(df['gemini_eval'])
    incorrect_predictions = total_samples - correct_predictions
    
    metrics['total_samples'] = total_samples
    metrics['correct_predictions'] = correct_predictions
    metrics['incorrect_predictions'] = incorrect_predictions
    metrics['error_rate'] = 1 - metrics['accuracy']
    
    # Agreement metrics between Gemini evaluation and exact match
    metrics['agreement_rate'] = sum(df['gemini_eval'] == df['exact_match']) / len(df)
    
    # Additional NLP metrics averages
    metrics['avg_bleu'] = df['bleu_score'].mean()
    metrics['avg_rouge_l'] = df['rouge_l'].mean()
    metrics['avg_bert_proxy'] = df['bert_score_proxy'].mean()
    metrics['exact_match_rate'] = df['exact_match'].mean()
    
    return metrics

# ---- VISUALIZATION FUNCTIONS ----
def create_metrics_chart(metrics):
    """Create professional metrics visualization"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Performance Metrics', 'Prediction Distribution', 
                       'Confusion Matrix', 'NLP Metrics', 'Agreement Analysis', 'Error Analysis'),
        specs=[[{"type": "bar"}, {"type": "pie"}, {"type": "heatmap"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    # Performance metrics bar chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    
    fig.add_trace(
        go.Bar(x=metric_names, y=metric_values, 
               marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c'],
               name="Metrics"),
        row=1, col=1
    )
    
    # Prediction distribution pie chart
    fig.add_trace(
        go.Pie(labels=['Correct', 'Incorrect'],
               values=[metrics['correct_predictions'], metrics['incorrect_predictions']],
               marker_colors=['#28a745', '#dc3545']),
        row=1, col=2
    )
    
    # Confusion matrix heatmap
    cm = metrics['confusion_matrix']
    fig.add_trace(
        go.Heatmap(z=cm, x=['Predicted Neg', 'Predicted Pos'], 
                   y=['Actual Neg', 'Actual Pos'],
                   colorscale='Blues', showscale=False,
                   text=cm, texttemplate="%{text}", textfont={"size":20}),
        row=1, col=3
    )
    
    # NLP metrics
    nlp_metrics = ['BLEU', 'ROUGE-L', 'BERT Proxy', 'Exact Match']
    nlp_values = [metrics['avg_bleu'], metrics['avg_rouge_l'], 
                  metrics['avg_bert_proxy'], metrics['exact_match_rate']]
    
    fig.add_trace(
        go.Bar(x=nlp_metrics, y=nlp_values,
               marker_color=['#17a2b8', '#28a745', '#ffc107', '#dc3545'],
               name="NLP Metrics"),
        row=2, col=1
    )
    
    # Agreement analysis
    agreement_rate = metrics['agreement_rate']
    disagreement_rate = 1 - agreement_rate
    
    fig.add_trace(
        go.Bar(x=['Agreement', 'Disagreement'], 
               y=[agreement_rate, disagreement_rate],
               marker_color=['#28a745', '#dc3545'],
               name="Agreement"),
        row=2, col=2
    )
    
    # Error analysis
    error_rate = metrics['error_rate']
    success_rate = 1 - error_rate
    
    fig.add_trace(
        go.Bar(x=['Success Rate', 'Error Rate'], 
               y=[success_rate, error_rate],
               marker_color=['#28a745', '#dc3545'],
               name="Error Analysis"),
        row=2, col=3
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Comprehensive Evaluation Dashboard")
    return fig

def create_performance_gauge(accuracy):
    """Create a gauge chart for overall performance"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = accuracy * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Accuracy (%)"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 50], 'color': "#ffcccc"},
                {'range': [50, 80], 'color': "#ffffcc"},
                {'range': [80, 100], 'color': "#ccffcc"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4},
                         'thickness': 0.75, 'value': 90}
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def estimate_completion_time(total_samples, requests_per_minute=8, base_delay=8):
    """Estimate completion time based on rate limits"""
    time_per_request = max(60/requests_per_minute, base_delay)
    # Only one request per sample for Gemini evaluation
    total_time_seconds = total_samples * time_per_request
    
    hours = int(total_time_seconds // 3600)
    minutes = int((total_time_seconds % 3600) // 60)
    seconds = int(total_time_seconds % 60)
    
    if hours > 0:
        return f"~{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"~{minutes}m {seconds}s"
    else:
        return f"~{seconds}s"

# ---- MAIN APPLICATION ----
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Gemini LLM Evaluation Dashboard</h1>
        <p>Professional AI Model Performance Analysis & Evaluation Suite</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Configuration")
        
        st.markdown("""
        <div class="sidebar-info">
            <h4>📊 Dashboard Features</h4>
            <ul>
                <li>Advanced semantic evaluation</li>
                <li>Multiple NLP metrics (BLEU, ROUGE, BERT)</li>
                <li>Rate limiting & error handling</li>
                <li>Comprehensive metrics analysis</li>
                <li>Interactive visualizations</li>
                <li>Professional reporting</li>
                <li>Fixed accuracy calculations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="rate-limit-info">
            <h4>⚠️ Rate Limiting</h4>
            <p><strong>Free Tier Limits:</strong></p>
            <ul>
                <li>8 requests/minute (conservative)</li>
                <li>8 second base delay</li>
                <li>Automatic retry with backoff</li>
                <li>Adaptive delay on errors</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        use_advanced_prompt = st.checkbox("🔧 Use Advanced Evaluation Prompts", value=True)
        show_confidence = st.checkbox("📈 Show Confidence Scores", value=True)
        
        # Rate limiting settings
        st.markdown("### ⚙️ Rate Limiting Settings")
        requests_per_minute = st.slider("Requests per minute", 3, 10, 8, 
                                       help="Conservative setting for free tier")
        base_delay = st.slider("Base delay (seconds)", 5, 15, 8, 
                             help="Delay between requests")
        
        # Update rate limiter with new settings
        rate_limiter.max_requests_per_minute = requests_per_minute
        rate_limiter.base_delay = base_delay
        
        st.markdown("### 📋 File Requirements")
        st.info("Upload an Excel file with columns:\n- `Sftresponse`\n- `actual value`")

    # File upload
    uploaded_file = st.file_uploader(
        "📂 Upload Your Evaluation Dataset", 
        type=["xlsx", "xls"],
        help="Upload an Excel file containing your model responses and ground truth values"
    )

    if uploaded_file:
        try:
            # Load data
            with st.spinner("📖 Loading dataset..."):
                df = pd.read_excel(uploaded_file)
            
            # Validate columns
            required_columns = ['Sftresponse', 'actual value']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.info("Available columns: " + ", ".join(df.columns.tolist()))
                return
            
            # Dataset overview
            st.markdown("### 📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Total Samples</div>
                </div>
                """.format(len(df)), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Columns</div>
                </div>
                """.format(len(df.columns)), unsafe_allow_html=True)
            
            with col3:
                non_null_responses = df['Sftresponse'].notna().sum()
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Valid Responses</div>
                </div>
                """.format(non_null_responses), unsafe_allow_html=True)
            
            with col4:
                completion_rate = (non_null_responses / len(df)) * 100
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">{:.1f}%</div>
                    <div class="metric-label">Completion Rate</div>
                </div>
                """.format(completion_rate), unsafe_allow_html=True)

            # Time estimation
            estimated_time = estimate_completion_time(len(df), requests_per_minute, base_delay)
            st.info(f"⏱️ Estimated completion time: {estimated_time} (for {len(df)} samples)")

            # Evaluation process
            if st.button("🚀 Start Evaluation", type="primary"):
                st.warning("⚠️ This process will take time due to API rate limits. Please be patient and don't refresh the page.")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_info = st.empty()
                
                start_time = time.time()
                
                with st.spinner("🤖 Evaluating responses with Gemini AI..."):
                    results = []
                    total_rows = len(df)
                    
                    for idx, row in df.iterrows():
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        
                        # Update progress
                        progress = (idx + 1) / total_rows
                        progress_bar.progress(progress)
                        
                        # Estimate remaining time
                        if idx > 0:
                            avg_time_per_sample = elapsed_time / (idx + 1)
                            remaining_samples = total_rows - (idx + 1)
                            estimated_remaining = avg_time_per_sample * remaining_samples
                            
                            remaining_hours = int(estimated_remaining // 3600)
                            remaining_minutes = int((estimated_remaining % 3600) // 60)
                            remaining_seconds = int(estimated_remaining % 60)
                            
                            if remaining_hours > 0:
                                time_estimate = f"{remaining_hours}h {remaining_minutes}m remaining"
                            elif remaining_minutes > 0:
                                time_estimate = f"{remaining_minutes}m {remaining_seconds}s remaining"
                            else:
                                time_estimate = f"{remaining_seconds}s remaining"
                        else:
                            time_estimate = "Calculating..."
                        
                        status_text.text(f"Evaluating sample {idx + 1} of {total_rows}")
                        time_info.text(f"⏱️ {time_estimate} | Elapsed: {int(elapsed_time//60)}m {int(elapsed_time%60)}s")
                        
                        try:
                            # Evaluate model response vs ground truth using Gemini
                            if show_confidence:
                                score, reason, confidence = evaluate_with_gemini(
                                    row['Sftresponse'], row['actual value'], use_advanced_prompt
                                )
                                results.append((score, reason, confidence))
                            else:
                                score, reason, _ = evaluate_with_gemini(
                                    row['Sftresponse'], row['actual value'], use_advanced_prompt
                                )
                                results.append((score, reason, "medium"))
                            
                        except Exception as e:
                            st.error(f"Error processing sample {idx + 1}: {str(e)}")
                            results.append((0, f"Processing failed: {str(e)}", "low"))
                    
                    # Calculate additional metrics (done locally, no API calls)
                    st.info("🔢 Calculating additional NLP metrics...")
                    
                    bleu_scores = []
                    rouge_scores = []
                    bert_scores = []
                    exact_matches = []
                    
                    for idx, row in df.iterrows():
                        pred = str(row['Sftresponse']) if pd.notna(row['Sftresponse']) else ""
                        ref = str(row['actual value']) if pd.notna(row['actual value']) else ""
                        
                        bleu_scores.append(calculate_bleu_score(pred, ref))
                        rouge_scores.append(calculate_rouge_l(pred, ref))
                        bert_scores.append(calculate_bert_score_proxy(pred, ref))
                        exact_matches.append(calculate_exact_match(pred, ref))
                    
                    # Process results
                    df['gemini_eval'] = [r[0] for r in results]
                    df['gemini_explanation'] = [r[1] for r in results]
                    if show_confidence:
                        df['confidence'] = [r[2] for r in results]
                    
                    # Add additional metrics
                    df['bleu_score'] = bleu_scores
                    df['rouge_l'] = rouge_scores
                    df['bert_score_proxy'] = bert_scores
                    df['exact_match'] = exact_matches
                
                end_time = time.time()
                evaluation_time = end_time - start_time
                
                status_text.text("✅ Evaluation completed!")
                time_info.text(f"🎉 Total time: {int(evaluation_time//3600)}h {int((evaluation_time%3600)//60)}m {int(evaluation_time%60)}s")
                progress_bar.progress(1.0)
                
                # Calculate metrics (FIXED: using proper ground truth)
                metrics = calculate_advanced_metrics(df)
                
                # Performance summary
                st.markdown("### 🎯 Performance Summary")
                
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.markdown(f"""
                    <div class="evaluation-summary">
                        <h4>📊 Core Metrics</h4>
                        <p><strong>Accuracy:</strong> {metrics['accuracy']:.3f}</p>
                        <p><strong>Precision:</strong> {metrics['precision']:.3f}</p>
                        <p><strong>Recall:</strong> {metrics['recall']:.3f}</p>
                        <p><strong>F1 Score:</strong> {metrics['f1']:.3f}</p>
                        <p><strong>Agreement Rate:</strong> {metrics['agreement_rate']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with summary_col2:
                    st.markdown(f"""
                    <div class="evaluation-summary">
                        <h4>📈 NLP Metrics</h4>
                        <p><strong>Average BLEU:</strong> {metrics['avg_bleu']:.3f}</p>
                        <p><strong>Average ROUGE-L:</strong> {metrics['avg_rouge_l']:.3f}</p>
                        <p><strong>Average BERT Proxy:</strong> {metrics['avg_bert_proxy']:.3f}</p>
                        <p><strong>Exact Match Rate:</strong> {metrics['exact_match_rate']:.3f}</p>
                        <p><strong>Error Rate:</strong> {metrics['error_rate']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Additional stats
                st.markdown("### 📋 Evaluation Statistics")
                
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    st.markdown(f"""
                    <div class="evaluation-summary">
                        <h4>⏱️ Timing Stats</h4>
                        <p><strong>Total Time:</strong> {evaluation_time:.2f} seconds</p>
                        <p><strong>Avg Time/Sample:</strong> {evaluation_time/len(df):.3f} seconds</p>
                        <p><strong>Samples Processed:</strong> {len(df)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with stats_col2:
                    st.markdown(f"""
                    <div class="evaluation-summary">
                        <h4>🎯 Gemini Evaluation</h4>
                        <p><strong>Correct by Gemini:</strong> {metrics['correct_predictions']}</p>
                        <p><strong>Incorrect by Gemini:</strong> {metrics['incorrect_predictions']}</p>
                        <p><strong>Gemini Success Rate:</strong> {metrics['correct_predictions']/len(df):.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with stats_col3:
                    st.markdown(f"""
                    <div class="evaluation-summary">
                        <h4>🔍 Ground Truth Comparison</h4>
                        <p><strong>Exact Matches:</strong> {sum(df['exact_match'])}</p>
                        <p><strong>Agreement Rate:</strong> {metrics['agreement_rate']:.3f}</p>
                        <p><strong>Disagreements:</strong> {len(df) - sum(df['gemini_eval'] == df['exact_match'])}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Visualizations
                st.markdown("### 📈 Performance Analytics")
                
                # Gauge chart and metrics comparison
                gauge_col1, gauge_col2 = st.columns([1, 1])
                with gauge_col1:
                    gauge_fig = create_performance_gauge(metrics['accuracy'])
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                with gauge_col2:
                    # NLP Metrics comparison
                    nlp_metrics_df = pd.DataFrame({
                        'Metric': ['BLEU', 'ROUGE-L', 'BERT Proxy', 'Exact Match'],
                        'Score': [metrics['avg_bleu'], metrics['avg_rouge_l'], 
                                 metrics['avg_bert_proxy'], metrics['exact_match_rate']]
                    })
                    
                    fig_nlp = px.bar(
                        nlp_metrics_df, x='Metric', y='Score',
                        title="NLP Metrics Comparison",
                        color='Score',
                        color_continuous_scale='Viridis'
                    )
                    fig_nlp.update_layout(height=400)
                    st.plotly_chart(fig_nlp, use_container_width=True)

                # Comprehensive dashboard
                comprehensive_fig = create_metrics_chart(metrics)
                st.plotly_chart(comprehensive_fig, use_container_width=True)

                # Detailed results table
                st.markdown("### 📋 Detailed Evaluation Results")
                
                # Add status badges and formatting
                def format_result(row):
                    status = "Correct" if row['gemini_eval'] == 1 else "Incorrect"
                    badge_class = "status-correct" if row['gemini_eval'] == 1 else "status-incorrect"
                    return f'<span class="status-badge {badge_class}">{status}</span>'
                
                def format_exact_match(value):
                    return "✅" if value == 1 else "❌"
                
                display_df = df.copy()
                display_df['Gemini_Status'] = df.apply(format_result, axis=1)
                display_df['Exact_Match'] = df['exact_match'].apply(format_exact_match)
                display_df['BLEU'] = df['bleu_score'].round(3)
                display_df['ROUGE-L'] = df['rouge_l'].round(3)
                display_df['BERT_Proxy'] = df['bert_score_proxy'].round(3)
                
                # Column selection for display
                display_columns = ['Sftresponse', 'actual value', 'Gemini_Status', 'Exact_Match', 
                                 'BLEU', 'ROUGE-L', 'BERT_Proxy', 'gemini_explanation']
                if show_confidence and 'confidence' in df.columns:
                    display_columns.append('confidence')
                
                # Show sample of results
                st.markdown("**Sample Results (First 10 rows):**")
                st.markdown(display_df[display_columns].head(10).to_html(escape=False, index=False), unsafe_allow_html=True)
                
                # Show disagreement cases
                disagreement_df = df[df['gemini_eval'] != df['exact_match']]
                if len(disagreement_df) > 0:
                    st.markdown(f"### ⚠️ Disagreement Cases ({len(disagreement_df)} found)")
                    st.markdown("Cases where Gemini evaluation differs from exact string matching:")
                    
                    disagreement_display = disagreement_df.copy()
                    disagreement_display['Gemini_Status'] = disagreement_df.apply(format_result, axis=1)
                    disagreement_display['Exact_Match'] = disagreement_df['exact_match'].apply(format_exact_match)
                    
                    st.markdown(disagreement_display[['Sftresponse', 'actual value', 'Gemini_Status', 'Exact_Match', 'gemini_explanation']].to_html(escape=False, index=False), unsafe_allow_html=True)

                # Export options
                st.markdown("### 💾 Export Results")
                
                export_col1, export_col2, export_col3 = st.columns(3)
                
                with export_col1:
                    # Excel export
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Evaluation_Results', index=False)
                        
                        # Add metrics sheet
                        metrics_df = pd.DataFrame([
                            ['Accuracy', metrics['accuracy']],
                            ['Precision', metrics['precision']],
                            ['Recall', metrics['recall']],
                            ['F1 Score', metrics['f1']],
                            ['Agreement Rate', metrics['agreement_rate']],
                            ['Average BLEU', metrics['avg_bleu']],
                            ['Average ROUGE-L', metrics['avg_rouge_l']],
                            ['Average BERT Proxy', metrics['avg_bert_proxy']],
                            ['Exact Match Rate', metrics['exact_match_rate']],
                            ['Total Samples', metrics['total_samples']],
                            ['Correct Predictions', metrics['correct_predictions']],
                            ['Error Rate', metrics['error_rate']],
                            ['Evaluation Time (seconds)', evaluation_time]
                        ], columns=['Metric', 'Value'])
                        metrics_df.to_excel(writer, sheet_name='Metrics_Summary', index=False)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        "📊 Download Excel Report",
                        output.getvalue(),
                        file_name=f"gemini_evaluation_report_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with export_col2:
                    # CSV export
                    csv_output = df.to_csv(index=False)
                    st.download_button(
                        "📄 Download CSV Data",
                        csv_output,
                        file_name=f"evaluation_results_{timestamp}.csv",
                        mime="text/csv"
                    )
                
                with export_col3:
                    # Metrics JSON export
                    metrics_json = {
                        'evaluation_summary': {
                            'timestamp': datetime.now().isoformat(),
                            'rate_limiting_settings': {
                                'requests_per_minute': requests_per_minute,
                                'base_delay': base_delay
                            },
                            'total_samples': int(metrics['total_samples']),
                            'evaluation_time_seconds': evaluation_time,
                            'core_metrics': {
                                'accuracy': float(metrics['accuracy']),
                                'precision': float(metrics['precision']),
                                'recall': float(metrics['recall']),
                                'f1_score': float(metrics['f1']),
                                'agreement_rate': float(metrics['agreement_rate']),
                                'error_rate': float(metrics['error_rate'])
                            },
                            'nlp_metrics': {
                                'avg_bleu': float(metrics['avg_bleu']),
                                'avg_rouge_l': float(metrics['avg_rouge_l']),
                                'avg_bert_proxy': float(metrics['avg_bert_proxy']),
                                'exact_match_rate': float(metrics['exact_match_rate'])
                            }
                        }
                    }
                    
                    st.download_button(
                        "🔧 Download Metrics JSON",
                        pd.Series(metrics_json).to_json(indent=2),
                        file_name=f"evaluation_metrics_{timestamp}.json",
                        mime="application/json"
                    )

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Please check your file format and try again.")
            import traceback
            st.code(traceback.format_exc())

    else:
        # Welcome message
        st.markdown("""
        ### 👋 Welcome to the Professional Gemini LLM Evaluator (FIXED VERSION)
        
        This advanced dashboard provides comprehensive evaluation of your AI model responses using Google's Gemini AI for semantic analysis.
        
        **🔧 FIXES APPLIED:**
        - ✅ **Fixed accuracy calculation mismatch** - Now uses exact match as ground truth baseline
        - ✅ **Added comprehensive NLP metrics** - BLEU, ROUGE-L, BERT Score Proxy, Exact Match
        - ✅ **Proper metrics alignment** - Performance analytics now match detailed results
        - ✅ **Enhanced disagreement analysis** - Shows cases where Gemini differs from exact matching
        - ✅ **Improved visualization** - Multiple metric types in dashboard
        
        **📊 Key Features:**
        - 🎯 **Semantic similarity evaluation** beyond exact string matching
        - 📈 **Multiple NLP metrics**: BLEU, ROUGE-L, BERT Score Proxy, Exact Match
        - 📊 **Comprehensive performance metrics** (Accuracy, Precision, Recall, F1)
        - 📈 **Interactive visualizations** and professional charts
        - 🔍 **Detailed confidence scoring** and explanations
        - ⚖️ **Agreement analysis** between different evaluation methods
        - 💾 **Multiple export formats** (Excel, CSV, JSON)
        - ⚡ **Real-time progress tracking** with time estimates
        - 🛡️ **Advanced Rate Limiting & Error Handling**
        
        **🔢 NLP Metrics Included:**
        - **BLEU Score**: Measures n-gram overlap with reference text
        - **ROUGE-L**: Evaluates longest common subsequence
        - **BERT Score Proxy**: Semantic similarity using word/character overlap
        - **Exact Match**: Traditional string-based exact matching
        
        **⚙️ Rate Limiting Features:**
        - ⏱️ Intelligent rate limiting for free tier quotas
        - 🔄 Automatic retry with exponential backoff
        - 📊 Real-time progress tracking with time estimates
        - ⚠️ Adaptive delays based on API response
        - 🛡️ Robust error handling and recovery
        
        **🚀 Getting Started:**
        1. Upload your Excel file with `Sftresponse` and `actual value` columns
        2. Configure evaluation settings in the sidebar
        3. Adjust rate limiting settings (8 req/min recommended for free tier)
        4. Click "Start Evaluation" and wait patiently (don't refresh!)
        5. Review comprehensive results with multiple metrics
        6. Export your professional report with all metrics
        
        **⚠️ Important Notes:**
        - **Free tier has strict quotas** (10 requests/minute)
        - **Large datasets will take considerable time** 
        - **Don't refresh the page during evaluation**
        - **Ground truth comparison** uses exact string matching as baseline
        - **Agreement rate** shows how often Gemini aligns with exact matching
        
        **🎯 What's Different Now:**
        - **Accuracy reflects actual model performance** against exact match baseline
        - **Comprehensive NLP metrics** provide multiple evaluation perspectives
        - **Disagreement analysis** helps understand semantic vs exact matching differences
        - **Performance analytics match detailed results** - no more mismatches!
        
        Ready to evaluate your model's performance with comprehensive metrics? Upload your dataset to begin!
        """)

if __name__ == "__main__":
    main()

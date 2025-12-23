import streamlit as st
import time
import warnings
import os
from sklearn.exceptions import InconsistentVersionWarning
from predict import predict_news, explain_prediction # Import the new function
import visualize as viz  

# --- 0. CONFIG & SETUP ---
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.set_page_config(
    page_title="Veritas | Fake News Detector",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. THEME-AWARE CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    .stTextArea>div>div>textarea {
        background-color: var(--secondary-background-color); 
        color: var(--text-color) !important;
        border: 1px solid var(--primary-color);
        border-radius: 12px;
        font-size: 16px;
    }
    
    /* Buttons */
    div[data-testid="column"] button { height: 50px; width: 100%; }
    
    /* Analyze Button */
    div[data-testid="column"]:first-child button {
        background: linear-gradient(45deg, #FF416C 0%, #FF4B2B 100%);
        color: white; border: none; border-radius: 8px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    div[data-testid="column"]:first-child button:hover { transform: scale(1.02); }

    /* Reset Button */
    div[data-testid="column"]:last-child button {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--primary-color);
        border-radius: 8px;
        font-weight: 600;
    }
    div[data-testid="column"]:last-child button:hover { color: #FF4B2B; border-color: #FF4B2B; }

    .title-text {
        background: -webkit-linear-gradient(left, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3rem; text-align: center; margin-bottom: 0px;
    }
    h4 { text-align: center; font-weight: 600; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""

# --- 2. SIDEBAR ---
with st.sidebar:
    st.info("**Model:** Logistic Regression")
    st.info("**Technique:** TF-IDF (N-Grams)")
    st.info("**Accuracy:** 94.5%")
    st.write("---")
    st.caption("Developed by: Faiza & Asad")

# --- 3. MAIN INTERFACE ---
st.markdown("<div class='title-text'>Veritas AI</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: var(--text-color); opacity: 0.7; margin-bottom: 30px;'>Advanced Fake News Detection System</p>", unsafe_allow_html=True)

text = st.text_area(
    "Enter News Article:",
    value=st.session_state['input_text'],
    height=200,
    placeholder="Paste the news article text here to analyze...",
    label_visibility="collapsed"
)

st.write("###") 
col_btn1, col_btn2 = st.columns(2, gap="medium")

with col_btn1:
    analyze_btn = st.button("✨ ANALYZE CREDIBILITY", type="primary", use_container_width=True)

with col_btn2:
    if st.button("🗑️ RESET INPUT", use_container_width=True):
        st.session_state['input_text'] = ""
        st.rerun()

# --- 5. RESULTS SECTION ---
if analyze_btn:
    if text.strip() == "":
        st.warning("⚠️ Please enter text to analyze.")
    else:
        with st.spinner("🔍 Analyzing linguistic patterns..."):
            time.sleep(1) 
            label, confidence, sentiment = predict_news(text)
            # --- NEW: Get Top Words ---
            explanation_df = explain_prediction(text)
        
        st.write("---")
        
        tab1, tab2 = st.tabs(["🔍 Current Prediction Dashboard", "📊 Model Performance Metrics"])
        
        with tab1:
            # HERO
            if label == "REAL":
                st.markdown(f"<h1 style='text-align: center; color: #4CAF50; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; margin-bottom: 30px;'>✅ REAL NEWS</h1>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h1 style='text-align: center; color: #FF4B2B; border-bottom: 3px solid #FF4B2B; padding-bottom: 10px; margin-bottom: 30px;'>🚨 FAKE NEWS</h1>", unsafe_allow_html=True)

            # 3-COLUMN DASHBOARD
            col_left, col_mid, col_right = st.columns([1, 1, 1], gap="large")
            
            with col_left:
                st.markdown("<h4>Confidence Level</h4>", unsafe_allow_html=True)
                gauge_chart = viz.create_gauge_chart(confidence, label)
                st.plotly_chart(gauge_chart, use_container_width=True)

            with col_mid:
                st.markdown("<h4>Visual Verification</h4>", unsafe_allow_html=True)
                st.write("##") 
                if label == "REAL":
                    st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbW00bnJvMmQ2dG4zbm14Z2Q4Z2Q4Z2Q4Z2Q4Z2Q4Z2Q4/3o7abKhOpu0NwenH3O/giphy.gif", use_container_width=True)
                else:
                    st.image("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2JtdmpsYzE3aTFuNnV5eHZtNXVpZnh0bGZheWt1NW13ZDV1d2J2cSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/d2Z9y3eDoq2cxPpK/giphy.gif", use_container_width=True)

            with col_right:
                st.markdown("<h4>Emotional Context</h4>", unsafe_allow_html=True)
                st.write("##")
                if sentiment < -0.1:
                    sent_label, sent_color = "Negative 😠", "#FF4B2B"
                elif sentiment > 0.1:
                    sent_label, sent_color = "Positive 😃", "#4CAF50"
                else:
                    sent_label, sent_color = "Neutral 😐", "#AAAAAA"
                
                st.markdown(f"""
                <div style="text-align: center; border: 2px solid {sent_color}; border-radius: 10px; padding: 10px; margin-top: 10px;">
                    <h2 style="color: {sent_color}; margin:0;">{sent_label}</h2>
                    <p style="margin:0; opacity: 0.8;">Score: {sentiment:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

            # --- NEW: EXPLAINABILITY SECTION ---
            st.write("---")
            st.markdown("### 📢 Why this result? (Top Triggers)")
            
            c_exp1, c_exp2 = st.columns([1, 1])
            with c_exp1:
                 st.info("These specific words/phrases in your text carried the most weight in the model's decision.")
                 # Apply color styling to dataframe
                 def color_survived(val):
                     color = '#FF4B2B' if 'Fake' in val else '#4CAF50'
                     return f'color: {color}; font-weight: bold'
                 
                 if not explanation_df.empty:
                    st.dataframe(
                        explanation_df.style.map(color_survived, subset=['Contribution']),
                        use_container_width=True,
                        hide_index=True
                    )
                 else:
                    st.write("No specific trigger words found in vocabulary.")

            with c_exp2:
                st.markdown("#### ☁️ Content Word Cloud")
                wc_fig = viz.create_wordcloud(text)
                st.pyplot(wc_fig, use_container_width=True)

            # DOWNLOAD BUTTON
            st.write("###")
            _, col_dl, _ = st.columns([1, 2, 1])
            with col_dl:
                pdf_bytes = viz.generate_pdf_report(text, label, confidence)
                st.download_button(
                    label="📄 Download Detailed Analysis Report (PDF)",
                    data=bytes(pdf_bytes),
                    file_name="analysis_report.pdf",
                    mime="application/pdf",
                    type="secondary",
                    use_container_width=True
                )

        with tab2:
            st.markdown("### 📈 Training Data Analysis")
            st.info("These plots represent the dataset used to train the model.")
            
            if not os.path.exists("plot_confusion_matrix.png"):
                with st.spinner("Generating analytics..."):
                    viz.generate_training_plots()
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                if os.path.exists("plot_confusion_matrix.png"):
                    st.image("plot_confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
            with col_m2:
                if os.path.exists("plot_distribution.png"):
                    st.image("plot_distribution.png", caption="Class Distribution", use_container_width=True)
            
            st.write("---")
            if os.path.exists("plot_wordcloud_fake.png"):
                _, col_cw, _ = st.columns([1, 4, 1])
                with col_cw:
                    st.image("plot_wordcloud_fake.png", caption="Common Words in Fake News", use_container_width=True)
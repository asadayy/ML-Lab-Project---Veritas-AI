import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objects as go
from fpdf import FPDF
import os
import tempfile
import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# --- PART 1: App Helper Functions (Gauge, PDF) ---

def create_gauge_chart(confidence, label):
    """Generates an interactive Plotly Gauge Chart."""
    color = "#4CAF50" if label == "REAL" else "#FF4B2B"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Prediction: {label}", 'font': {'size': 24, 'color': color}},
        number = {'suffix': "%", 'font': {'color': color}}, 
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 255, 255, 0.1)'},
                {'range': [50, 100], 'color': 'rgba(255, 255, 255, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font={'color': "white"}, 
        margin=dict(l=20, r=20, t=50, b=20), 
        height=250
    )
    return fig

def create_wordcloud(text):
    """Generates a Matplotlib figure for the WordCloud."""
    if not text or not isinstance(text, str):
        text = "No Data Available"
        
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        stopwords=STOPWORDS,
        colormap='viridis'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 10)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, 'Veritas AI Analysis Report', 0, 0, 'R')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(text, label, confidence):
    """Generates a detailed, well-formatted PDF report."""
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    # --- 1. TITLE SECTION ---
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(33, 33, 33)
    pdf.cell(0, 10, txt="Credibility Assessment", ln=True, align='C')
    
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(100, 100, 100)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d at %H:%M:%S")
    pdf.cell(0, 10, txt=f"Generated on: {timestamp}", ln=True, align='C')
    pdf.ln(10)
    
    # --- 2. SUMMARY BOX ---
    pdf.set_fill_color(245, 245, 245)
    pdf.set_draw_color(220, 220, 220)
    # x, y, w, h
    pdf.rect(x=15, y=pdf.get_y(), w=180, h=35, style='FD')
    
    pdf.set_y(pdf.get_y() + 8) # Move inside box
    
    # Verdict
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(90, 8, txt="      Verdict:", border=0)
    
    pdf.set_font("Arial", 'B', 16)
    if label == "REAL":
        pdf.set_text_color(0, 128, 0) # Green
    else:
        pdf.set_text_color(200, 0, 0) # Red
    pdf.cell(0, 8, txt=f"{label} NEWS", ln=True)
    
    # Confidence
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(90, 8, txt="      Confidence:", border=0)
    
    pdf.set_font("Arial", '', 14)
    pdf.cell(0, 8, txt=f"{confidence}%", ln=True)
    
    pdf.ln(20) # Move past box
    
    # --- 3. WORD CLOUD ---
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(33, 33, 33)
    pdf.cell(0, 10, txt="Linguistic Analysis", ln=True, align='L')
    
    # Generate and save temp image
    fig = create_wordcloud(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig.savefig(tmpfile.name, format='png', bbox_inches='tight', dpi=100)
        tmp_img_path = tmpfile.name
    plt.close(fig)

    # Place image centered
    pdf.image(tmp_img_path, x=15, w=180)
    pdf.ln(5)
    
    # --- 4. ANALYZED TEXT ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Text Excerpt", ln=True, align='L')
    pdf.ln(2)
    
    # Body Text
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(60, 60, 60)
    
    # Clean text to prevent PDF errors
    safe_text = text[:3000] # Limit char count
    # Remove weird characters for basic PDF compatibility
    safe_text = safe_text.encode('latin-1', 'replace').decode('latin-1')
    
    pdf.multi_cell(0, 6, txt=safe_text + "...")
    
    # Output
    pdf_output = pdf.output(dest='S').encode('latin-1')
    os.remove(tmp_img_path)
    return pdf_output

# --- PART 2: Auto-Generation of Training Plots ---

def generate_training_plots():
    """Checks if plots exist. If not, loads data and generates them."""
    
    if os.path.exists("plot_confusion_matrix.png") and os.path.exists("plot_distribution.png"):
        return # Exit if already done

    print("⏳ Generating training plots (this happens once)...")
    
    try:
        fake = pd.read_csv("data/Fake.csv")
        real = pd.read_csv("data/True.csv")
    except FileNotFoundError:
        return "Error: Data files not found in 'data/' folder."

    fake["label"] = 0
    real["label"] = 1
    
    df = pd.concat([fake, real]).sample(frac=0.5, random_state=42).reset_index(drop=True)

    # Fill NaNs
    df['title'] = df['title'].fillna('') 

    # Split Data
    X = df['title']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Temp Model
    vec = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    
    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix (Model Accuracy: {acc:.2%})')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('plot_confusion_matrix.png')
    plt.close()

    # 3. Distribution Plot
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='label', data=df, palette='viridis')
    plt.title('Distribution of Real (1) vs Fake (0) News')
    plt.xticks([0, 1], ['Fake News', 'Real News'])
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('plot_distribution.png')
    plt.close()
    
    # 4. Fake Word Cloud
    fake_text = " ".join(df[df['label'] == 0]['title'].astype(str))
    
    wc_fake = WordCloud(
        width=800, 
        height=400, 
        background_color='black', 
        colormap='Reds',
        stopwords=STOPWORDS
    ).generate(fake_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc_fake, interpolation='bilinear')
    plt.axis('off')
    plt.title("Common Words in Fake News", color='white', pad=20)
    plt.savefig('plot_wordcloud_fake.png', facecolor='black')
    plt.close()
    
    print("✅ Plots generated successfully!")
    return None
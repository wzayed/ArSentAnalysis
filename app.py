import re

def preprocess_arabic_text(text):
    # Remove diacritics
    text = re.sub(r'[\u064B-\u0652]', '', text)
    # Remove punctuation and non-Arabic characters
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    # Normalize Arabic letters
    text = re.sub(r'\u0629', '\u0647', text)  # Replace Teh Marbuta with Heh
    text = re.sub(r'\u064A', '\u0649', text)  # Replace Yeh with Alef Maqsura

      # Remove diacritics (optional, depending on use case)
    text = re.sub(r'[\u064B-\u065F]', '', text)
    
    # Normalize elongated letters (e.g., "جدااا" -> "جدا")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Remove non-Arabic characters (e.g., English words, numbers, special symbols)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model
model_name = 'aubmindlab/bert-base-arabertv02'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_map[predicted_class]

import gradio as gr

def process_text_and_analyze_sentiment(text):
    preprocessed_text = preprocess_arabic_text(text)
    sentiment = analyze_sentiment(preprocessed_text)
    return preprocessed_text, sentiment

# Create the Gradio interface
iface = gr.Interface(
    fn=process_text_and_analyze_sentiment,
    inputs=gr.Textbox(label="Enter Arabic Text"),
    outputs=[
        gr.Textbox(label="Preprocessed Text"),
        gr.Textbox(label="Sentiment")
    ],
    title="Arabic Text Analysis",
    description="This application preprocesses Arabic text using regex and analyzes sentiment using a pre-trained model."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True)


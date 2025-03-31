# Arabic Sentiment Analysis with Regex Preprocessing and AraBERT
# ArSentAnalysis
Arabic Sentiment Analysis using AraBert LLM


This repository provides a Python implementation for preprocessing Arabic text using regular expressions (regex) and performing sentiment analysis using the pre-trained **AraBERT** model. The solution is designed to handle common challenges in Arabic text, such as diacritics, elongated letters, and mixed content, while delivering accurate sentiment predictions.

---

## **Effectiveness Through Fine-Tuning**
To ensure the program is effective for your specific use case, it is recommended to:
- **Train or Fine-Tune the Model**: Adapt the pre-trained model to your organization's dataset to capture domain-specific nuances.
- **Use RAFT**: Incorporate Retrieval-Augmented Fine-Tuning (RAFT) to leverage external knowledge sources during training.
- **Evaluate on Real Data**: Test the model on real-world examples from your domain to measure performance and identify areas for improvement.

---

## **Contributing**
Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request.

---
---

## **Model Details**
- **Pre-trained Model**: [AraBERT](https://huggingface.co/aubmindlab/bert-base-arabertv02)
- **Tokenizer**: Automatically handles Arabic text tokenization.
- **Sentiment Labels**: ["Negative", "Neutral", "Positive"]

### **Alternative Models**
While **AraBERT** is used in this implementation, other state-of-the-art Arabic sentiment analysis models can also be employed, including:
- **MARBERT**: A multilingual Arabic BERT model trained on both Arabic and English text.
- **CAMeL Tools**: A suite of tools for Arabic NLP, including sentiment analysis.
- **ARBERT**: Another variant of BERT specifically fine-tuned for Arabic tasks.
- **JABER**: A Joint Arabic-BERT model for various NLP tasks.

These models can be swapped in by updating the `model_name` parameter in the code.

---

## **Customization**
1. **Add More Preprocessing Steps**:
   - Modify the `preprocess_arabic_text` function to include additional regex patterns or transformations.
   
2. **Fine-Tune the Model**:
   - Fine-tune the sentiment analysis model on your organization's data to improve accuracy and relevance.
   - Use techniques like **RAFT (Retrieval-Augmented Fine-Tuning)** to adapt the model to domain-specific language and experiences.

3. **Extend Sentiment Classes**:
   - Update the `num_labels` parameter in `AutoModelForSequenceClassification` to support more sentiment categories.

---

## **Features**
1. **Regex-Based Text Preprocessing**:
   - Removes diacritics (optional).
   - Normalizes elongated letters (e.g., "جدااا" → "جدا").
   - Filters out non-Arabic characters (e.g., English words, emojis, special symbols).
   - Normalizes whitespace.

2. **Sentiment Analysis with AraBERT**:
   - Leverages the `transformers` library to load and use the pre-trained **AraBERT** model.
   - Predicts sentiment labels: **Negative**, **Neutral**, or **Positive**.
   - Outputs confidence scores for each sentiment class.

3. **Modular Design**:
   - Easily extendable for additional preprocessing steps or sentiment models.

---

## **Dependencies**
To run this code, ensure you have the following Python libraries installed:

- `re` (built-in)
- `transformers`
- `torch`

You can install the required libraries using `pip`:
```bash
pip install transformers torch
```

---

## **Usage**

### **Step 1: Clone the Repository**
Clone this repository to your local machine:
```bash
git clone https://github.com/wzayed/arabic-sentiment-analysis.git
cd arabic-sentiment-analysis
```

### **Step 2: Run the Code**
The code consists of two main functions:
1. `preprocess_arabic_text`: Cleans and normalizes Arabic text.
2. `predict_sentiment`: Predicts sentiment using the AraBERT model.

#### Example Usage
```python
# Input Arabic text
raw_text = "الخدمة كانت ممتازة جدااا! 😍 ولكن السعر مرتفع بعض الشيء."

# Step 1: Preprocess the text
cleaned_text = preprocess_arabic_text(raw_text)
print("Cleaned Text:", cleaned_text)

# Step 2: Predict sentiment
sentiment, confidence_scores = predict_sentiment(cleaned_text)
print(f"Sentiment: {sentiment}")
print(f"Confidence Scores: {confidence_scores}")
```

#### Expected Output
```
Cleaned Text: الخدمة كانت ممتازة جدا ولكن السعر مرتفع بعض الشيء
Sentiment: Positive
Confidence Scores: [0.1, 0.2, 0.7]
```

---

## **Code Explanation**

### **Regex Preprocessing**
The `preprocess_arabic_text` function applies the following transformations:
1. **Remove Diacritics**: Strips Arabic diacritical marks like fatha (َ), kasra (ِ), and damma (ُ).
2. **Normalize Elongation**: Reduces repeated characters to at most two repetitions (e.g., "جدااا" → "جدا").
3. **Filter Non-Arabic Characters**: Removes English letters, numbers, emojis, and special symbols.
4. **Normalize Whitespace**: Replaces multiple spaces with a single space and trims leading/trailing spaces.

### **Sentiment Prediction**
The `predict_sentiment` function uses the **AraBERT** model (`aubmindlab/bert-base-arabertv02`) to classify Arabic text into one of three sentiment categories:
- **Negative**
- **Neutral**
- **Positive**

The function also outputs confidence scores for each sentiment class.



## **License**
This project is licensed under the MIT License. See the licence file for details.

---

## **Contact**
For questions or collaboration opportunities, please reach out to me via:
- Email: wmzayed@gmail.com
- LinkedIn: (https://www.linkedin.com/in/walid-zayed)

---

Thank you for using this repository! I hope it helps you build robust Arabic text analysis solutions. 😊

import sys
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

# Download NLTK stopwords if they're not already there
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# The necessary class with the cleaning and prediction logic
class BestSpamFilter:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def advanced_text_cleaning(self, text):
        text = str(text).lower()
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        text = re.sub(r'(\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-.\s]??\d{4}|\d{3}[-.\s]??\d{4})', '', text)
        text = re.sub(r'[$£€¥₹]\s*\d+[.,]?\d*', 'MONEY', text)
        text = re.sub(r'\d+[.,]?\d*\s*[$£€¥₹]', 'MONEY', text)
        text = re.sub(r'\b\d+\b', 'NUMBER', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        words = text.split()
        processed_words = [self.stemmer.stem(word) for word in words if len(word) >= 2 and word not in self.stop_words]
        return ' '.join(processed_words)

    def predict_message(self, message):
        try:
            with open('best_spam_model.pkl', 'rb') as f:
                model_package = pickle.load(f)
            
            model = model_package['model']
            vectorizer = model_package['vectorizer']
            
            cleaned_message = self.advanced_text_cleaning(message)
            message_vec = vectorizer.transform([cleaned_message])
            
            prediction = model.predict(message_vec)[0]
            probability = model.predict_proba(message_vec)[0]
            
            result = "SPAM" if prediction == 1 else "HAM"
            confidence = max(probability) * 100
            spam_prob = probability[1] * 100
            
            return result, confidence, spam_prob
            
        except FileNotFoundError:
            return "Error", "Model file not found. Make sure 'best_spam_model.pkl' is in the same folder.", None
        except Exception as e:
            return "Error", f"An error occurred: {e}", None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_spam.py \"Your message here\"")
        sys.exit(1)
        
    user_message = sys.argv[1]
    
    spam_filter = BestSpamFilter()
    result, confidence, spam_prob = spam_filter.predict_message(user_message)
    
    if result == "Error":
        print(f"Error: {confidence}")
        sys.exit(1)
    
    print("\n--- Result ---")
    print(f"Message: \"{user_message}\"")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Spam Probability: {spam_prob:.2f}%")
    print("--------------")

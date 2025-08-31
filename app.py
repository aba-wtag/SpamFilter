from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str

class BestSpamFilter:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def advanced_text_cleaning(self, text):
        text = str(text).lower()
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        text = re.sub(r'(\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-.\s]??\d{4}|\d{3}[-.\s]??\d{4})', '', text)
        text = re.sub(r'[$Â£â‚¬Â¥â‚¹]\s*\d+[.,]?\d*', 'MONEY', text)
        text = re.sub(r'\d+[.,]?\d*\s*[$Â£â‚¬Â¥â‚¹]', 'MONEY', text)
        text = re.sub(r'\b\d+\b', 'NUMBER', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        words = text.split()
        processed_words = [self.stemmer.stem(word) for word in words if len(word) >= 2 and word not in self.stop_words]
        return ' '.join(processed_words)

    def predict_message(self, message):
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

spam_filter = BestSpamFilter()

@app.post("/predict")
def predict_spam(request: MessageRequest):
    result, confidence, spam_prob = spam_filter.predict_message(request.message)
    
    return {
        "message": request.message,
        "prediction": result,
        "confidence": round(confidence, 2),
        "spam_probability": round(spam_prob, 2),
        "is_spam": result == "SPAM"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

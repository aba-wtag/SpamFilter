import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class BestSpamFilter:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.best_model = None
        self.vectorizer = None
        
    def advanced_text_cleaning(self, text):
        """Advanced text preprocessing for best performance"""
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove phone numbers
        text = re.sub(r'(\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-.\s]??\d{4}|\d{3}[-.\s]??\d{4})', '', text)
        
        # Remove currency symbols and amounts
        text = re.sub(r'[$£€¥₹]\s*\d+[.,]?\d*', 'MONEY', text)
        text = re.sub(r'\d+[.,]?\d*\s*[$£€¥₹]', 'MONEY', text)
        
        # Replace numbers with NUMBER token
        text = re.sub(r'\b\d+\b', 'NUMBER', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Split into words and process
        words = text.split()
        
        # Remove stopwords and apply stemming
        processed_words = []
        for word in words:
            if len(word) >= 2 and word not in self.stop_words:
                # Stem the word
                stemmed_word = self.stemmer.stem(word)
                processed_words.append(stemmed_word)
        
        # Join words back
        cleaned_text = ' '.join(processed_words)
        
        return cleaned_text
    
    def create_best_vectorizer(self):
        """Create optimized TF-IDF vectorizer"""
        return TfidfVectorizer(
            max_features=8000,           # Increased vocabulary
            min_df=2,                    # Ignore terms in less than 2 documents
            max_df=0.95,                 # Ignore terms in more than 95% of documents
            ngram_range=(1, 2),          # Use unigrams and bigrams
            stop_words='english',
            lowercase=True,
            sublinear_tf=True,           # Apply sublinear scaling
            smooth_idf=True,             # Smooth IDF weights
            norm='l2'                    # L2 normalization
        )
    
    def train_best_model(self, X_train, y_train):
        """Train and tune the best performing model"""
        print("Training and tuning models...")
        
        # Create individual models
        nb_model = MultinomialNB()
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Hyperparameter tuning for Naive Bayes
        nb_params = {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
        }
        
        # Hyperparameter tuning for Logistic Regression
        lr_params = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'solver': ['liblinear', 'lbfgs']
        }
        
        # Grid search with cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print("   Tuning Naive Bayes...")
        nb_grid = GridSearchCV(nb_model, nb_params, cv=cv, scoring='f1', n_jobs=-1)
        nb_grid.fit(X_train, y_train)
        best_nb = nb_grid.best_estimator_
        
        print("   Tuning Logistic Regression...")
        lr_grid = GridSearchCV(lr_model, lr_params, cv=cv, scoring='f1', n_jobs=-1)
        lr_grid.fit(X_train, y_train)
        best_lr = lr_grid.best_estimator_
        
        # Create ensemble model (Voting Classifier)
        print("   Creating ensemble model...")
        ensemble_model = VotingClassifier(
            estimators=[
                ('nb', best_nb),
                ('lr', best_lr)
            ],
            voting='soft'  # Use probability voting
        )
        
        # Train ensemble
        ensemble_model.fit(X_train, y_train)
        
        print(f"   Best Naive Bayes params: {nb_grid.best_params_}")
        print(f"   Best Logistic Regression params: {lr_grid.best_params_}")
        
        return ensemble_model, best_nb, best_lr
    
    def evaluate_models(self, models, model_names, X_test, y_test):
        """Evaluate all models and return the best one"""
        results = {}
        
        print("\nModel Evaluation:")
        print("-" * 60)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score'}")
        print("-" * 60)
        
        for model, name in zip(models, model_names):
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred
            }
            
            print(f"{name:<20} {accuracy:<10.4f} {precision:<12.4f} {recall:<10.4f} {f1:.4f}")
        
        # Find best model based on F1-score
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"🎯 Best F1-Score: {results[best_model_name]['f1_score']:.4f}")
        
        return best_model, best_model_name, results
    
    def save_model_and_vectorizer(self, model, vectorizer, model_name, performance_metrics):
        """Save the best model, vectorizer, and performance metrics"""
        model_package = {
            'model': model,
            'vectorizer': vectorizer,
            'model_name': model_name,
            'performance': performance_metrics,
            'text_cleaner': self.advanced_text_cleaning
        }
        
        # Save model
        with open('best_spam_model.pkl', 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"\n💾 Model saved as 'best_spam_model.pkl'")
        
        # Save performance report
        with open('performance_report.txt', 'w') as f:
            f.write("BEST SPAM FILTER PERFORMANCE REPORT\n")
            f.write("=" * 45 + "\n\n")
            f.write(f"Best Model: {model_name}\n")
            f.write(f"Accuracy:  {performance_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {performance_metrics['precision']:.4f}\n")
            f.write(f"Recall:    {performance_metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {performance_metrics['f1_score']:.4f}\n")
        
        print("📊 Performance report saved as 'performance_report.txt'")
    
    def predict_message(self, message, model_file='best_spam_model.pkl'):
        """Load model and predict a single message"""
        # Load model
        with open(model_file, 'rb') as f:
            model_package = pickle.load(f)
        
        model = model_package['model']
        vectorizer = model_package['vectorizer']
        
        # Clean and vectorize message
        cleaned_message = self.advanced_text_cleaning(message)
        message_vec = vectorizer.transform([cleaned_message])
        
        # Predict
        prediction = model.predict(message_vec)[0]
        probability = model.predict_proba(message_vec)[0]
        
        result = "SPAM" if prediction == 1 else "HAM"
        confidence = max(probability) * 100
        
        return result, confidence, probability[1] * 100

def main():
    spam_filter = BestSpamFilter()
    
    print("🚀 BEST PERFORMANCE SPAM FILTER PIPELINE")
    print("=" * 50)
    
    # 1. Load Data
    print("1. Loading dataset...")
    df = pd.read_csv('./data/data.csv')
    print(f"   📋 Dataset shape: {df.shape}")
    print(f"   📧 Ham messages: {sum(df['Category'] == 'ham')}")
    print(f"   ⚠️  Spam messages: {sum(df['Category'] == 'spam')}")
    
    # 2. Advanced Text Cleaning
    print("\n2. Advanced text preprocessing...")
    df['cleaned_message'] = df['Message'].apply(spam_filter.advanced_text_cleaning)
    df['label'] = df['Category'].map({'spam': 1, 'ham': 0})
    print("   ✨ Text cleaning with stemming and advanced preprocessing completed")
    
    # 3. Train-Test Split
    print("\n3. Splitting dataset...")
    X = df['cleaned_message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   📚 Training set: {len(X_train)} messages")
    print(f"   🧪 Test set: {len(X_test)} messages")
    
    # 4. Advanced Vectorization
    print("\n4. Advanced TF-IDF vectorization...")
    vectorizer = spam_filter.create_best_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"   🔢 Feature matrix: {X_train_vec.shape}")
    print(f"   📖 Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"   🔗 Using n-grams: {vectorizer.ngram_range}")
    
    # 5. Train Best Models
    print("\n5. Training optimized models...")
    ensemble_model, best_nb, best_lr = spam_filter.train_best_model(X_train_vec, y_train)
    
    # 6. Evaluate All Models
    models = [best_nb, best_lr, ensemble_model]
    model_names = ['Tuned Naive Bayes', 'Tuned Logistic Regression', 'Ensemble Model']
    
    best_model, best_model_name, results = spam_filter.evaluate_models(
        models, model_names, X_test_vec, y_test
    )
    
    # 7. Detailed Classification Report for Best Model
    print(f"\n📋 Detailed Report for {best_model_name}:")
    print("-" * 45)
    y_pred_best = best_model.predict(X_test_vec)
    print(classification_report(y_test, y_pred_best, target_names=['Ham', 'Spam']))
    
    # 8. Save Best Model
    best_performance = results[best_model_name]
    spam_filter.save_model_and_vectorizer(
        best_model, vectorizer, best_model_name, best_performance
    )
    
    # 9. Test with Real Examples
    print(f"\n🧪 Testing Best Model ({best_model_name}):")
    print("-" * 40)
    
    test_messages = [
        "Hi John, can we reschedule our meeting to tomorrow at 3pm?",
        "CONGRATULATIONS! You've WON $5000! Click here to claim your prize NOW!",
        "Thanks for sending the project files. I'll review them tonight.",
        "URGENT: Your bank account has been compromised! Verify immediately!",
        "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121",
        "Call me when you get this message. It's about the weekend plans.",
        "WINNER! You have been selected for a FREE vacation! Call 1800-FREE-NOW!",
        "The meeting has been moved to conference room B."
    ]
    
    for i, message in enumerate(test_messages, 1):
        result, confidence, spam_prob = spam_filter.predict_message(message)
        print(f"\n{i}. Message: {message}")
        print(f"   Prediction: {result} (Confidence: {confidence:.1f}%, Spam Prob: {spam_prob:.1f}%)")
    
    print(f"\n🎉 PIPELINE COMPLETED!")
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📈 Peak F1-Score: {best_performance['f1_score']:.4f}")
    print(f"🎯 Accuracy: {best_performance['accuracy']:.4f}")

if __name__ == "__main__":
    main()

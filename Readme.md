# Spam Filter

A beginner-friendly machine learning project to classify messages as **spam** or **ham** using traditional ML techniques, complete with a FastAPI backend and a simple client interface.

## Features

* **End-to-end ML pipeline:**
   * Text cleaning and preprocessing
   * Vectorization using TF-IDF
   * Model training with **Logistic Regression** and **Naive Bayes**
   * Hyperparameter tuning with **Grid Search**
   * Ensemble approach with **Voting Classifier** for best performance

* **Model evaluation with:**
   * Accuracy
   * Precision
   * Recall
   * F1-score

* Save and load trained model and vectorizer
* FastAPI backend exposing a prediction API
* Client interface to input a message and get spam/ham prediction

## Getting Started

### Prerequisites

* Python 3.9+
* `pip` for installing dependencies

Install required libraries:

```bash
pip install -r requirements.txt
```

### Running the FastAPI Server

```bash
uvicorn main:app --reload
```

The server will start at `http://127.0.0.1:8000`.

## API Usage

Send a POST request to `/predict` with a JSON payload:

```json
{
  "message": "Congratulations! You've won a free ticket!"
}
```

The response will return the prediction:

```json
{
  "prediction": "spam"
}
```

## Running the Client

* Use the provided HTML client to input a message
* Submit the message to the FastAPI endpoint
* Get real-time spam/ham classification

## Project Workflow

1. **Dataset**: Loaded from GitHub (SMS spam dataset)
2. **Preprocessing**: Clean and normalize text data
3. **Vectorization**: Convert text to numerical features using TF-IDF
4. **Model Training**: Train Logistic Regression and Naive Bayes
5. **Hyperparameter Tuning**: Optimize models with Grid Search
6. **Ensemble**: Voting Classifier to select the best-performing model
7. **Evaluation**: Accuracy, precision, recall, F1-score metrics
8. **Deployment**: Save model & vectorizer, serve via FastAPI
9. **Client Interaction**: Accept user input and display prediction

## Folder Structure

.
├── app.py
├── best_spam_model.pkl
├── client
│   └── spamfilter.html
├── data
│   └── data.csv
├── env
│   ├── bin
│   ├── include
│   ├── lib
│   ├── pyvenv.cfg
│   └── share
├── performance_report.txt
├── pipeline.py
├── predict_spam.py
├── Readme.md
└── requirements.txt

8 directories, 10 files

## License

This project is open-source and free to use under the MIT License# Spam Filter

A beginner-friendly machine learning project to classify messages as **spam** or **ham** using traditional ML techniques, complete with a FastAPI backend and a simple client interface.

## Features

* **End-to-end ML pipeline:**
   * Text cleaning and preprocessing
   * Vectorization using TF-IDF
   * Model training with **Logistic Regression** and **Naive Bayes**
   * Hyperparameter tuning with **Grid Search**
   * Ensemble approach with **Voting Classifier** for best performance

* **Model evaluation with:**
   * Accuracy
   * Precision
   * Recall
   * F1-score

* Save and load trained model and vectorizer
* FastAPI backend exposing a prediction API
* Client interface to input a message and get spam/ham prediction

## Getting Started

### Prerequisites

* Python 3.9+
* `pip` for installing dependencies

Install required libraries:

```bash
pip install -r requirements.txt
```

### Running the FastAPI Server

```bash
uvicorn main:app --reload
```

The server will start at `http://127.0.0.1:8000`.

## API Usage

Send a POST request to `/predict` with a JSON payload:

```json
{
  "message": "Congratulations! You've won a free ticket!"
}
```

The response will return the prediction:

```json
{
  "prediction": "spam"
}
```

## Running the Client

* Use the provided HTML client to input a message
* Submit the message to the FastAPI endpoint
* Get real-time spam/ham classification

## Project Workflow

1. **Dataset**: Loaded from GitHub (SMS spam dataset)
2. **Preprocessing**: Clean and normalize text data
3. **Vectorization**: Convert text to numerical features using TF-IDF
4. **Model Training**: Train Logistic Regression and Naive Bayes
5. **Hyperparameter Tuning**: Optimize models with Grid Search
6. **Ensemble**: Voting Classifier to select the best-performing model
7. **Evaluation**: Accuracy, precision, recall, F1-score metrics
8. **Deployment**: Save model & vectorizer, serve via FastAPI
9. **Client Interaction**: Accept user input and display prediction

## Folder Structure

```
spam-filter/
│
├── data/                  # Raw and processed datasets
├── models/                # Saved trained models and vectorizers
├── app/
│   ├── main.py            # FastAPI server
│   └── client.html        # Simple client interface
├── notebooks/             # Jupyter notebooks for training & evaluation
├── requirements.txt       # Python dependencies
└── README.md
```

## License

This project is open-source and free to use under the MIT License..
# SpamFilter

## How to run

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### To run train the model

```
python pipeline.py
```

### To run the server

```
python app.py
```

### To run the client just open the `./client.spamfilter.html` in any browser

### To run the CLI

```
python predict_spam.py {your message}
```


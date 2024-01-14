# Fake News Detection

This project focuses on building a Fake News Detection system using natural language processing (NLP) and machine learning techniques. The model is trained on a dataset containing news articles and utilizes a Decision Tree Classifier for prediction.

## Dataset

The dataset (`train.csv`) consists of columns like `id`, `title`, `author`, `text`, and `label`. Preprocessing steps include handling missing values, text cleaning, and dropping unnecessary columns.

### Data Summary

- Total Entries: 20800
- Features: 'id', 'title', 'author', 'text'
- Target Variable: 'label' (0 for reliable, 1 for unreliable)

## Text Preprocessing

The text data is preprocessed using NLTK library. This involves stemming, lowercasing, and removal of stopwords for better model performance.

## Model Training

A Decision Tree Classifier is chosen for its simplicity and interpretability. The model is trained on TF-IDF transformed textual features.

### Model Performance

- Achieved an accuracy of approximately 88% on the test set.
- Utilized ROC curves and confusion matrices for model evaluation.

## Deployment

The trained model, along with the TF-IDF Vectorizer, is saved using Pickle for deployment. A Streamlit application is a potential option for interactive use.

### Requirements

- pandas
- scikit-learn
- nltk
- streamlit

Install the required dependencies using:

```bash
pip install -r requirements.txt

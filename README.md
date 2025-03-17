# Fake News Detection with Machine Learning

Overview

This project aims to classify news articles as either fake or real using various machine learning models, including deep learning and traditional classifiers. The dataset contains articles with their respective labels:

0: Fake News

1: Real News

Dataset

The dataset consists of the following columns:

label: 0 (fake) or 1 (real)

title: The headline of the news article

text: The full content of the article

subject: The category or topic of the news

date: The publication date of the article

A separate validation dataset is also provided, containing some entries labeled as 2, which need to be predicted by the trained model.

Preprocessing

Lowercasing: Convert all text to lowercase.

Punctuation Removal: Remove special characters and punctuation.

Tokenization: Split text into words.

Stopword Removal: Remove common words that do not add much meaning (e.g., "the", "is").

Lemmatization: Convert words to their base form (e.g., "running" → "run").

Model Selection & Training

Several models were tested:

Logistic Regression - Accuracy: 0.9912

Multinomial Naive Bayes (Tuned) - Accuracy: 0.9509

Support Vector Machine (SVM) - Accuracy: 0.9929

Random Forest - Accuracy: 0.9972

Convolutional Neural Network (CNN) - Accuracy: 0.9962

Long Short-Term Memory (LSTM) (Word2Vec Embeddings)

Best Model

The Random Forest model with Word2Vec embeddings achieved the highest accuracy (0.9972) and was selected for the final predictions.

Predictions on Validation Data

The validation dataset contained articles with labels 2, which needed to be classified as 0 (Fake) or 1 (Real).

The trained Random Forest model was used to predict the missing labels.

The updated dataset was saved as validation_data_updated.csv.

Visualizations

Bar chart comparing model accuracies

Distribution of Fake vs. Real News in training data

Distribution of Fake vs. Real News in validation data

How to Use the Model

Ensure required dependencies are installed (pip install -r requirements.txt).

Run the script to train the model and classify the validation dataset.

The updated predictions will be stored in dataset/validation_data_updated.csv.

Future Improvements

Experiment with Transformer-based models (e.g., BERT, RoBERTa) for better performance.

Further hyperparameter tuning for CNN and LSTM models.

Explore ensemble methods combining multiple classifiers.

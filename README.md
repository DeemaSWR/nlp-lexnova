# NLP-Lexnova.
# Fake News Detection with Machine Learning

## Overview

This project aims to classify news articles as either fake or real using various machine learning models, including deep learning and traditional classifiers. The dataset contains articles with their respective labels:

- **0:** Fake News
- **1:** Real News

## Model Selection & Training

Several models were tested:

- **Logistic Regression** - Accuracy: **0.9866**
- **Multinomial Naive Bayes (Tuned)** - Accuracy: **0.9461**
- **Support Vector Machine (SVM)** - Accuracy: **0.9929**
- **Random Forest** - Accuracy: **0.9971**
- **Convolutional Neural Network (CNN)** - Accuracy: **0.9984**
- **Long Short-Term Memory (LSTM)** - Accuracy: **0.9964**
- **Logistic Regression (Word2Vec Embeddings)** - Accuracy: **0.9933**

## Best Model

The **CNN model** achieved the highest accuracy (**0.9984**) and was selected for the final predictions.

## Predictions on Validation Data

- The validation dataset contained articles with labels **2**, which needed to be classified as **0 (Fake)** or **1 (Real)**.
- The trained **CNN model** was used to predict the missing labels.
- The updated dataset was saved as **validation_data_CNN.csv**.

## Future Improvements

- Experiment with **Transformer-based models** (e.g., **BERT, RoBERTa**) for better performance.
- Further **hyperparameter tuning** for **CNN** and **LSTM** models.
- Explore **ensemble methods** combining multiple classifiers.

  
## Google Drive Link
Access all project files here: [Google Drive Folder](https://drive.google.com/drive/folders/1lJUwuWGWSneAnLcY_U4SCD7nuyJ-WTn7?usp=drive_link)

## Fake News Detector Application
Access the application here: [Fake News Detector](https://fakenewsdetectorlexnova.streamlit.app/)


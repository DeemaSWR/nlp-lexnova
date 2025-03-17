# Fake News Detection with Machine Learning

## Overview
This project aims to classify news articles as either fake or real using various machine learning models, including deep learning and traditional classifiers. The dataset contains articles with their respective labels:

- **0**: Fake News
- **1**: Real News

A separate validation dataset is also provided, containing some entries labeled as **2**, which need to be predicted by the trained model.

## Dataset
The dataset consists of the following columns:

- **label**: 0 (Fake) or 1 (Real)
- **title**: The headline of the news article
- **text**: The full content of the article
- **subject**: The category or topic of the news
- **date**: The publication date of the article

## Preprocessing
To prepare the dataset for training, the following preprocessing steps were applied:

1. **Lowercasing**: Convert all text to lowercase.
2. **Punctuation Removal**: Remove special characters and punctuation.
3. **Tokenization**: Split text into words.
4. **Stopword Removal**: Remove common words that do not add much meaning (e.g., "the", "is").
5. **Lemmatization**: Convert words to their base form (e.g., "running" → "run").

## Model Selection & Training
Several machine learning models were tested to classify news articles:

- **Logistic Regression** - Accuracy: **0.9912**
- **Multinomial Naive Bayes (Tuned)** - Accuracy: **0.9509**
- **Support Vector Machine (SVM)** - Accuracy: **0.9929**
- **Random Forest** - Accuracy: **0.9972**
- **Convolutional Neural Network (CNN)** - Accuracy: **0.9962**
- **Long Short-Term Memory (LSTM) (Word2Vec Embeddings)**

## Best Model
The **Random Forest model** with **Word2Vec embeddings** achieved the highest accuracy (**0.9972**) and was selected for the final predictions.

## Predictions on Validation Data
The validation dataset contained articles with labels **2**, which needed to be classified as either **0 (Fake)** or **1 (Real)**. The trained **Random Forest model** was used to predict these missing labels. The updated dataset was saved as:

```
dataset/validation_data_updated.csv
```

## Visualizations
Several visualizations were created to analyze the dataset and model performance:
- **Bar chart comparing model accuracies**
- **Distribution of Fake vs. Real News in training data**
- **Distribution of Fake vs. Real News in validation data**

## How to Use the Model
1. Ensure required dependencies are installed:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the script to train the model and classify the validation dataset:
   ```sh
   python train_model.py
   ```
3. The updated predictions will be stored in:
   ```
   dataset/validation_data_updated.csv
   ```

## Future Improvements
- Experiment with **Transformer-based models** (e.g., BERT, RoBERTa) for improved performance.
- Further **hyperparameter tuning** for CNN and LSTM models.
- Explore **ensemble methods** combining multiple classifiers for better accuracy.


                                              🐦 Twitter Sentiment Analysis

A complete Machine Learning + NLP + Flask Web App project that performs Sentiment Analysis on Twitter text data and classifies tweets into Positive, Negative, or Neutral sentiments.

                                      🚀 Features : 

📊 Cleaned & preprocessed Twitter dataset (text normalization, stopwords removal, lemmatization).

☁️ 3D WordCloud visualization for Positive, Negative, and Neutral tweets.

🔠 Feature extraction using TF-IDF Vectorizer with n-grams.

⚖️ Handled class imbalance using SMOTE.

🤖 Trained & optimized multiple ML models (Logistic Regression, SVM, Random Forest).

🎯 Final model: SVM with Hyperparameter Tuning (RandomizedSearchCV).

🏆 Achieved high accuracy with balanced classification report.

🌍 Built Flask web app with a modern UI for real-time sentiment prediction.

💾 Model & vectorizer saved using pickle.

                          🛠️ Tech Stack :

Programming Language: Python

Libraries: scikit-learn, pandas, numpy, nltk, seaborn, matplotlib, plotly, wordcloud

NLP: TF-IDF, Lemmatization, Stopword removal

ML Algorithms: Logistic Regression, SVM, Random Forest

Deployment: Flask API + HTML/CSS/JS frontend

Visualization: WordCloud, Plotly 3D scatter

                            🔬 Model Training Workflow : 

1. Dataset Loading: Used twitter_training.csv with 70K samples.

2. Data Cleaning: Removed duplicates, handled missing values, preprocessed text.

3. Text Processing:

Stopwords removal

Lemmatization

TF-IDF vectorization (bigrams included, 40K max features)

4. Class Balancing: SMOTE oversampling

5. Model Training:

Compared Logistic Regression, SVM, and Random Forest

Used RandomizedSearchCV to tune hyperparameters of SVM

Evaluation: Accuracy, Confusion Matrix, Classification Report

              📊 Results:

Final Model: SVM (with optimized hyperparameters)

Accuracy: ~90% (depending on dataset split)

Balanced performance across Positive, Negative, Neutral classes.

                🎨 Web Application : 

Built using Flask (backend API) + HTML/CSS/JS (frontend).

Users can enter one or multiple tweets.

Real-time prediction displayed with styled sentiment categories:

✅ Positive → Green highlight

⚠️ Neutral → Yellow highlight

❌ Negative → Red highlight

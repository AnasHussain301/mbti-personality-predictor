MBTI Personality Predictor

An end-to-end Natural Language Processing (NLP) and Machine Learning application that predicts Myers–Briggs Type Indicator (MBTI) personality types from textual data.

The project implements a complete ML pipeline—from data preprocessing and feature engineering to model evaluation and deployment—leveraging both traditional machine learning techniques and modern NLP representations.

The application is deployed using Streamlit, providing an interactive interface for real-time personality prediction.

Live Demo

🌐 Application

https://mbti-personality-predictor-dwq4vrj65e6mdl2wg9onf7.streamlit.app/

Project Highlights
End-to-End Machine Learning Pipeline
NLP-based Personality Classification
Advanced Feature Engineering
Multiple Model Benchmarking
Class Imbalance Handling using SMOTE
Cross-Validation and Model Selection
Streamlit Deployment
Modular Python Architecture
Features
Data Processing
Data cleaning and preprocessing
Text normalization
Feature extraction
Label encoding
Dataset balancing using SMOTE
Feature Engineering

The project combines multiple feature extraction techniques to improve prediction performance:

TF-IDF Vectorization
BERT Sentence Embeddings
VADER Sentiment Analysis
Stylometric Features (writing style analysis)

This hybrid feature engineering approach captures both semantic meaning and linguistic writing patterns.

Machine Learning Models

Multiple classification algorithms were trained and evaluated:

Logistic Regression
Support Vector Machine (Linear SVM)
Random Forest
Multi-Layer Perceptron (MLP)

Each model was benchmarked using cross-validation before selecting the final production model.

Model Evaluation

The project includes:

Cross Validation
Accuracy Comparison
Classification Metrics
Best Model Selection
Performance Visualization

Rather than relying on a single algorithm, the pipeline evaluates multiple approaches and deploys the best-performing model.

Machine Learning Pipeline
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Text Preprocessing
      │
      ▼
Feature Engineering
(TF-IDF + BERT + VADER + Stylometric)
      │
      ▼
SMOTE
(Class Balancing)
      │
      ▼
Multiple ML Models
(LogReg, SVM, RF, MLP)
      │
      ▼
Cross Validation
      │
      ▼
Model Comparison
      │
      ▼
Best Model Selection
      │
      ▼
Model Serialization
      │
      ▼
Streamlit Web Application
Technology Stack
Programming
Python
Machine Learning
Scikit-learn
Imbalanced-Learn (SMOTE)
Joblib
Natural Language Processing
BERT Embeddings
TF-IDF
VADER Sentiment Analysis
Stylometric Feature Extraction
Models
Logistic Regression
Linear SVM
Random Forest
Multi-Layer Perceptron (MLP)
Deployment
Streamlit
Project Structure
Dataset
      │
      ▼
EDA
      │
      ▼
Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Training
      │
      ▼
Evaluation
      │
      ▼
Prediction
      │
      ▼
Streamlit UI
Why This Project?

Unlike many personality prediction projects that rely on a single machine learning algorithm, this implementation follows a complete ML experimentation workflow:

Comprehensive data preprocessing
Feature engineering using multiple NLP techniques
Handling class imbalance with SMOTE
Training and benchmarking multiple machine learning models
Cross-validation for robust evaluation
Deployment of the best-performing model as a web application

This makes the project a practical demonstration of an end-to-end NLP and Machine Learning pipeline suitable for real-world text classification tasks.

Skills Demonstrated
Natural Language Processing (NLP)
Machine Learning
Feature Engineering
Data Preprocessing
Class Imbalance Handling (SMOTE)
Text Classification
Model Evaluation
Cross Validation
Hyperparameter Selection
Streamlit Deployment
Python Software Engineering

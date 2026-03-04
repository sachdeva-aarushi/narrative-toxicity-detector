# Narrative Toxicity Detector

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-green)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)

## Overview

Narrative Toxicity Detector is a machine learning system designed to identify toxic language, insults, and harmful narrative patterns within online conversations.

The project combines natural language processing techniques and transformer-based models to analyze textual data and classify various forms of toxic or harmful communication. It is designed as an end-to-end ML pipeline including data preprocessing, feature engineering, model training, evaluation, and deployment through an interactive interface.

The system is capable of detecting conversational signals such as insults, harassment, aggressive tone, and emerging toxic narratives.

---

## Problem Statement

Online communication platforms frequently experience harmful interactions including harassment, toxic comments, and targeted narratives that degrade discourse quality.

Traditional moderation systems often rely on rule-based filtering, which fails to capture contextual toxicity and evolving linguistic patterns.

This project aims to build a machine learning system capable of:

* Detecting toxic language and insults
* Identifying harmful narrative patterns
* Classifying different forms of conversational toxicity
* Providing explainable predictions through interpretable ML outputs

---

## Key Features

* Transformer-based text classification
* Toxicity and narrative pattern detection
* End-to-end ML pipeline architecture
* Data preprocessing and feature engineering
* Model evaluation and visualization
* Interactive Streamlit application
* Modular and scalable repository structure

---

## Repository Structure

```
narrative-toxicity-detector
│
├── data
│   ├── raw
│   ├── processed
│   └── external
│
├── notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src
│   ├── data
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   │
│   ├── features
│   │   └── feature_engineering.py
│   │
│   ├── models
│   │   ├── train_model.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   ├── visualization
│   │   └── plot_metrics.py
│   │
│   └── utils
│       └── helpers.py
│
├── models
│   ├── trained_model.pkl
│   └── tokenizer.pkl
│
├── reports
│   ├── figures
│   └── model_performance.md
│
├── app
│   └── streamlit_app.py
│
├── tests
│   └── test_preprocessing.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Dataset

The project can be trained using publicly available toxicity and hate-speech datasets including:

* Jigsaw Toxic Comment Classification Dataset
* Hate Speech and Offensive Language Dataset
* GoEmotions Dataset

These datasets provide labeled text samples for categories such as:

* Toxic
* Insult
* Threat
* Harassment
* Neutral

The datasets are stored under the `data/raw` directory and processed into structured training formats under `data/processed`.

---

## Methodology

The pipeline follows a structured machine learning workflow.

### Data Preprocessing

Steps include:

* Text normalization
* Lowercasing
* Removal of stopwords
* Tokenization
* Lemmatization
* Handling missing values

### Feature Engineering

Multiple representations can be explored:

* TF-IDF vectorization
* Transformer embeddings
* Sentence embeddings

### Model Architecture

The system supports multiple model types:

Baseline models

* Logistic Regression
* Support Vector Machines

Deep learning models

* BERT
* RoBERTa
* Transformer-based classifiers

---

## System Architecture

```
Raw Text Input
      │
      ▼
Text Preprocessing
      │
      ▼
Tokenization
      │
      ▼
Feature Representation
(TF-IDF / Transformer Embeddings)
      │
      ▼
Classification Model
      │
      ▼
Toxicity Prediction
```

---

## Installation

Clone the repository:

```
git clone https://github.com/sachdeva-aarushi/narrative-toxicity-detector.git
cd narrative-toxicity-detector
```

Install dependencies:

```
pip install -r requirements.txt
```

Download spaCy language model:

```
python -m spacy download en_core_web_sm
```

---

## Running the Model

Training the model:

```
python src/models/train_model.py
```

Evaluating the model:

```
python src/models/evaluate.py
```

Generating predictions:

```
python src/models/predict.py
```

---

## Running the Web Application

To launch the interactive Streamlit interface:

```
streamlit run app/streamlit_app.py
```

The application allows users to input text and receive real-time toxicity predictions.

---

## Evaluation Metrics

Model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Performance visualizations are generated under:

```
reports/figures
```

---

## Future Improvements

Potential extensions include:

* Conversation-level toxicity detection
* Narrative propagation analysis
* Real-time moderation integration
* Multilingual toxicity detection
* Explainable AI for model interpretation
* Graph-based conversation modeling

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

---

## Author

Aarushi Sachdeva


# Support Ticket Classification
### Automated Banking Complaint Classification using FastText

---

## Overview

This project presents an end-to-end automated complaint classification 
system developed for the banking sector. The system leverages the 
FastText machine learning model to classify customer complaints into 
predefined departments, reducing manual effort and improving complaint 
resolution speed.

According to the Reserve Bank of India, approximately 13.34 lakh 
complaints were recorded across all banking departments in 2025. 
This system was developed to address this large-scale classification 
challenge through machine learning and natural language processing.

The system incorporates a multilingual translation pipeline, a trained 
FastText classification model, and a real-time web application built 
using Flask, enabling end-to-end complaint management from submission 
to resolution.

---

## Research

This project was developed as part of an academic research study 
at Alliance University, Bangalore. A research paper based on this 
work is currently under preparation for publication.

**Title:** Automatic Banking Ticket Classification using FastText
**Author:** Gurrala Rohith Kumar
**Institution:** Department of Data Science, Alliance University, 
Bangalore, India

---

## Key Features

- Automated multi-class complaint classification using FastText
- Multilingual input support with automatic language detection
  and translation to English (Hinglish, Tanglish)
- Real-time prediction delivered in under one second on 
  standard hardware without GPU
- Agent feedback mechanism for correcting misclassified complaints
- Jobs Dashboard for managing and tracking active complaints
- History page for viewing resolved complaints
- Lightweight compressed model (.ftz) for easy deployment

---

## Model Details

| Property               | Details                            |
|------------------------|------------------------------------|
| Algorithm              | FastText Supervised Classification |
| Dataset                | CFPB Consumer Complaint Dataset    |
| Original Records       | Approximately 20 lakh              |
| Records After Cleaning | Approximately 7.6 lakh             |
| Number of Categories   | 5                                  |
| Model Format           | fasttext_model.ftz                 |
| Training Framework     | Python, FastText Library           |
| Web Framework          | Flask                              |

### Classification Categories

- Loans
- Credit Card Services
- Bank Accounts and Services
- Debt Collection
- Credit Reporting

---

## Performance

### Overall Results

| Metric      | Score  |
|-------------|--------|
| Accuracy    | 91.69% |
| Precision   | 91.66% |
| Recall      | 91.69% |
| F1 Score    | 91.67% |
| Weighted F1 | 0.92   |

### Category-wise Classification Report

| Category                   | Precision | Recall | F1 Score | Support  |
|----------------------------|-----------|--------|----------|----------|
| Bank Accounts and Services | 0.92      | 0.94   | 0.93     | 31,555   |
| Credit Card Services       | 0.88      | 0.85   | 0.86     | 30,652   |
| Credit Reporting           | 0.94      | 0.95   | 0.94     | 1,38,202 |
| Debt Collection            | 0.88      | 0.86   | 0.87     | 47,811   |
| Loans                      | 0.91      | 0.91   | 0.91     | 45,741   |
| Weighted Average           | 0.92      | 0.92   | 0.92     | 2,93,961 |

---

## Model Experiments

Five configurations were evaluated to identify the optimal 
hyperparameter settings.

| Experiment | Configuration       | Accuracy |
|------------|---------------------|----------|
| 1          | Default             | 86.0%    |
| 2          | Increased Epochs    | 86.1%    |
| 3          | Higher Epochs       | 87.0%    |
| 4          | N-grams with Epochs | 88.6%    |
| 5          | AutoTune (Final)    | 91.69%   |

The final model used FastText AutoTuning with wordNgrams=3 and 
F1 as the optimization metric, achieving the best overall accuracy 
of 91.69%.

---

## System Pipeline

```
User Input (Complaint Text)
          |
          v
Language Detection and Translation Module
(Multilingual Input converted to English)
          |
          v
Text Cleaning and Preprocessing
(Lowercasing, Noise Removal, Tokenization)
          |
          v
Feature Extraction
(Subword N-grams + Word N-grams + Word Embeddings)
          |
          v
FastText Classification Model
          |
          v
Predicted Category
          |
          v
Flask Web Application
          |
          v
Jobs Dashboard
          |
          v
Agent Review and Feedback
          |
          v
History Page (Resolved Complaints)
```

---

## Project Structure

```
support-ticket-classification/
|
|-- templates/
|   |-- home.html                   Complaint submission and result page
|   |-- job.html                    Jobs dashboard
|   |-- history.html                Resolved complaints page
|
|-- static/
|   |-- css/
|       |-- style.css               Stylesheet
|
|-- app.py                          Flask web application
|-- train.py                        Model training script
|-- EvaluationMetrics.py            Model evaluation script
|-- data_cleaning.ipynb             Data cleaning and preprocessing notebook
|-- README.md                       Project documentation
```

> **Note:** The trained model file (fasttext_model.ftz) and dataset
> files (train.txt, test.txt) are not included in this repository
> due to their large file size. The dataset is publicly available
> on Kaggle — CFPB Consumer Complaint Dataset.

---

## Installation

### Prerequisites

- Python 3.8 or above
- pip package manager

## Technology Stack

| Component        | Technology                    |
|------------------|-------------------------------|
| Machine Learning | FastText                      |
| Backend          | Python, Flask                 |
| Frontend         | HTML, CSS, JavaScript         |
| Translation      | deep-translator (Google API)  |
| Data Processing  | Pandas, NumPy                 |
| Evaluation       | Scikit-learn                  |
| Model Storage    | FastText .ftz compressed file |

---

## Limitations

- The model supports single-label classification only and cannot 
  handle multi-intent complaints
- Translation accuracy for highly informal mixed-language text 
  may affect classification performance
- The absence of labeled Indian banking complaint datasets limits 
  further domain-specific fine-tuning

---

## Future Work

- Implementation of multi-label classification to handle 
  multi-intent complaints
- Integration of a permanent database using MySQL or MongoDB 
  for persistent complaint storage
- Development of an automatic retraining pipeline using 
  agent-corrected complaint data
- Incorporation of sentiment-based analysis for complaint 
  priority assignment based on account type and issue severity

---

## Acknowledgements

- Consumer Financial Protection Bureau (CFPB) for the complaint 
  dataset available on Kaggle
- Facebook AI Research for the FastText library
- Flask open-source web framework community
- deep-translator open-source library

---

## Author

**Gurrala Rohith Kumar**
MSc Data Science (2025 - 2027)
Alliance University, Bangalore, India

GitHub: github.com/AutomationArtist01
Email: rohithgurrala14@gmail.com
LinkedIn: linkedin.com/in/rohith-kumar-gurrala/

---

## License

This project is intended for academic and research purposes.

---

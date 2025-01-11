# Breast Cancer Diagnosis Using Machine Learning

This project implements and evaluates seven machine learning models for breast cancer diagnosis using the Breast Cancer dataset from `scikit-learn`. The study focuses on preprocessing, classification, and performance comparison, with the goal of identifying the most effective model for diagnosing breast cancer.

---

## Project Overview

Breast cancer diagnosis is a critical task in medical diagnostics, where early and accurate detection significantly improves patient outcomes. This project explores the performance of seven distinct machine learning models across key metrics like accuracy, precision, and recall. The dataset used contains real-world data on tumor features, categorized as malignant or benign.

---

## Models Implemented

1. **Naive Bayes (GaussianNB):** A probabilistic model leveraging Bayes' theorem with a Gaussian distribution assumption.
2. **K-Nearest Neighbors (KNN):** A distance-based classification method using `k=8`.
3. **Decision Tree (DT):** A tree-based model for splitting data hierarchically, configured with `max_depth=64`.
4. **Random Forest (RF):** An ensemble model combining multiple decision trees with `500 estimators`.
5. **Support Vector Machine (SVM):** A model finding the optimal hyperplane to separate classes.
6. **Logistic Regression (LR):** A linear model for binary classification.
7. **Artificial Neural Network (ANN):** A one-hidden-layer ANN with 256 neurons and ReLU activation.

---

## Methodology

### 1. Dataset
- **Source:** Breast Cancer dataset from `sklearn.datasets`.
- **Features:** 30 numerical features related to tumor characteristics.
- **Labels:** Binary classification into malignant (1) or benign (0).

### 2. Data Preprocessing
- Data splitting into training (80%) and testing (20%) subsets.
- Feature scaling using Min-Max normalization.

### 3. Model Evaluation
- Training each model on preprocessed data.
- Testing and evaluating using metrics:
  - **Accuracy** (train and test)
  - **Precision** (test)
  - **Recall** (test)

### 4. Visualization
- Comparative bar charts for accuracy, precision, and recall across models.

---

## Results

Each model's performance is evaluated across accuracy, precision, and recall metrics. The ANN and Random Forest models show promising results, while simpler models like Naive Bayes provide a baseline for comparison.

| **Model**         | **Train Accuracy** | **Test Accuracy** | **Precision** | **Recall** |
|--------------------|--------------------|-------------------|---------------|------------|
| Naive Bayes (GNB) | *To be updated*    | *To be updated*   | *To be updated* | *To be updated* |
| KNN               | *To be updated*    | *To be updated*   | *To be updated* | *To be updated* |
| Decision Tree     | *To be updated*    | *To be updated*   | *To be updated* | *To be updated* |
| Random Forest     | *To be updated*    | *To be updated*   | *To be updated* | *To be updated* |
| SVM               | *To be updated*    | *To be updated*   | *To be updated* | *To be updated* |
| Logistic Regression | *To be updated* | *To be updated*   | *To be updated* | *To be updated* |
| ANN               | *To be updated*    | *To be updated*   | *To be updated* | *To be updated* |

---

## Key Features

- **Comprehensive Comparison:** Seven diverse machine learning models analyzed for breast cancer diagnosis.
- **Performance Metrics:** Evaluation using accuracy, precision, and recall for robust comparison.
- **Scalable Workflow:** Designed for extensibility with other datasets and classifiers.

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/breast-cancer-diagnosis.git
cd breast-cancer-diagnosis

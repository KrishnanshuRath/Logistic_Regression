# Logistic_Regression

## Logistic Regression Using Gradient Descent (No Library Support)

---

## Overview

This project implements **Logistic Regression from scratch** using **Gradient Descent**, without relying on machine learning libraries like Scikit-Learn. The model is trained on a binary classification dataset with two independent features. All fundamental components like cost calculation, sigmoid activation, gradient update, and prediction are manually implemented.

---

## Features

- Logistic Regression with Binary Classification
- Gradient Descent Optimization
- Sigmoid Activation Function
- No External ML Libraries (like `sklearn`)
- Feature Normalization Support
- Confusion Matrix, Accuracy, Precision, Recall, and F1-Score Calculations
- Decision Boundary Visualization
- Cost Function vs Iteration Graphs
- Multi-Learning Rate Comparison (e.g., 0.1 and 5.0)

---

## Dataset

The dataset is composed of two CSV files:

- `logisticX.csv`: Contains 100 rows with two feature columns (X₁ and X₂)
- `logisticY.csv`: Contains 100 binary labels (0 or 1)

---

## Model Details

### Hypothesis Function
The sigmoid activation:
```
h(X) = 1 / (1 + exp(-XW))
```

### Cost Function (Log Loss)
```
J(W) = -1/m ∑ [ y * log(h(X)) + (1 - y) * log(1 - h(X)) ]
```

### Gradient Descent Update
```
W := W - α * (1/m) * Xᵀ (h(X) - y)
```

### Decision Boundary
The decision boundary is computed from:
```
w₀ + w₁X₁ + w₂X₂ = 0
⇒ X₂ = -(w₀ + w₁X₁) / w₂
```

---

## Visualizations

- Decision Boundary: Plotted over the data with colored classes.
- Cost vs Iteration Graphs:
  - For learning rate = 0.1
  - For comparison between 0.1 and 5.0
- Confusion Matrix: Visual plot using Matplotlib
- Accuracy Report: Includes precision, recall, and F1-score

---

## Evaluation Metrics

- Accuracy (as both point value and %)
- Precision
- Recall
- F1-Score
- Confusion Matrix Plot

---

## Requirements

- Python 3.x
- Numpy
- Matplotlib

Note: No ML libraries (like scikit-learn) are used in model training and evaluation.

---

## Observations

- The cost function converges smoothly for a learning rate of 0.1
- A high learning rate like 5.0 shows oscillations, and very high values (like 31) cause divergence
- Logistic regression shows clear decision boundary and separation between two classes
- Accuracy and F1-score demonstrate solid performance on the training dataset

---

## Output Summary

- Final Weights: Interpreted into decision boundary equation
- Cost Function after Convergence
- Classification Accuracy (in %)
- Evaluation metrics with confusion matrix
- All outputs visualized and saved

# Titanic Survival Prediction

Predicting passenger survival on the Titanic using a machine learning model in Google Colab.

---

## Highlights
- Based on the famous Titanic dataset from Kaggle
- Data preprocessing, feature engineering, and model training in one notebook
- Built using Python, Pandas, Scikit-learn
- Interactive and reproducible environment in Google Colab

---

## Overview

This project aims to predict which passengers survived the Titanic shipwreck using various passenger attributes (age, gender, class, etc.). The dataset used is one of the most well-known benchmarks for classification tasks.

The entire workflow—from loading and exploring the data to preprocessing, model building, and evaluation—has been implemented in a Google Colab notebook for easy access and use.

---

## Usage Instructions

1. Open the notebook in Google Colab:
   [Open in Colab](https://colab.research.google.com/github/YOUR_USERNAME/titanic-survival-prediction/blob/main/Titanic_Survival_Prediction.ipynb)

2. Run the cells sequentially to:
   - Load and inspect the data
   - Clean and preprocess features
   - Train a classification model
   - Evaluate model performance

---

## Installation Instructions

To run locally (optional):
```bash
!pip install pandas numpy matplotlib seaborn scikit-learn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

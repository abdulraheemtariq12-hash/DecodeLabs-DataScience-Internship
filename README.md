# Iris Classification — DecodeLabs Data Science Internship Project

## 📌 Description
A beginner-level Machine Learning project using the Iris dataset to classify flower species.

This project demonstrates data loading, preprocessing, modeling, and evaluation using Python.

## 📊 Dataset
- Built-in Iris dataset from `scikit-learn`
- Includes 150 samples with 4 features and 3 target classes

## ⚙️ What We Did
- Loaded the Iris dataset using `sklearn.datasets.load_iris`
- Converted the data into a Pandas DataFrame
- Added the target label column
- Checked for missing values
- Removed duplicate records
- Performed basic exploratory analysis
- Split the data into training and test sets
- Trained a `LogisticRegression` model
- Evaluated model accuracy on the test set

## 🧠 Model Used
- `LogisticRegression` from `scikit-learn`

## 🚀 How to Run
```bash
pip install scikit-learn pandas
python main.py
```

## 📁 Project Files
- `main.py` — main Python script for data preparation, model training, and evaluation
- `requirements.txt` — required Python packages
- `README.md` — project overview and instructions
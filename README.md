# 🚢 Titanic Survival Prediction

## 🎯 Objective

This project aims to develop a **Machine Learning model** that predicts whether a passenger survived the Titanic disaster, based on features such as age, sex, passenger class, and more. The model is built using historical data provided by Kaggle.

---

## 📂 Dataset Source

- [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- The dataset contains demographic and other details of passengers, such as:
  - PassengerId
  - Pclass (Ticket class)
  - Name
  - Sex
  - Age
  - SibSp (Siblings/Spouses aboard)
  - Parch (Parents/Children aboard)
  - Ticket
  - Fare
  - Cabin
  - Embarked (Port of Embarkation)
  - Survived (Target Variable)

---

## 🧰 Libraries Used

- `pandas` - for data manipulation
- `numpy` - for numerical operations
- `matplotlib` and `seaborn` - for data visualization
- `scikit-learn` - for building and evaluating the ML model

---

## ⚙️ Features Used

After data preprocessing and feature engineering, the following features are used for prediction:

- `Pclass`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `Embarked`

---

## 🛠️ How to Run the Project

### 1. Clone the Repository

...bash
git clone https://github.com/your-username/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction



Install Required Packages

pip install -r requirements.txt

Run the Script
python titanic_model.py


This script will:

Load the dataset

Preprocess the data

Train a machine learning model

Output predictions

Show model performance metrics


📊 Model Evaluation Metrics
To evaluate the model, the following metrics are used:

✅ Accuracy – Overall correctness of the model

🎯 Precision – Correct positive predictions vs. total predicted positives

🔁 Recall – Correct positive predictions vs. total actual positives

⚖️ F1 Score – Harmonic mean of precision and recall

📋 Classification Report – Full summary including all of the above


📈 Sample Output

Accuracy Score: 0.81
Precision: 0.79
Recall: 0.75
F1 Score: 0.77

              precision    recall  f1-score   support

           0       0.83      0.85      0.84       100
           1       0.78      0.75      0.77        79

    accuracy                           0.81       179
   macro avg       0.81      0.80      0.80       179
weighted avg       0.81      0.81      0.81       179


📌 Highlights
Clean and well-commented code

Exploratory Data Analysis (EDA) included

Robust preprocessing (handling missing values, encoding, scaling)

Model tuning for better performance

🤝 Contributing
Feel free to fork this project, raise issues, and submit pull requests. All suggestions are welcome!

# 📉 Customer Churn Prediction using Artificial Neural Networks (ANN)

This project focuses on predicting **customer churn** using an Artificial Neural Network (ANN). The goal is to help businesses identify customers who are likely to leave so they can take proactive retention actions.

---

## 🚀 Project Overview

Customer churn is a major business problem across telecom, banking, SaaS, and subscription-based industries.  
This project:

- Performs **Exploratory Data Analysis (EDA)**
- Cleans and preprocesses data  
- Builds an **ANN-based churn prediction model**
- Evaluates performance using accuracy, precision, recall, F1 score, and confusion matrix  
- Generates actionable business insights

---

## 🧠 Model Used

### **Artificial Neural Network (ANN)**
- Built using **TensorFlow / Keras**
- Input layer → Hidden layers → Output layer
- Activation functions: ReLU, Sigmoid
- Loss: Binary Crossentropy  
- Optimizer: Adam

---

## 📊 Key Steps in the Project

### ✔ 1. Data Preprocessing
- Handling missing values  
- Label encoding / One-hot encoding  
- Feature scaling (StandardScaler/MinMaxScaler)

### ✔ 2. EDA (Exploratory Data Analysis)
- Churn distribution  
- Customer tenure analysis  
- Contract type, monthly charges, payment method  
- Correlation heatmaps & outlier detection

### ✔ 3. Model Building
- Train/validation split  
- ANN model creation  
- Hyperparameter tuning  
- Training with early stopping  

### ✔ 4. Evaluation
- Confusion matrix  
- Precision, Recall, F1-score  
- ROC/AUC curve  
- Feature importance (permutation-based)

---

## 📈 Results

- **Model Accuracy:** XX%  
- **AUC Score:** XX  
- **Top Factors Influencing Churn:**  
  - Contract Type  
  - Monthly Charges  
  - Tenure  
  - Internet Service Type  

---

## 🛠 Technologies Used

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- TensorFlow / Keras  
- Jupyter Notebook  

---

## ⚙️ How to Run

```bash
git clone https://github.com/Rayyaan23/churn-rate-analysis
cd churn-rate-analysis
pip install -r requirements.txt

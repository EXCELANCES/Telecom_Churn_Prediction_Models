# Telecom Churn Prediction Project

## 📄 Project Overview

This project aims to predict customer churn in the telecommunications industry using various supervised machine learning models. By analyzing a rich dataset of customer demographics, account details, and service usage behaviors, we built and evaluated predictive models that help telecom companies identify at-risk customers and improve retention strategies.

## 📊 Dataset Summary

* **Source:** Synthetic telecom customer dataset (7,043 entries)
* **Features:** 21 attributes including:

  * Demographics: gender, SeniorCitizen, Partner, Dependents
  * Account info: tenure, Contract, PaymentMethod
  * Services: PhoneService, InternetService, TechSupport, StreamingTV, etc.
  * Charges: MonthlyCharges, TotalCharges
* **Target Variable:** `Churn` (Yes/No)

## ⚖️ Models Implemented

* **Support Vector Machines (SVM):**

  * Kernels: Linear, Polynomial, RBF, Sigmoid
  * Best performance: **Linear SVM** with \~82% accuracy

* **Decision Tree Classifier:**

  * Criteria: Entropy and Gini
  * Pruning and hyperparameter tuning applied
  * Best performance after pruning: **81.12% accuracy**

* **Random Forest Classifier:**

  * Default and grid-search tuned versions
  * Best performance after tuning: **79.79% accuracy**, **78.57% F1-score**

## 🔢 Data Preparation Steps

1. **Cleaning:**

   * Converted spaces to NaN, handled missing values (<0.2%) by dropping rows
   * Converted categorical/binary variables to appropriate types

2. **Feature Engineering:**

   * Created new features: `charges_per_month`, `high_monthly_charge`, `long_term_customer`
   * Applied binary encoding and one-hot encoding where appropriate

3. **Feature Scaling:**

   * Applied `StandardScaler` to `tenure`, `MonthlyCharges`, `TotalCharges`
   * Generated normalized versions for improved model performance

## ⚖️ Model Evaluation Metrics

Each model is evaluated on:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

## 📁 File Structure

```
Telecom_Churn_Prediction/
├── Telecom_Churn_Prediction_Models.py    # Main codebase
├── Telecom_Churn_Report.pdf              # Full project report
├── Telecom_Churn_Data.csv                # Dataset (to be added manually)
├── README.md                             # This file
```

## 📅 How to Run

1. Make sure Python 3 and required libraries are installed:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

2. Place `Telecom_Churn_Data.csv` in the same directory.

3. Run the script:

```bash
python Telecom_Churn_Prediction_Models.py
```

## 🚀 Results Snapshot

| Model               | Accuracy | F1 Score | Precision |
| ------------------- | -------- | -------- | --------- |
| SVM (Linear Kernel) | 82%      | 76%      | High      |
| Random Forest       | 79.79%   | 78.57%   | 78.64%    |
| Decision Tree       | 81.12%   | 59%      | 69%       |

## 🌐 License

This project is open-source and free to use for educational and research purposes.

## 📝 Acknowledgements

* IBM Machine Learning Docs
* Scikit-learn Documentation
* Telecom customer churn datasets

---

✨ *Built with curiosity and code to understand customer behavior better.*

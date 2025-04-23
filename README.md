# 🩺 Maternal Health Risk Prediction using Machine Learning

This project focuses on predicting **maternal health risk levels** (Low, Mid, High) using clinical data collected through IoT-based risk monitoring systems. The goal is to assist healthcare providers in identifying high-risk pregnancies early through data-driven insights.

## 📌 Project Highlights

- 📊 **Exploratory Data Analysis (EDA)** with visualizations and statistical summaries
- ⚙️ **Data preprocessing** with label encoding and feature scaling
- 🌲 **Model training** using a Random Forest Classifier
- 🧠 **Model evaluation** with precision, recall, F1-score, and confusion matrix
- 🚀 Suggestions for further improvements: tuning, alternative models, and ensemble methods

## 📁 Dataset

- The dataset includes features such as:
  - `Age`: Age of the woman
  - `SystolicBP`: Upper blood pressure
  - `DiastolicBP`: Lower blood pressure
  - `BS`: Blood sugar level (mmol/L)
  - `HeartRate`: Heart rate (bpm)
  - `RiskLevel`: Target label (Low, Mid, High)

- Sourced from: [Kaggle Dataset](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data)

## 🛠️ Tech Stack

- Python
- Google Colab
- `pandas`, `numpy`
- `seaborn`, `matplotlib`
- `scikit-learn`

## 📈 Model Performance

| Class        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Low Risk     | 0.82      | 0.91   | 0.86     |
| Mid Risk     | 0.83      | 0.75   | 0.79     |
| High Risk    | 0.76      | 0.78   | 0.77     |

- **Overall Accuracy**: **80%**
- Evaluated on a 30% test split
- Includes a confusion matrix for detailed insight

## ✅ Possible Improvements

- 🔧 Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- 🤖 Try alternative models: XGBoost, SVM, CatBoost
- 🔄 Use K-Fold Cross Validation
- ⚖️ Handle class imbalance with SMOTE or class weights
- 🔁 Ensemble learning methods (Voting, Bagging, Boosting)

## 📸 Sample Visualizations

- Count plot of `RiskLevel`
- Boxplots showing relationship between `Age`, `BP`, and `RiskLevel`
- Confusion matrix for model evaluation

## 📜 License

This project is part of an educational exercise under the **Data Science and Machine Learning Club**. Feel free to fork, clone, and build on it for non-commercial use.

---

### 👤 Author
**Ndabezinhle Mqhele Dlamini**  
Data Science and Machine Learning Club  
University of Johannesburg

---

> _"Data saves lives — especially when paired with good predictions."_ 💡

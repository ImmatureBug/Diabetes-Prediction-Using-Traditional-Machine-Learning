# Diabetes-Prediction-Using-Traditional-Machine-Learning
A machine learning mini project for predicting diabetes using decision tree classifier as the main, k-nearest neighbors, and naive bayes with SHAP explainability
# Diabetes Prediction Using Traditional Machine Learning Techniques

This project applies traditional machine learning models for predicting the onset of Diabetes in using the **Pima Indians Diabetes Dataset.** This project maps to **SDG 3 (Good Health and Well-being)** by allowing preventative and early detection.

## 🔍 Problem statement.
Diabetes continues to be a public health crisis and the early diagnosis is critical. The goal of this project was to create a model that will classify a person as a diabetic or not based on clinical observations using interpretable machine learning models.

## 📊 Dataset.
- **Source**: UCI Machine Learning Repository
- **Dataset**: Pima Indians Diabetes
- **Sample Size**: 768
- **Features**: Glucose, BMI, Age, Blood Pressure, Skin Thickness, Insulin, Diabetes Pedigree Function, Pregnancies, Outcome
- There were a variety of values that were implausible and missing in the data, so zero entries were replaced by median values.

## ⚙️ The models trained.
Four traditional models were trained and compared:
- Decision Tree Classifier
- Naive Bayes
- k-Nearest Neighbors (kNN)
- Logistic Regression

## 📈 Evaluation Metrics.
All models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC Score
- Confusion Matrix metric
- 5-fold cross validated

### 🏆 Top Models
| Model              | Accuracy | F1 Score  | ROC AUC      |
|--------------------|----------|-----------|--------------|
| Naive Bayes        | 72.7%    | 0.625     | 0.775        |
| Logistic Regression | 70.1%    | 0.540     | 0.813        |
| Decision Tree      | 67.5%    | 0.456     | 0.742        |
| kNN                | 70.8%    | 0.545     | 0.801        |

## 🔍 Explainability
- **SHAP (SHapley Additive exPlanations)** was used to interpret model decisions.
- **Mutual Information** scores were used to evaluate feature importance.
- Risk reports were generated to explain the most influential features per prediction.

## ✅ SDG 3 Alignment
This project supports Sustainable Development Goal 3 by promoting early detection of diabetes using accessible and interpretable ML tools, especially in low-resource healthcare settings.

## 📁 Files in This Repository
- `diabetes_prediction_using_traditional_machine_learning_techniques.py`: Main code
- `diabetes_model.pkl`: Saved Decision Tree model
- `scaler.pkl`: Standard scaler for future predictions
- `*.png`: Visualizations (confusion matrix, MI scores, SHAP plots)
- `README.md`: Project summary and overview

## 📌 How to Run
1. Clone this repo
2. Open the notebook/script in Jupyter or Google Colab
3. Run the cells step-by-step
4. Use the saved model and scaler to make predictions

## 🤝 Authors
- **Olowe Oluwadarasimi Oluwaseun** (LCU/PG/007728)  
- **Ademoroti Ayobami** (LCU/PG/009018)  

## 📜 License
For academic and educational use only.

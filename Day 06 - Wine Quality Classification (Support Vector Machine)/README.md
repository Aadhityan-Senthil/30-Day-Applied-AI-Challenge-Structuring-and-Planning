# 🎯 Day 6: Wine Quality Classification using Support Vector Machine (SVM)  

## 📌 Overview  
In this project, we implement **Wine Quality Classification** using **Support Vector Machine (SVM)**. The goal is to classify wine as good or bad based on physicochemical properties such as acidity, alcohol content, and pH levels.  

---

## 🚀 What You'll Learn  
✅ **Exploratory Data Analysis (EDA)** to identify key factors affecting wine quality  
✅ **Data Preprocessing** (Handling Missing Values, Feature Scaling, Encoding)  
✅ **Support Vector Machine (SVM)** for Classification  
✅ **Hyperparameter Tuning** using **GridSearchCV**  
✅ Model Evaluation using **Accuracy, Precision, Recall & F1-score**  
✅ **Feature Importance Analysis** using **SHAP**  

---

## 📂 Project Structure  
```
Day-6-Wine-Quality-Classification/
│── README.md  # Documentation  
│── wine_quality_svm.ipynb  # Interactive Notebook  
│── wine_quality_svm.py  # Standalone Python Script  
│── winequality.csv  # Dataset  
```
---

## 📌 Why Both .py and .ipynb?  
This project includes:  

📒 **Jupyter Notebook (.ipynb)** → For interactive exploration, visualization, and debugging.  
💻 **Python Script (.py)** → For direct execution and efficiency.  

---

## 📊 Dataset  
We use the **Wine Quality Dataset** from Kaggle, which contains:  

📌 **Features:** Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol  
🎯 **Target Variable:** Wine Quality (Good/Bad)  

---

## 🔧 Technologies Used  
🔹 Python  
🔹 Pandas & NumPy (Data Processing)  
🔹 Scikit-learn (ML Models & Evaluation)  
🔹 Matplotlib & Seaborn (Visualization)  
🔹 Support Vector Machine (SVM)  
🔹 SHAP (Feature Importance)  

---

## 📜 How to Run the Project?  
1️⃣ Clone the repository  
```bash
git clone https://github.com/Aadhityan-Senthil/30-Day-Applied-AI-Challenge.git  
cd 30-Day-Applied-AI-Challenge/Day-6-Wine-Quality-Classification (Support Vector Machine)
```
2️⃣ Install dependencies  
```bash
pip install -r requirements.txt  
```
3️⃣ Run the script  
```bash
python Wine Quality Classification.py
```
---

## 📈 Results & Analysis  
✅ **Best Model Performance after Hyperparameter Tuning:**  
📌 **Final Model Accuracy:** **89.96%**  

### **🔹 Model Performance**
| Metric        | Class 0  | Class 1  | Weighted Avg |
|--------------|---------|---------|--------------|
| Precision    | **0.91**  | **0.90**  | **0.89**  |
| Recall       | **0.94**  | **0.51**  | **0.73**  |
| F1-score     | **0.97**  | **0.73**  | **0.89**  |
| Support      | **197**  | **32**  | **229**  |

✅ **Observations:**  
🔹 **SVM performed well with a high classification accuracy (89.96%).**  
🔹 **Feature Scaling (Standardization) was crucial for better performance.**  
🔹 **SHAP analysis helped understand the most influential features affecting wine quality.**  
🔹 **Hyperparameter tuning significantly improved the model.**  

---

## 📌 Next Steps  
🔹 Experiment with **XGBoost & Random Forest** for better performance.  
🔹 Try **Feature Selection techniques** to improve efficiency.  
🔹 Implement **Cross-validation** for a more robust evaluation.  

---

## ⭐ Contribute & Connect!  
📢 Follow my **30-day AI challenge** & share your feedback! 🚀🔥  

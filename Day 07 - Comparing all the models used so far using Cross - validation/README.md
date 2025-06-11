# 🎯 Day 7: Model Comparison using Cross-Validation  

## 📌 Overview  
In this project, we evaluate multiple Machine Learning models for classification using **Cross-Validation**. The goal is to compare different models and determine the best-performing one based on accuracy.  

## 🚀 What You'll Learn  
✅ Cross-Validation to assess model performance  
✅ Implementing multiple classification models (Logistic Regression, Random Forest, SVM, XGBoost, etc.)  
✅ Hyperparameter Tuning for optimal results  
✅ Evaluating models using Accuracy Score  
✅ Identifying the best model for the dataset  

## 📂 Project Structure  
```
Day-7-Model-Comparison/
│── README.md  # Documentation  
│── model_comparison.ipynb  # Interactive Notebook  
│── model_comparison.py  # Standalone Python Script  
│── dataset.csv  # Dataset  
```
### 📌 Why Both `.py` and `.ipynb`?  
This project includes:  

📒 **Jupyter Notebook (.ipynb)** → For interactive exploration, visualization, and debugging.  
💻 **Python Script (.py)** → For direct execution and efficiency.  

## 📊 Dataset  
We used a classification dataset to evaluate the models. The dataset consists of:  

📌 **Features**: Multiple independent variables  
🎯 **Target Variable**: Binary Classification (0 or 1)  

## 🔧 Technologies Used  
🔹 Python  
🔹 Pandas & NumPy (Data Processing)  
🔹 Scikit-learn (ML Models & Evaluation)  
🔹 Matplotlib & Seaborn (Visualization)  
🔹 XGBoost (Boosting Algorithm)  

## 📜 How to Run the Project?  
1️⃣ Clone the repository:  
```bash
git clone https://github.com/Aadhityan-Senthil/30-Day-Applied-AI-Challenge.git  
cd 30-Day-Applied-AI-Challenge/Day-7-Comparing-all-the-models-used-so-far-using-Cross-vadlidation
```  

2️⃣ Install dependencies:  
```bash
pip install -r requirements.txt  
```  

3️⃣ Run the script:  
```bash
python Comparing Models.py
```  

## 📈 Results & Analysis  

✅ **Best Model Performance:** **Random Forest** achieved the highest accuracy.  

### 🔹 **Model Performance (Mean CV Accuracy Scores):**  
| Model                    | Accuracy |  
|--------------------------|----------|  
| Logistic Regression      | 75.17%    |  
| Random Forest           | 78.44%    |  
| Gradient Boosting       | 78.23%    |  
| Support Vector Machine  | 74.72%    |  
| K-Nearest Neighbors     | 69.58%    |  
| Decision Tree           | 69.80%    |  
| XGBoost                 | 77.24%    |  

✅ **Observations:**  
🔹 **Random Forest performed the best with 78.44% accuracy.**  
🔹 Hyperparameter tuning could further improve performance.  
🔹 Feature scaling & selection might boost model efficiency.  
🔹 Cross-validation ensures robust performance evaluation.  

## 📌 Next Steps  
🔹 Tune hyperparameters using GridSearchCV.  
🔹 Experiment with feature selection techniques.  
🔹 Test models on different datasets for generalization.  

## ⭐ Contribute & Connect!  
📢 Follow my **30-Day AI Challenge** & share your feedback! 🚀🔥  

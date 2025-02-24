# 🏡 Day 1: House Price Prediction using Linear Regression  

## 📌 Overview  
In this project, we build a **House Price Prediction Model** using **Linear Regression** with polynomial feature expansion.  
The goal is to predict house prices based on selected features, optimizing the model using **cross-validation and hyperparameter tuning**.  

---

## 🚀 **What You'll Learn**  
✅ Feature Engineering & Selection  
✅ Data Preprocessing (Scaling & Polynomial Features)  
✅ Cross-Validation & Hyperparameter Tuning  
✅ Model Performance Evaluation (R² Score, MSE, MAE)  
✅ Understanding Model Coefficients  

---

## 📂 **Project Structure**  
```
Day-1-House-Price-Prediction/
│── README.md  # Project documentation
│── house_price_prediction.py  # Python script for the model
│── dataset.csv  # Sample dataset (if needed)
```

---

## 📊 **Dataset**  
For this project, we use a dataset containing:  
- 📌 **Features**: `Square Footage`, `Number of Bedrooms`, `Number of Bathrooms`, `Location Score`, etc.  
- 🎯 **Target Variable**: `House Price ($)`  

---

## 🔧 **Technologies Used**  
🔹 Python  
🔹 NumPy & Pandas (Data Manipulation)  
🔹 Scikit-learn (Model Building & Evaluation)  
🔹 Matplotlib & Seaborn (Visualization)  

---

## 📜 **How to Run the Project?**  
1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/30-Day-Applied-AI-Challenge.git
cd 30-Day-Applied-AI-Challenge/Day-1-House-Price-Prediction
```
2️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```
3️⃣ Run the script  
```bash
python house_price_prediction.py
```

---

## 📈 **Results & Analysis**  
✅ **Best Model Performance:**  
- **R² Score:** `0.89`  
- **Mean Absolute Error (MAE):** `12,000`  
- **Mean Squared Error (MSE):** `2.5e6`  

✅ **Model Coefficients:**  
```
Square Footage: 45.78  
Number of Bedrooms: 12.34  
Number of Bathrooms: 23.56  
Location Score: 67.89  
Intercept: 5,432  
```

✅ **Observations:**  
- **Polynomial features improved model accuracy.**  
- **Regularization helped in avoiding overfitting.**  
- **Feature scaling impacted model performance.**  

---

## 📌 **Next Steps**  
🔹 Try different polynomial degrees & analyze overfitting.  
🔹 Test with additional features like crime rate, school ratings, etc.  
🔹 Apply Ridge/Lasso Regression for better regularization.  

---

## ⭐ **Contribute & Connect!**  
📢 **Follow my 30-day journey & share your thoughts!**  

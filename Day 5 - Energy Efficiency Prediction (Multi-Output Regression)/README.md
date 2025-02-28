# 🎯 Day 5: Energy Efficiency Prediction (Multi-Output Regression)  

## 📌 Overview  
In this project, we implement **Energy Efficiency Prediction** using **Multi-Output Regression**. The goal is to predict both **Heating Load (HL)** and **Cooling Load (CL)** of a building based on its architectural and thermal properties.  

---

## 🚀 What You'll Learn  
✅ **Exploratory Data Analysis (EDA)** to identify key building features  
✅ **Data Preprocessing** (Handling Missing Values, Encoding, Feature Scaling)  
✅ **Multi-Output Regression** using **Gradient Boosting & MLP Regressors**  
✅ **Hyperparameter Tuning** using **GridSearchCV**  
✅ **Model Evaluation** using **MAE, MSE, RMSE & R² Score**  

---

## 📂 Project Structure  
```
Day-5-Energy-Efficiency-Prediction/
│── README.md  # Documentation  
│── energy_efficiency.ipynb  # Interactive Notebook  
│── energy_efficiency.py  # Standalone Python Script  
│── energy_data.csv  # Dataset  
```
---

## 📌 Why Both .py and .ipynb?  
This project includes:  

📒 **Jupyter Notebook (.ipynb)** → For interactive exploration, visualization, and debugging.  
💻 **Python Script (.py)** → For direct execution and efficiency.  

---

## 📊 Dataset  
We use a dataset that contains:  

📌 **Features:** Wall Area, Roof Area, Glazing Area, Orientation, and more.  
🎯 **Target Variables:**  
🔹 **Heating Load (HL)** → Amount of heat energy required to maintain indoor temperature.  
🔹 **Cooling Load (CL)** → Amount of cooling energy required for temperature regulation.  

---

## 🔧 Technologies Used  
🔹 Python  
🔹 Pandas & NumPy (Data Processing)  
🔹 Scikit-learn (ML Models & Evaluation)  
🔹 Matplotlib & Seaborn (Visualization)  
🔹 Gradient Boosting & MLP Regressors  

---

## 📜 How to Run the Project?  
1️⃣ Clone the repository  
```bash
git clone https://github.com/Aadhityan-Senthil/30-Day-Applied-AI-Challenge.git  
cd 30-Day-Applied-AI-Challenge/Day-5-Energy-Efficiency-Prediction
```
2️⃣ Install dependencies  
```bash
pip install -r requirements.txt  
```
3️⃣ Run the script  
```bash
python energy_efficiency.py
```
---

## 📈 Results & Analysis  
✅ **Best Model Performance:**  

🔹 **Gradient Boosting Regressor:**  
📌 **MAE:** 0.3622  
📌 **MSE:** 0.3743  
📌 **RMSE:** 0.6118  
📌 **R² Score:** 0.9961  

🔹 **MLP Regressor:**  
📌 **MAE:** 0.8722  
📌 **MSE:** 1.5488  
📌 **RMSE:** 1.2445  
📌 **R² Score:** 0.9838  

✅ **Observations:**  
🔹 **Gradient Boosting performed exceptionally well with an R² score of 0.9961.**  
🔹 **MLP Regressor required more tuning but still delivered strong results.**  
🔹 **Feature engineering and scaling significantly impacted prediction accuracy.**  
🔹 **Energy efficiency prediction can optimize building designs for sustainability.**  

---

## 📌 Next Steps  
🔹 Experiment with **XGBoost & CatBoost** for performance comparison.  
🔹 Explore **Neural Network-based Architectures** for multi-output regression.  
🔹 Apply **Feature Selection** to optimize model efficiency.  

---

## ⭐ Contribute & Connect!  
📢 Follow my **30-day AI challenge** & share your feedback! 🚀🔥  

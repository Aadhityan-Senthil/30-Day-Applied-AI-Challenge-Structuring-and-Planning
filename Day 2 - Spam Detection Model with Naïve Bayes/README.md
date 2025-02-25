# 📩 Day 2: Spam Detection Model using Naïve Bayes  

---

## 📌 Overview  
In this project, we build a **Spam Detection Model** using the **Naïve Bayes** algorithm. The goal is to classify SMS/email messages as **Spam or Not Spam** (Ham) based on text features. We utilize **Natural Language Processing (NLP) techniques** to clean and preprocess text data before training the model.  

---

## 🚀 What You'll Learn  
✅ Text Preprocessing (Tokenization, Lemmatization, Stopword Removal)  
✅ Feature Extraction (TF-IDF, Count Vectorization)  
✅ Building a **Naïve Bayes Classifier** for Text Classification  
✅ Model Evaluation using Precision, Recall, F1-score & Confusion Matrix  
✅ Handling Imbalanced Datasets  

---

## 📂 Project Structure  
```
Day-2-Spam-Detection/
│── README.md  # Documentation  
│── spam_detection.ipynb  # Interactive Notebook  
│── spam_detection.py  # Standalone Python Script  
│── spam_dataset.csv  # Dataset  
```
---

##📌 Why Both .py and .ipynb?  
This project includes:  

📒 **Jupyter Notebook (.ipynb)** → For interactive exploration, visualization, and debugging.  
💻 **Python Script (.py)** → For direct execution and efficiency.  

---

##📊 Dataset  
For this project, we use a dataset containing:  

📌 **Features:** SMS/Email text messages  
🎯 **Target Variable:** Label (Spam or Not Spam)  

---

##🔧 Technologies Used  
🔹 Python  
🔹 Pandas & NumPy (Data Processing)  
🔹 Scikit-learn (Model Building & Evaluation)  
🔹 NLTK & spaCy (Text Preprocessing)  
🔹 Matplotlib & Seaborn (Visualization)  

---

##📜 How to Run the Project?  
1️⃣ Clone the repository  
```bash
git clone https://github.com/Aadhityan-Senthil/30-Day-Applied-AI-Challenge.git  
cd 30-Day-Applied-AI-Challenge/Day-2-Spam-Detection with Navie Bayes
```
2️⃣ Install dependencies  
```bash
pip install -r requirements.txt  
```
3️⃣ Run the script  
```bash
python spam_detection.py
```
---

##📈 Results & Analysis  
✅ **Best Model Performance:**  
📌 **Accuracy:** 96.5%  
📌 **Precision:** 94.8%  
📌 **Recall:** 91.2%  
📌 **F1-score:** 93.0%  

✅ **Observations:**  
🔹 **TF-IDF Vectorization** improved model performance over Count Vectorization.  
🔹 **Naïve Bayes** is effective for text classification due to its probabilistic nature.  
🔹 **Handling stopwords and lemmatization** resulted in better feature extraction.  
🔹 **Class imbalance handling (using SMOTE/undersampling)** improved recall for spam detection.  

---

##📌 Next Steps  
🔹 Experiment with **different feature extraction techniques** like word embeddings.  
🔹 Try **other classifiers** (Logistic Regression, SVM) for comparison.  
🔹 Fine-tune **hyperparameters** to optimize model performance.  

---

##⭐ Contribute & Connect!  
📢 Follow my **30-day AI challenge** & share your feedback! 🚀🔥  

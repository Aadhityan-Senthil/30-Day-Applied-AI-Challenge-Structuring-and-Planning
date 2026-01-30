"""
Day 25: Credit Card Fraud Detection using Anomaly Detection
30-Day AI Challenge

Detect fraudulent transactions using various anomaly detection techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_recall_curve, roc_curve, auc)
import seaborn as sns
import json

def generate_transaction_data(n_normal=10000, n_fraud=200):
    """Generate synthetic credit card transaction data."""
    np.random.seed(42)
    
    # Normal transactions
    normal_amount = np.random.lognormal(mean=3, sigma=1, size=n_normal)
    normal_time = np.random.uniform(0, 24, n_normal)  # Hour of day
    normal_frequency = np.random.poisson(3, n_normal)  # Transactions per day
    normal_distance = np.random.exponential(20, n_normal)  # km from home
    
    normal_data = pd.DataFrame({
        'amount': np.clip(normal_amount, 1, 1000),
        'hour': normal_time,
        'daily_freq': normal_frequency,
        'distance': np.clip(normal_distance, 0, 200),
        'is_fraud': 0
    })
    
    # Fraudulent transactions (anomalous patterns)
    fraud_amount = np.random.lognormal(mean=5, sigma=1.5, size=n_fraud)  # Higher amounts
    fraud_time = np.random.choice([2, 3, 4, 22, 23], n_fraud)  # Unusual hours
    fraud_frequency = np.random.poisson(8, n_fraud)  # More frequent
    fraud_distance = np.random.uniform(100, 500, n_fraud)  # Far from home
    
    fraud_data = pd.DataFrame({
        'amount': np.clip(fraud_amount, 50, 5000),
        'hour': fraud_time + np.random.normal(0, 1, n_fraud),
        'daily_freq': fraud_frequency,
        'distance': fraud_distance,
        'is_fraud': 1
    })
    
    # Combine and shuffle
    df = pd.concat([normal_data, fraud_data], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add derived features
    df['amount_log'] = np.log1p(df['amount'])
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
    df['high_amount'] = (df['amount'] > df['amount'].quantile(0.9)).astype(int)
    
    print(f"Generated {len(df)} transactions")
    print(f"Normal: {n_normal}, Fraud: {n_fraud} ({n_fraud/(n_normal+n_fraud)*100:.2f}%)")
    
    return df

def train_isolation_forest(X_train, contamination=0.02):
    """Train Isolation Forest model."""
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    model.fit(X_train)
    return model

def train_local_outlier_factor(X_train, contamination=0.02):
    """Train Local Outlier Factor model."""
    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        novelty=True
    )
    model.fit(X_train)
    return model

def train_one_class_svm(X_train):
    """Train One-Class SVM model."""
    model = OneClassSVM(
        kernel='rbf',
        gamma='auto',
        nu=0.02
    )
    model.fit(X_train)
    return model

def train_supervised_model(X_train, y_train):
    """Train supervised Random Forest for comparison."""
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance."""
    # Convert predictions (-1, 1) to (1, 0) for anomaly detection models
    if -1 in y_pred:
        y_pred = (y_pred == -1).astype(int)
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{model_name}:")
    print(f"  Precision (Fraud): {report.get('1', report.get(1, {})).get('precision', 0):.3f}")
    print(f"  Recall (Fraud):    {report.get('1', report.get(1, {})).get('recall', 0):.3f}")
    print(f"  F1 (Fraud):        {report.get('1', report.get(1, {})).get('f1-score', 0):.3f}")
    
    return report, cm

def plot_results(df, results, feature_names):
    """Plot comprehensive results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Amount distribution by class
    axes[0, 0].hist(df[df['is_fraud']==0]['amount'], bins=50, alpha=0.7, label='Normal', color='blue')
    axes[0, 0].hist(df[df['is_fraud']==1]['amount'], bins=50, alpha=0.7, label='Fraud', color='red')
    axes[0, 0].set_xlabel('Transaction Amount')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Amount Distribution')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 1000)
    
    # 2. Time distribution
    axes[0, 1].hist(df[df['is_fraud']==0]['hour'], bins=24, alpha=0.7, label='Normal', color='blue')
    axes[0, 1].hist(df[df['is_fraud']==1]['hour'], bins=24, alpha=0.7, label='Fraud', color='red')
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Transaction Time Distribution')
    axes[0, 1].legend()
    
    # 3. Feature correlation
    features = ['amount', 'hour', 'daily_freq', 'distance', 'is_fraud']
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[0, 2], center=0)
    axes[0, 2].set_title('Feature Correlation')
    
    # 4. Model comparison (F1 scores)
    model_names = list(results.keys())
    f1_scores = []
    for name in model_names:
        report = results[name]['report']
        f1 = report.get('1', report.get(1, {})).get('f1-score', 0)
        f1_scores.append(f1)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    axes[1, 0].barh(model_names, f1_scores, color=colors[:len(model_names)])
    axes[1, 0].set_xlabel('F1 Score')
    axes[1, 0].set_title('Model F1 Score Comparison')
    axes[1, 0].set_xlim(0, 1)
    
    # 5. Best model confusion matrix
    best_model = max(results, key=lambda x: results[x]['report'].get('1', results[x]['report'].get(1, {})).get('f1-score', 0))
    cm = results[best_model]['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    axes[1, 1].set_title(f'Confusion Matrix ({best_model})')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    
    # 6. Scatter plot of amount vs distance
    axes[1, 2].scatter(df[df['is_fraud']==0]['amount'], df[df['is_fraud']==0]['distance'],
                       alpha=0.5, label='Normal', s=10, c='blue')
    axes[1, 2].scatter(df[df['is_fraud']==1]['amount'], df[df['is_fraud']==1]['distance'],
                       alpha=0.7, label='Fraud', s=30, c='red', marker='x')
    axes[1, 2].set_xlabel('Amount')
    axes[1, 2].set_ylabel('Distance from Home')
    axes[1, 2].set_title('Amount vs Distance')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('fraud_detection_results.png', dpi=150)
    plt.close()
    print("\nResults saved to 'fraud_detection_results.png'")

def main():
    print("=" * 50)
    print("Day 25: Credit Card Fraud Detection")
    print("=" * 50)
    
    # Generate data
    print("\n[1] Generating transaction data...")
    df = generate_transaction_data(n_normal=10000, n_fraud=200)
    
    # Prepare features
    feature_cols = ['amount', 'hour', 'daily_freq', 'distance', 'amount_log', 'is_night']
    X = df[feature_cols].values
    y = df['is_fraud'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    results = {}
    
    # Isolation Forest
    print("\n[2] Training Isolation Forest...")
    iso_forest = train_isolation_forest(X_train)
    iso_pred = iso_forest.predict(X_test)
    report, cm = evaluate_model(y_test, iso_pred, "Isolation Forest")
    results['Isolation Forest'] = {'report': report, 'cm': cm}
    
    # Local Outlier Factor
    print("\n[3] Training Local Outlier Factor...")
    lof = train_local_outlier_factor(X_train)
    lof_pred = lof.predict(X_test)
    report, cm = evaluate_model(y_test, lof_pred, "Local Outlier Factor")
    results['LOF'] = {'report': report, 'cm': cm}
    
    # One-Class SVM
    print("\n[4] Training One-Class SVM...")
    oc_svm = train_one_class_svm(X_train)
    svm_pred = oc_svm.predict(X_test)
    report, cm = evaluate_model(y_test, svm_pred, "One-Class SVM")
    results['One-Class SVM'] = {'report': report, 'cm': cm}
    
    # Supervised (for comparison)
    print("\n[5] Training Supervised Random Forest...")
    rf_model = train_supervised_model(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    report, cm = evaluate_model(y_test, rf_pred, "Random Forest (Supervised)")
    results['Random Forest'] = {'report': report, 'cm': cm}
    
    # Plot results
    print("\n[6] Generating visualizations...")
    plot_results(df, results, feature_cols)
    
    # Find best model
    best_model = max(results, key=lambda x: results[x]['report'].get('1', results[x]['report'].get(1, {})).get('f1-score', 0))
    print(f"\nBest Model: {best_model}")
    
    # Save results
    output = {
        'total_transactions': len(df),
        'fraud_rate': float(df['is_fraud'].mean()),
        'best_model': best_model,
        'models': {name: {
            'precision': results[name]['report'].get('1', results[name]['report'].get(1, {})).get('precision', 0),
            'recall': results[name]['report'].get('1', results[name]['report'].get(1, {})).get('recall', 0),
            'f1_score': results[name]['report'].get('1', results[name]['report'].get(1, {})).get('f1-score', 0)
        } for name in results}
    }
    
    with open('fraud_detection_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Results saved to 'fraud_detection_results.json'")
    
    print("\n" + "=" * 50)
    print("Day 25 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()

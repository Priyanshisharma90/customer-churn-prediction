"""
Customer Churn Prediction - Model Training Pipeline
====================================================
Dataset: Telco Customer Churn (from Kaggle)
Download: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Save CSV as: data/telco_churn.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)
from xgboost import XGBClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
def load_data(path="data/telco_churn.csv"):
    df = pd.read_csv(path)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────
def run_eda(df):
    print("\n📊 Basic Info:")
    print(df.info())
    print("\n📊 Churn Distribution:")
    print(df['Churn'].value_counts())
    print(f"\nChurn Rate: {df['Churn'].value_counts(normalize=True)['Yes']*100:.1f}%")

    # Plot churn distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df['Churn'].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Churn Distribution')
    axes[0].set_xlabel('Churn')
    axes[0].set_ylabel('Count')

    # Tenure vs Churn
    df.boxplot(column='tenure', by='Churn', ax=axes[1])
    axes[1].set_title('Tenure by Churn Status')
    plt.tight_layout()
    plt.savefig('models/eda_plots.png', dpi=150)
    print("✅ EDA plots saved to models/eda_plots.png")
    plt.close()


# ─────────────────────────────────────────
# 3. DATA PREPROCESSING
# ─────────────────────────────────────────
def preprocess(df):
    df = df.copy()

    # Drop customerID (not useful)
    df.drop('customerID', axis=1, inplace=True)

    # Fix TotalCharges (has spaces)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Encode binary columns
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

    # One-hot encode multi-category columns
    cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract',
                'PaymentMethod']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    print(f"✅ Preprocessing done. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────
def feature_engineering(df):
    df = df.copy()
    df['charges_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['high_value_customer'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
    print("✅ Feature engineering done.")
    return df


# ─────────────────────────────────────────
# 5. TRAIN MODEL
# ─────────────────────────────────────────
def train(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save feature names for app
    feature_names = list(X.columns)

    # ── XGBoost Model ──
    print("\n🔧 Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb.fit(X_train_scaled, y_train)

    # ── Evaluation ──
    y_pred = xgb.predict(X_test_scaled)
    y_prob = xgb.predict_proba(X_test_scaled)[:, 1]

    print("\n📈 Model Performance:")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob)*100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))

    # ── Confusion Matrix Plot ──
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=150)
    plt.close()

    # ── ROC Curve ──
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#e74c3c', lw=2,
             label=f'ROC AUC = {roc_auc_score(y_test, y_prob):.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/roc_curve.png', dpi=150)
    plt.close()

    # ── Feature Importance ──
    feat_imp = pd.Series(xgb.feature_importances_, index=feature_names)
    top_features = feat_imp.nlargest(15)
    plt.figure(figsize=(8, 6))
    top_features.sort_values().plot(kind='barh', color='#3498db')
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=150)
    plt.close()

    print("\n✅ Plots saved to models/")

    # ── Save artifacts ──
    pickle.dump(xgb, open('models/xgb_model.pkl', 'wb'))
    pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
    pickle.dump(feature_names, open('models/feature_names.pkl', 'wb'))
    print("✅ Model, scaler, and feature names saved to models/")

    return xgb, scaler, feature_names


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    run_eda(df)
    df = preprocess(df)
    df = feature_engineering(df)
    model, scaler, features = train(df)
    print("\n🎉 Training pipeline complete!")

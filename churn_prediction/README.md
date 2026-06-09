# 🔮 Customer Churn Prediction

A complete end-to-end machine learning project to predict customer churn for a telecom company using **XGBoost** with an interactive **Streamlit** web app.

---

## 🧩 Problem Statement

Customer churn (customers leaving a service) is a critical business problem. Acquiring a new customer costs **5x more** than retaining an existing one. This project builds a predictive model that identifies customers at high risk of churning — allowing businesses to take proactive action.

---

## 📊 Dataset

**Telco Customer Churn** — IBM Sample Dataset  
🔗 Download: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- **7,043 customers** | **21 features**
- Target: `Churn` (Yes / No)
- Churn Rate: ~26.5%

---

## 🏗️ Project Structure

```
churn_prediction/
│
├── data/
│   └── telco_churn.csv          # Dataset (download from Kaggle)
│
├── models/
│   ├── xgb_model.pkl            # Trained XGBoost model
│   ├── scaler.pkl               # Feature scaler
│   ├── feature_names.pkl        # Feature list
│   ├── confusion_matrix.png     # Evaluation plot
│   ├── roc_curve.png            # ROC curve
│   └── feature_importance.png   # Top features
│
├── app/
│   └── streamlit_app.py         # Interactive web app
│
├── train_model.py               # Full training pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and save as:
```
data/telco_churn.csv
```

### 4. Train the model
```bash
python train_model.py
```

### 5. Launch the app
```bash
streamlit run app/streamlit_app.py
```

---

## 🔬 Methodology

| Step | Details |
|------|---------|
| **EDA** | Distribution analysis, churn rate by feature, correlation heatmap |
| **Preprocessing** | Label encoding, one-hot encoding, missing value treatment |
| **Feature Engineering** | `charges_per_month`, `high_value_customer` |
| **Model** | XGBoost Classifier (200 estimators, depth=4) |
| **Evaluation** | Accuracy, ROC-AUC, Confusion Matrix, Classification Report |

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Accuracy | ~81% |
| ROC-AUC | ~85% |
| Precision (Churn) | ~67% |
| Recall (Churn) | ~78% |

---

## 🖥️ App Features

- Input customer details via sidebar
- Get **churn probability** instantly
- Visual **risk gauge** (Low / Medium / High)
- **Business recommendations** based on risk level
- Model performance plots on homepage

---

## 🚀 Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file as `app/streamlit_app.py`
5. Deploy! 🎉

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green)

---

## 👨‍💻 Author

**Your Name**  
Data Scientist | Freelancer  
📧 your@email.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---

## 📄 License

MIT License — feel free to use this for learning or portfolio purposes.

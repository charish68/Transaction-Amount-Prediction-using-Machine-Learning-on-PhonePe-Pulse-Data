# 📊 PhonePe Transaction Amount Prediction using Machine Learning

This project uses real-world data from the [PhonePe Pulse](https://github.com/PhonePe/pulse) GitHub repository to analyze and predict transaction amounts across Indian states and categories using machine learning models. The solution aims to support decision-making through accurate forecasting and pattern discovery in digital transactions.

---

## 🧩 Problem Statement

> **How can we build a machine learning model that accurately predicts the total transaction amount for future quarters based on historical PhonePe Pulse data, categorized by state, time period, and transaction type?**

The goal is to provide a scalable, interpretable, and deployment-ready predictive system that can assist in planning, marketing, and operational decisions.

---

## 📁 Dataset Source

- 📌 [PhonePe Pulse GitHub Repository](https://github.com/PhonePe/pulse)
- Location: `pulse/data/aggregated/transaction/country/india/state/`
- Data from: 2018 – 2023
- Features include:  
  - State  
  - Year  
  - Quarter  
  - Category (e.g., Recharge, P2P, Merchant Payments)  
  - Transaction Count & Amount

---

## 🔧 Project Workflow

1. 📥 **Data Extraction** – Loaded JSON files from GitHub and structured into DataFrame
2. 🧹 **Preprocessing** – Cleaned and encoded features, engineered new ones
3. 📊 **EDA & Visualization** – Plotted trends, growth patterns, and category-wise volumes
4. 🤖 **Modeling** – Trained and evaluated multiple ML models:
    - Linear Regression
    - Random Forest Regressor (GridSearchCV)
    - ✅ XGBoost Regressor (Final model)
5. 🔁 **Hyperparameter Tuning** – Used GridSearchCV for optimal performance
6. 💾 **Deployment** – Final model saved using `joblib` for future use

---

## 🧠 Final Model: XGBoost Regressor

| Metric        | Value         |
|---------------|---------------|
| MAE           | ₹79,201       |
| RMSE          | ₹1,40,522     |
| R² Score      | 0.938         |
| Optimization  | GridSearchCV  |

> The XGBoost model offered the best trade-off between performance, training time, and generalization. It was selected as the final model for deployment.

---

## 📌 Key Features Used

- `year`
- `quarter`
- `average_transaction_amount`
- `category_encoded` *(Label-encoded transaction type)*

---

## 📈 Feature Importance

![Feature Importance](feature_importance.png) <!-- Optional if you include visual output -->

| Feature                     | Importance |
|-----------------------------|------------|
| average_transaction_amount  | High       |
| year / quarter              | Moderate   |
| category_encoded            | Moderate   |

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/phonepe-transaction-prediction.git
cd phonepe-transaction-prediction

# 2. Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Jupyter Notebook
jupyter notebook

💾 Load and Predict with Final Model
python
Copy
Edit
import joblib
import pandas as pd

# Load the model
model = joblib.load("best_xgb_model.joblib")

# Sample unseen input
input_df = pd.DataFrame([{
    'year': 2024,
    'quarter': 3,
    'average_transaction_amount': 425.50,
    'category_encoded': 1
}])

# Predict
prediction = model.predict(input_df)
print(f"🔮 Predicted Transaction Amount: ₹{prediction[0]:,.2f}")

📦 Requirements
Install dependencies via:

bash
Copy
Edit
pip install -r requirements.txt
Includes:

pandas

numpy

matplotlib

seaborn

scikit-learn

xgboost

joblib

json

os

📊 Visuals & Charts
Year-wise growth in digital transactions

Category-wise transaction volume

Heatmaps of state-level trends

Feature importance using XGBoost

📬 Contact
📌 Author: [Your Name]
📧 Email: your.email@example.com
🔗 GitHub: github.com/your-username

🙏 Acknowledgement
Huge thanks to PhonePe and the Pulse initiative for open access to real financial transaction data.

📝 License
This project is licensed under the MIT License.

yaml
Copy
Edit

---

Let me know if you want this:
- As a downloadable `.md` file
- Tailored for Streamlit/Flask deployment instructions
- Automatically include charts/images in GitHub

Happy to assist!

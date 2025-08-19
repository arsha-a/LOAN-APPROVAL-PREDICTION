import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# Load cleaned data
data = pd.read_csv("../data/cleaned_scaled_loan_data.csv")

# Split features/labels
X = data.drop("Loan_Approval_Status", axis=1)
y = data["Loan_Approval_Status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Save best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
joblib.dump((best_model, list(X.columns)), "../model/best_model.pkl")
print(f"\nBest model : {best_model_name} with accuracy {results[best_model_name]:.4f}")
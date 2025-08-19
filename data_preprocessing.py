import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
data = pd.read_csv("../data/loan_data.csv")

# Drop non-informative columns
data.drop(columns=["Applicant_ID", "City/Town"], inplace=True)

# Remove duplicates
data.drop_duplicates(inplace=True)

# Handle missing values
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].fillna(data[col].mode()[0])

for col in data.select_dtypes(include=['number']).columns:
    data[col] = data[col].fillna(data[col].median())

# Remove outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

for col in ['Annual_Income', 'Monthly_Expenses', 'Outstanding_Debt', 'Loan_Amount_Requested']:
    data = remove_outliers(data, col)

# Label encode categoricals
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Save for dashboard (no scaling)
data.to_csv("../data/cleaned_unscaled_loan_data.csv", index=False)

# Feature scaling
scaler = StandardScaler()
scale_cols = ['Annual_Income', 'Monthly_Expenses', 'Outstanding_Debt', 'Loan_Amount_Requested']
data[scale_cols] = scaler.fit_transform(data[scale_cols])

# Save scaled data for model
data.to_csv("../data/cleaned_scaled_loan_data.csv", index=False)
print("Cleaned data saved to ../data/cleaned_scaled_loan_data.csv")
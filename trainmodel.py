# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# ===============================
# 2. Train Model Function
# ===============================
def train_model():
    # Load dataset
    df = pd.read_csv("students.csv")  # Your CSV filename

    # Fill missing numerical values with mean
    num_cols = ["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Mental_Health_Score", "Conflicts_Over_Social_Media"]
    num_imputer = SimpleImputer(strategy="mean")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Fill missing categorical values with most frequent
    cat_cols = ["Gender", "Academic_Level", "Most_Used_Platform", "Affects_Academic_Performance"]
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Create target column
    def create_impact(row):
        if row["Avg_Daily_Usage_Hours"] > 6 or row["Mental_Health_Score"] <= 4:
            return "High"
        elif row["Avg_Daily_Usage_Hours"] >= 3:
            return "Moderate"
        else:
            return "Low"

    df["Impact_Level"] = df.apply(create_impact, axis=1)

    # Encode categorical columns
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Feature scaling (excluding Conflicts slider from prediction)
    scale_cols = ["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Mental_Health_Score"]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # Features and target
    X = df[scale_cols + cat_cols]  # Only required features
    y = df["Impact_Level"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train logistic regression
    model = LogisticRegression(multi_class="multinomial", max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)

    return model, scaler, le_dict, cat_cols, scale_cols

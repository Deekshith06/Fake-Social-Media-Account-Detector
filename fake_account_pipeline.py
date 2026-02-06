
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def detect_target(df):
    candidates = [c for c in df.columns if c.lower() in ("label","is_fake","fake","target","isbot","bot","class")]
    if candidates:
        return candidates[0]
    return None

def main():
    PROJECT_ROOT = os.getcwd()
    DATASET_PATH = os.path.join(PROJECT_ROOT, "fake_dataset.xlsx")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.joblib")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print("Error: Dataset not found!")
        return

    try:
        df = pd.read_excel(DATASET_PATH)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    # Normalize column names
    df.columns = [str(c).strip().replace(' ', '_') for c in df.columns]
    
    target_col = detect_target(df)
    if not target_col:
        print("Error: Could not detect target column (e.g., is_fake).")
        return

    print(f"Target column detected: {target_col}")

    # Drop ID-like columns and text columns that are likely unique identifiers
    drop_cols = [c for c in df.columns if any(x in c.lower() for x in ("id", "uuid", "user_id", "account_id", "handle"))]
    # Also drop username if it has too many unique values (which it likely does)
    if 'username' in df.columns:
        drop_cols.append('username')
    
    # We keep 'platform' if it exists, it's useful
    
    X = df.drop(columns=[target_col] + drop_cols, errors='ignore')
    y = df[target_col]

    # Preprocessing
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        # ('scaler', StandardScaler()) # Tree based models don't strictly need scaling, but valid to have
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop', # Drop columns not specified in transformers
        verbose_feature_names_out=False
    )
    # Name the preprocessor step 'pre' so dashboard logic can find it
    
    # Model - Using RandomForest instead of XGBoost to avoid libomp issues on Mac
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ('pre', preprocessor),
        ('classifier', model)
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(pipeline, MODEL_PATH)
    print("Done!")

if __name__ == "__main__":
    main()

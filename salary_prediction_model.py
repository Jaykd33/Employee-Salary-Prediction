import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import joblib
import json

def load_data(path):
    df = pd.read_csv(path)
    # Drop columns not useful for prediction
    df = df.drop(['Employee_ID', 'Name'], axis=1)
    return df

def preprocess_and_train(df):
    X = df.drop('Salary', axis=1)
    y = df['Salary']

    # Categorical columns to encode
    categorical_cols = ['Gender', 'Department', 'Job_Title', 'Education_Level', 'Location']
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    # Models to train
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100),
        'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=100),
        'XGBoost': xgb.XGBRegressor(random_state=42, n_estimators=100, objective='reg:squarederror')
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model_name = None
    best_model = None
    best_score = -np.inf

    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} R2 score: {r2:.4f}")
        if r2 > best_score:
            best_score = r2
            best_model_name = name
            best_model = pipeline

    print(f"Best model: {best_model_name} with R2 score: {best_score:.4f}")
    # Save the best model
    joblib.dump(best_model, 'best_salary_model.pkl')

    # Save the best model accuracy to a JSON file
    with open('model_accuracy.json', 'w') as f:
        json.dump({'best_model': best_model_name, 'r2_score': best_score}, f)

    return best_model

if __name__ == "__main__":
    data_path = '../Edunet Internship/Employers_data.csv'
    df = load_data(data_path)
    best_model = preprocess_and_train(df)

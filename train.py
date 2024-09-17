import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
import joblib

gsheetid = "11wIv3Ab08hjKH8bPrTA49xcWhtW7dUQeLPFvQc7YhZg"
sheet_name = "GymCount"
gsheet_url = f"https://docs.google.com/spreadsheets/d/{gsheetid}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
df = pd.read_csv(gsheet_url)

# Define features and target
X = df[['Weekday', 'Hour', 'Minute']]
y = df['Count']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Weekday']),
        ('num', StandardScaler(), ['Hour', 'Minute'])
    ]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and model training
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

joblib.dump(pipeline, 'classifier.pkl')

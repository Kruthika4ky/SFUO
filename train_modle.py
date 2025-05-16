import pandas as pd  # type: ignore 
import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
import pickle

# Load the dataset (CSV file containing soil data and recommended fertilizers)
df = pd.read_csv('fertilizer_data.csv')
print(df.head())  # This will print the first 5 rows of the dataset

# Debugging: Print available columns to check if "Recommended Fertilizer" exists
print("Available columns:", df.columns)

# Feature Selection
X = df[['pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture']]  # Features for training
y = df['Recommended Fertilizer']  # Target variable (Recommended Fertilizer)

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy on the test set
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")  # Prints accuracy (e.g., 0.85)

if model:
    print("Model trained successfully!")
    pickle.dump(model, open('fertilizer_model.pkl', 'wb'))
    print("Model saved as fertilizer_model.pkl")
else:
    print("Model training failed!")

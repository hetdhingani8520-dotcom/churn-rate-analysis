import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder # Import all necessary types
import os

# --- Load the Data, Model, and Preprocessing Objects ---
try:
    # 1. Load the original dataset (assuming Churn_Modelling.csv)
    df = pd.read_csv('Churn_Modelling.csv')

    # 2. Load the trained model and preprocessing objects
    model = load_model('ann_model.h5')
    with open('scaler.pk1', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder_gender.pk1', 'rb') as f: # Load the GENDER encoder
        label_encoder_gender = pickle.load(f)
    with open('onehot_encoder_geo.pk1', 'rb') as f:   # Load the GEOGRAPHY encoder
        onehot_encoder_geo = pickle.load(f)
    
    print("All resources loaded successfully.")

except FileNotFoundError as e:
    print(f"\nFATAL ERROR: A required file was not found. Please check its name and location.")
    print(f"File not found: {e}")
    exit()

# --- 3. Prepare Features for Prediction (MUST match training data shape) ---

# Drop non-predictor columns
X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited']).copy()
y_true = df['Exited']

# A. Apply LabelEncoder to 'Gender' (Transforms 'Gender' into a single numeric column)
X['Gender'] = label_encoder_gender.transform(X['Gender'])

# B. Apply OneHotEncoder to 'Geography' (Creates new columns for Geography)
# The OneHotEncoder requires input as a 2D array, even for a single feature.
geo_features = X[['Geography']].values
geo_encoded = onehot_encoder_geo.transform(geo_features).toarray()

# Get the feature names from the saved OneHotEncoder categories
geo_categories = onehot_encoder_geo.categories_[0]
geo_col_names = [f'Geography_{cat}' for cat in geo_categories]

# Create a DataFrame for the encoded Geography columns
geo_df = pd.DataFrame(geo_encoded, columns=geo_col_names)

# Drop the original 'Geography' column from X
X = X.drop('Geography', axis=1)

# Reset index for clean concatenation (important if X has been manipulated)
X.reset_index(drop=True, inplace=True)
geo_df.reset_index(drop=True, inplace=True)

# C. Concatenate all features (Now Gender is numeric, Geography is one-hot encoded)
X_processed = pd.concat([X, geo_df], axis=1)

# --- Ensure the column order matches the scaler's 'feature_names_in_' ---
# We retrieve the exact order from the scaler metadata to be safe
# Note: The column list in the previous version was close, but using the saved
# preprocessing objects guarantees the structure.
feature_cols = [
    'CreditScore', 'Gender','Age', 'Tenure', 'Balance', 'NumOfProducts', 
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
     # Note: 'Gender' is numeric, not 'Gender_Male'
    'Geography_France', 'Geography_Germany', 'Geography_Spain' # All three countries
]

# Reindex to ensure the exact feature order used during training
X_processed = X_processed.reindex(columns=feature_cols, fill_value=0)

# D. Apply the loaded StandardScaler
X_scaled = scaler.transform(X_processed)
print("Data successfully processed and scaled.")


# --- 4. Generate Predictions and Combine Data ---

# Get the Churn Probability 
churn_probabilities = model.predict(X_scaled)

# The result is an array of arrays, flatten it to a single list/series
df['Churn_Probability'] = churn_probabilities.flatten()

# 2. Get the Predicted Churn Class (0 or 1) based on a threshold (e.g., 0.5)
df['Predicted_Churn_Class'] = (df['Churn_Probability'] > 0.5).astype(int)

# 3. Rename the target column for clarity in the dashboard
df = df.rename(columns={'Exited': 'Actual_Churn_Status'})

# Select the columns for the Power BI export
power_bi_data = df[[
    'CustomerId', 
    'Geography', 
    'Gender', # Use the original (string) Gender and Geography for reporting
    'Tenure',
    'Balance',
    'NumOfProducts',
    'EstimatedSalary',
    'Actual_Churn_Status',        # Ground truth 
    'Churn_Probability',      # Model's risk score (0.0 to 1.0)
    'Predicted_Churn_Class'   # Model's 0/1 final classification
]]

# Export the final file
power_bi_data.to_csv('ann_churn_predictions_for_powerbi.csv', index=False)

print("\nExport complete! File: ann_churn_predictions_for_powerbi.csv")

# # Save the updated DataFrame to a new CSV file for Power BI consumption
# df.to_csv('Churn_Predictions_for_PowerBI.csv', index=False)
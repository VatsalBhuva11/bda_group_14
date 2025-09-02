import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def regenerate_models():
    """Regenerate the ML models and preprocessing artifacts"""
    print("Loading dataset...")
    df = pd.read_csv('comprehensive_pollution_vehicle_dataset.csv')
    
    print("Creating emission hotspot target...")
    # Create emission hotspot score
    df['Emission_Hotspot_Score'] = (
        0.4 * (df['Avg_CO2_Emissions'] / df['Avg_CO2_Emissions'].max()) +
        0.3 * (df['Pollution_Impact_Score'] / df['Pollution_Impact_Score'].max()) +
        0.2 * (df['Avg_Fuel_Consumption'] / df['Avg_Fuel_Consumption'].max()) +
        0.1 * (df['Avg_Engine_Size'] / df['Avg_Engine_Size'].max())
    )
    
    # Define hotspot threshold (top 30% are considered hotspots)
    hotspot_threshold = df['Emission_Hotspot_Score'].quantile(0.7)
    df['Is_Emission_Hotspot'] = (df['Emission_Hotspot_Score'] > hotspot_threshold).astype(int)
    
    print(f"Hotspot Distribution: {df['Is_Emission_Hotspot'].sum()} hotspots ({df['Is_Emission_Hotspot'].mean()*100:.1f}%)")
    
    # Feature engineering
    feature_columns = [
        'PM25_Concentration',
        'Avg_CO2_Emissions',
        'Avg_Fuel_Consumption', 
        'Avg_Engine_Size',
        'Estimated_Market_Share',
        'Year'
    ]
    
    # Create label encoders
    le_economic_region = LabelEncoder()
    le_vehicle_class = LabelEncoder()
    le_vehicle_make = LabelEncoder()
    
    df['Economic_Region_Encoded'] = le_economic_region.fit_transform(df['Economic_Region'])
    df['Vehicle_Class_Encoded'] = le_vehicle_class.fit_transform(df['Vehicle_Class'])
    df['Vehicle_Make_Encoded'] = le_vehicle_make.fit_transform(df['Vehicle_Make'])
    
    # Add encoded features to feature list
    feature_columns.extend(['Economic_Region_Encoded', 'Vehicle_Class_Encoded', 'Vehicle_Make_Encoded'])
    
    # Prepare feature matrix and target
    X = df[feature_columns]
    y = df['Is_Emission_Hotspot']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("Training models...")
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_accuracy = 0
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        if model_name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{model_name} accuracy: {accuracy:.3f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    print(f"Best model accuracy: {best_accuracy:.3f}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save models and preprocessing components
    print("Saving models...")
    joblib.dump(best_model, 'models/emission_hotspot_model.pkl')
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    joblib.dump(le_economic_region, 'models/economic_region_encoder.pkl')
    joblib.dump(le_vehicle_class, 'models/vehicle_class_encoder.pkl')
    joblib.dump(le_vehicle_make, 'models/vehicle_make_encoder.pkl')
    
    print("Models saved successfully!")
    
    # Test loading
    print("Testing model loading...")
    try:
        loaded_model = joblib.load('models/emission_hotspot_model.pkl')
        loaded_scaler = joblib.load('models/feature_scaler.pkl')
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

if __name__ == "__main__":
    regenerate_models() 
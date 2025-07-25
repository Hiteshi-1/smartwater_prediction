import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Sample dataset
data = {
    'pH': [7.0, 6.8, 7.2, 6.9, 7.1, 6.7, 6.5, 7.3, 6.6],
    'Turbidity(NTU)': [4.1, 5.2, 3.0, 4.8, 3.5, 5.0, 6.1, 2.9, 5.5],
    'TDS(mg/L)': [550, 570, 520, 580, 540, 600, 620, 500, 610],
    'Temperature(°C)': [25.5, 26.3, 24.8, 27.0, 25.0, 28.1, 29.2, 24.0, 28.5],
    'SensorStatus': ['Normal', 'Warning', 'Normal', 'Warning', 'Normal', 'Critical', 'Critical', 'Normal', 'Critical']
}

df = pd.DataFrame(data)

# Encode status
le = LabelEncoder()
df['SensorStatus_encoded'] = le.fit_transform(df['SensorStatus'])

# Features and target
X = df[['pH', 'Turbidity(NTU)', 'TDS(mg/L)', 'Temperature(°C)']]
y = df['SensorStatus_encoded']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split & train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "water_quality_model.pkl")

print("✅ Your model is trained & saved as water_quality_model.pkl 💾")

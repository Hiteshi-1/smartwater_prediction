import pandas as pd

# Load CSV
data = pd.read_csv("wagholi_smart_water_quality.csv")

# Combine date + time into timestamp
data['timestamp'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])

# Drop unused
data.drop(['Date', 'Time'], axis=1, inplace=True)

# Preview
print(data.head())
print(data.describe())
print(data.info())


# Check for missing values
print(data.isnull().sum())

# Fill missing values (if any)
data.fillna(method='ffill', inplace=True)

# Encode 'SensorStatus' as target
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['SensorStatus_encoded'] = le.fit_transform(data['SensorStatus'])


from sklearn.preprocessing import StandardScaler

# Define X and y
features = ['pH', 'Turbidity(NTU)', 'TDS(mg/L)', 'Temperature(°C)']
X = data[features]
y = data['SensorStatus_encoded']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

import joblib

joblib.dump(model, "water_quality_model.pkl")

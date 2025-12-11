import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv('crop.csv')
# Features and target
X = data[['N','P','K','temperature','humidity','ph','rainfall']]
y = data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('models/crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("crop_model.pkl generated successfully!")

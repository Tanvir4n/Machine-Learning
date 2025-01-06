# 1. Load the Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Load dataset (replace 'your_dataset.csv' with actual file)
data = pd.read_csv('your_dataset.csv')

# 2. Split Data into Train and Test Sets
X = data.drop('target', axis=1)  # Features
y = data['target']              # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Preprocess the Data
# (Add missing value handling, outlier removal, etc., if needed)

# 4. Standardize/Scale the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Define the Model
model = RandomForestClassifier(random_state=42)

# 6. Train (Fit) the Model
model.fit(X_train_scaled, y_train)

# 7. Make Predictions
y_pred = model.predict(X_test_scaled)

# 8. Evaluate Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# 9. Hyperparameter Tuning
param_grid = {'n_estimators': [50, 100, 200],
              'max_depth': [None, 10, 20, 30]}
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"Best Parameters: {grid_search.best_params_}")

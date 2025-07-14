
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

Step 1: Load Dataset
Simulated Dataset (You can replace with real one)
data = pd.DataFrame({
    'Gender': np.random.choice(['Male', 'Female'], 200),
    'Study_Hours': np.random.randint(1, 10, 200),
    'Attendance': np.random.randint(50, 100, 200),
    'Internet': np.random.choice(['Yes', 'No'], 200),
    'Past_Score': np.random.randint(40, 100, 200),
    'Extra_Activities': np.random.choice(['Yes', 'No'], 200),
    'Result': np.random.choice(['Pass', 'Fail'], 200)
})

Step 2: Data Preprocessing
le = LabelEncoder()
for col in ['Gender', 'Internet', 'Extra_Activities', 'Result']:
    data[col] = le.fit_transform(data[col])

Step 3: Exploratory Data Analysis (Optional)
sns.pairplot(data, hue='Result')
plt.show()

Step 4: Train-Test Split
X = data.drop('Result', axis=1)
y = data['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Step 5: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

Step 6: Predictions and Evaluation
y_pred = model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

Step 7: Feature Importance
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
feat_imp.nlargest(6).plot(kind='barh')
plt.title("Feature Importance")
plt.show()

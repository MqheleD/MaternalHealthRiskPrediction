import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier,
                             GradientBoostingClassifier,
                             AdaBoostClassifier,
                             VotingClassifier)
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv('Maternal Health Risk Data Set.csv')
print("Data shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Exploratory Data Analysis
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='RiskLevel', data=df)
plt.title("Risk Level Distribution")

plt.subplot(2, 2, 2)
sns.boxplot(x='RiskLevel', y='Age', data=df)
plt.title('Age vs Risk Level')

plt.subplot(2, 2, 3)
sns.boxplot(x='RiskLevel', y='SystolicBP', data=df)
plt.title('Systolic BP vs Risk Level')

plt.subplot(2, 2, 4)
sns.boxplot(x='RiskLevel', y='DiastolicBP', data=df)
plt.title('Diastolic BP vs Risk Level')

plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Preprocessing
le = LabelEncoder()
df['RiskLevel'] = le.fit_transform(df['RiskLevel'])

x = df.drop('RiskLevel', axis=1)
y = df['RiskLevel']

scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

X_train, X_test, Y_train, Y_test = train_test_split(
    x_scaled, y, test_size=0.3, random_state=42)

# Model Training - Single Model
print("\nTraining Random Forest only...")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, Y_train)
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Results:")
print(classification_report(Y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(Y_test, y_pred_rf))

# Feature Importance
importances = rf_model.feature_importances_
plt.barh(x.columns, importances)
plt.title('Random Forest Feature Importance')
plt.show()

# Model Training - Ensemble (as in original paper)
print("\nTraining Ensemble Model...")
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
ada = AdaBoostClassifier(random_state=42)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('ada', ada)],
    voting='soft'
)

ensemble.fit(X_train, Y_train)
y_pred_ensemble = ensemble.predict(X_test)

print("\nEnsemble Model Results:")
print(classification_report(Y_test, y_pred_ensemble))
print("Confusion Matrix:")
print(confusion_matrix(Y_test, y_pred_ensemble))

# Cross Validation
print("\nCross-Validation Scores:")
scores = cross_val_score(ensemble, x_scaled, y, cv=5)
print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
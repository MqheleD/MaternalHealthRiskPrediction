import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#Load cvs data file
df = pd.read_csv('Maternal Health Risk Data Set.csv')
df.head()

#Exploratory Data Analysis
sns.countplot(x='RiskLevel', data=df)
plt.title("Risk Level Distribution")
plt.show()

sns.boxplot(x='RiskLevel', y='Age', data=df)
plt.title('Age vs Risk Level')
plt.show()

sns.boxplot(x='RiskLevel', y='SystolicBP', data=df)
plt.title('Systolic BP vs Risk Level')
plt.show()

#preprocessing
le = LabelEncoder()
df['RiskLevel'] = le.fit_transform(df['RiskLevel'])

x = df.drop('RiskLevel', axis=1)
y = df['RiskLevel']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

X_train, X_test, Y_train, Y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

#Train model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

y_predict = model.predict(X_test)

print("Classification Report:\n", classification_report(Y_test, y_predict))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_predict))
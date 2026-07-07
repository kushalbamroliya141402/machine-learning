from sklearn import datasets;
from sklearn.model_selection import train_test_split;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.metrics import accuracy_score, classification_report;

cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_classifire = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifire.fit(X_train, y_train)

y_pred = rf_classifire.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred)) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
data = {
    "StudyHours": [2, 3, 5, 7, 8, 1, 4, 6, 9, 2, 5, 6, 3, 7, 8],
    "Attendance": [60, 65, 75, 85, 90, 50, 70, 80, 95, 55, 78, 82, 68, 88, 92],
    "PreviousMarks": [45, 50, 60, 70, 85, 40, 55, 65, 90, 48, 62, 72, 58, 75, 88],
    "Assignments": [5, 6, 8, 9, 10, 4, 7, 8, 10, 5, 7, 9, 6, 9, 10],
    "Result": [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1] # 0 = Fail, 1 = Pass
}
df = pd.DataFrame(data)
print("Dataset Preview:")
print(df.head())
X = df[["StudyHours", "Attendance", "PreviousMarks", "Assignments"]]
y = df["Result"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("\nModel Accuracy:", accuracy * 100, "%")
print("\nConfusion Matrix:\n", cm)
plt.scatter(df["StudyHours"], df["PreviousMarks"], c=df["Result"])
plt.xlabel("Study Hours")
plt.ylabel("Previous Marks")
plt.title("Study Hours vs Previous Marks")
plt.show()
new_student = np.array([[6, 80, 70, 8]])
prediction = model.predict(new_student)
if prediction[0] == 1:
    print("\nPrediction: Student is likely to PASS")
else:
    print("\nPrediction: Student is likely to FAIL")
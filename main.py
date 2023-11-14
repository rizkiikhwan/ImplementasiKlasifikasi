import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# Here is link for Dataset: https://app.datacamp.com/workspace/external-link?url=https%3A%2F%2Fwww.kaggle.com%2Fitssuru%2Floan-data

loan_data = pd.read_csv("loan_data.csv")
print(loan_data.head().to_string(), "\n")

# Helper function for data distribution
# Visualize the proportion of borrowers
def show_loan_distrib(data):
  count = ""
  if isinstance(data, pd.DataFrame):
      count = data["not.fully.paid"].value_counts()
  else:
      count = data.value_counts()

  count.plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
  plt.ylabel("Loan: Fully Paid Vs. Not Fully Paid")
  plt.legend(["Fully Paid", "Not Fully Paid"])
  plt.show()

# Visualize the proportion of borrowers
show_loan_distrib(loan_data)

# Check for null values.
print(loan_data.isnull().sum(), "\n")

# Check column types
print(loan_data.dtypes, "\n")


encoded_loan_data = pd.get_dummies(loan_data, prefix="purpose", drop_first=True)
print(encoded_loan_data.dtypes, "\n")

X = encoded_loan_data.drop('not.fully.paid', axis = 1)
y = encoded_loan_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify = y, random_state=2022)


X_train_cp = X_train.copy()
X_train_cp['not.fully.paid'] = y_train
y_0 = X_train_cp[X_train_cp['not.fully.paid'] == 0]
y_1 = X_train_cp[X_train_cp['not.fully.paid'] == 1]

y_0_undersample = y_0.sample(y_1.shape[0])
loan_data_undersample = pd.concat([y_0_undersample, y_1], axis=0)

# Visualize the proportion of borrowers
show_loan_distrib(loan_data_undersample)

smote = SMOTE(sampling_strategy='minority')

X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X, y)

# Visualize the proportion of borrowers
show_loan_distrib(y_train_SMOTE)

# Classification Models
from sklearn.metrics import classification_report, confusion_matrix

# Logistic Regression
from sklearn.linear_model import LogisticRegression

X = loan_data_undersample.drop('not.fully.paid', axis = 1)
y = loan_data_undersample['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify = y, random_state=2022)
logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train, y_train)
y_pred = logistic_classifier.predict(X_test)
confmatrix = confusion_matrix(y_test,y_pred)
classreport = classification_report(y_test,y_pred)
print(confmatrix)
print(classreport)

# SVM
from sklearn.svm import SVC
svc_classifier = SVC(kernel='linear')
svc_classifier.fit(X_train, y_train)

# Make Prediction & print the result
y_predsvm = svc_classifier.predict(X_test)
SVMTest = classification_report(y_test,y_predsvm)
print(SVMTest)
# Naive Bayes Classification

## Overview

Naive Bayes is a **probability-based supervised machine learning algorithm** mainly used for classification problems.

It is based on **Bayes' Theorem** and assumes that the features are conditionally independent of each other given the target class.

In this practical, **Gaussian Naive Bayes** is used to predict whether a person belongs to the diabetes class based on glucose and blood pressure values.

---

## Dataset

The dataset contains information related to glucose level, blood pressure, and diabetes classification.

### Dataset Columns

| Column          | Description                               |
| --------------- | ----------------------------------------- |
| `glucose`       | Glucose measurement                       |
| `bloodpressure` | Blood pressure measurement                |
| `diabetes`      | Target variable indicating diabetes class |

### Target Classes

* `0` → Non-diabetic class
* `1` → Diabetic class

---

## Objective

The objective of this project is to:

* Understand the Naive Bayes classification algorithm
* Prepare the dataset for classification
* Separate features and target
* Split the dataset into training and testing data
* Train a Gaussian Naive Bayes model
* Make predictions
* Evaluate the classification performance

---

## Algorithm Used

### Gaussian Naive Bayes

`GaussianNB` from Scikit-learn was used because the input features are continuous numerical values.

```python
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
```

---

## Workflow

The practical follows these steps:

```text
Load Dataset
     ↓
Inspect Dataset
     ↓
Data Preprocessing
     ↓
Separate X and y
     ↓
Train-Test Split
     ↓
Create Gaussian Naive Bayes Model
     ↓
Train Model
     ↓
Make Predictions
     ↓
Evaluate Model
```

---

## Features and Target

The independent variables are:

```text
glucose
bloodpressure
```

The dependent variable is:

```text
diabetes
```

The feature matrix is represented by `X`, while the target variable is represented by `y`.

---

## Model Training

The Gaussian Naive Bayes model was trained using the training dataset.

```python
nb_model.fit(X_train, y_train)
```

After training, predictions were generated using the test data:

```python
y_pred = nb_model.predict(X_test)
```

---

## Model Evaluation

### Accuracy

Accuracy was used to measure the percentage of correctly classified observations.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100)
```

### Confusion Matrix

A confusion matrix can be used to understand the correctly and incorrectly classified observations for each class.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
```

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

---

## Bayes' Theorem

Naive Bayes is based on Bayes' Theorem.

The algorithm calculates the probability of a class given the observed features and assigns the observation to the class with the highest probability.

---

## Why Gaussian Naive Bayes?

Gaussian Naive Bayes assumes that continuous features follow a Gaussian (normal) distribution within each class.

Since `glucose` and `bloodpressure` are numerical continuous features, `GaussianNB` is suitable for this classification practical.

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Jupyter Notebook / VS Code

---

## Machine Learning Concepts Learned

Through this practical, I learned:

* Supervised Learning
* Classification
* Naive Bayes
* Bayes' Theorem
* Gaussian Naive Bayes
* Train-Test Split
* Model Training
* Prediction
* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score
* Classification Report

---

## Key Learning

This practical helped me understand how a probability-based classification algorithm can use input features such as glucose and blood pressure to classify observations into different diabetes classes.

I also learned how to evaluate a classification model using accuracy, confusion matrix, precision, recall, and F1-score.

---

## Conclusion

Gaussian Naive Bayes was implemented to classify diabetes observations using glucose and blood pressure features.

The model was trained on the training data, used to make predictions on unseen test data, and evaluated using standard classification metrics.

This practical provided hands-on experience with **probability-based classification using Naive Bayes**.

---



# 🤖 Support Vector Machine (SVM) — Machine Learning

<div align="center">

# 📊 Support Vector Machine

### A Complete Machine Learning Classification Project

**Learning • Implementation • Visualization • Evaluation • Optimization**

<br>

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter\&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy\&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas\&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0)](https://seaborn.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github\&logoColor=white)](https://github.com/Maganpreet-Singh/svm-machine-learning)

</div>

---

## 📌 Overview

**Support Vector Machine (SVM)** is one of the most powerful and widely used supervised machine learning algorithms for classification and regression problems.

This repository provides a practical implementation of **Support Vector Machine classification using Python and Scikit-learn**.

The project follows a complete machine learning workflow:

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
SVM Model
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Hyperparameter Optimization
```

The main objective of this repository is not just to train an SVM model, but to understand **how SVM works, why it works, and when it should be used**.

---

# 🎯 Project Objectives

This project focuses on understanding both the **theory and practical implementation** of Support Vector Machines.

### Core Objectives

* Understand Support Vector Machines from fundamentals
* Understand hyperplanes and decision boundaries
* Understand the concept of maximum margin
* Understand support vectors
* Understand hard-margin and soft-margin classification
* Understand the `C` parameter
* Understand kernel functions
* Understand the `gamma` parameter
* Perform exploratory data analysis
* Prepare data for machine learning
* Apply feature scaling
* Train an SVM classifier
* Generate predictions
* Evaluate classification performance
* Analyze the confusion matrix
* Understand model behavior
* Perform hyperparameter tuning
* Build an end-to-end machine learning workflow

---

# 🧠 What is Support Vector Machine?

Support Vector Machine is a **supervised learning algorithm** that attempts to find the best possible decision boundary between different classes.

For a binary classification problem, SVM attempts to find a hyperplane:

```text
wᵀx + b = 0
```

where:

* `w` → weight vector
* `x` → input feature vector
* `b` → bias/intercept

The key idea is:

> **Find the decision boundary that separates the classes while maximizing the margin between them.**

---

# 📐 Hyperplane

A hyperplane represents the decision boundary separating classes.

For two-dimensional data, the hyperplane becomes a line:

```text
w₁x₁ + w₂x₂ + b = 0
```

For higher-dimensional datasets, it becomes a higher-dimensional surface.

Conceptually:

```text
                 Class 1
              ●    ●    ●

                   │
                   │
                   │  Decision
                   │  Boundary
                   │
              ○    ○    ○
                 Class 2
```

The goal is to find the boundary that provides the best separation between the classes.

---

# 📏 Maximum Margin

One of the defining characteristics of SVM is the concept of the **maximum margin**.

The margin is the distance between the decision boundary and the closest observations from both classes.

```text
       ●
       ●      |        ○
              |
              |
--------------|--------------  Decision Boundary
              |
              |
       ●      |        ○
```

SVM tries to maximize this distance.

A larger margin generally means a model that is more robust and better able to generalize to unseen data.

---

# 🎯 Support Vectors

The observations closest to the decision boundary are called **Support Vectors**.

These points are critical because they determine the position of the optimal decision boundary.

```text
Class A

   ●
      ●  ← Support Vector

----------------------------- Decision Boundary

      ○  ← Support Vector
   ○

Class B
```

The name **Support Vector Machine** comes from these observations.

---

# 🧱 Hard Margin SVM

Hard-margin SVM attempts to separate classes perfectly.

Conceptually:

```text
Class A     |     Class B

●  ●  ●     |     ○  ○  ○
●  ●        |     ○  ○
            |
        Decision
        Boundary
```

This approach works only when:

* Classes are clearly separable
* There is very little noise
* There are no problematic outliers

In real-world datasets, perfect separation is often unrealistic.

---

# 🪶 Soft Margin SVM

Soft-margin SVM allows some observations to violate the margin.

This creates a balance between:

```text
Large Margin
      ↕
Classification Errors
```

Soft-margin SVM is much more practical because real-world data often contains:

* Noise
* Outliers
* Overlapping classes

---

# 🔧 The `C` Parameter

The `C` parameter controls how strongly the model penalizes classification errors.

### Small `C`

A smaller value of `C`:

* Allows more classification errors
* Creates a wider margin
* Applies stronger regularization
* Can reduce overfitting

```text
Small C
   ↓
Wider Margin
   ↓
More Tolerance
   ↓
Simpler Model
```

### Large `C`

A larger value of `C`:

* Penalizes errors heavily
* Produces a narrower margin
* Attempts to classify training observations more accurately
* Can increase the risk of overfitting

```text
Large C
   ↓
Narrower Margin
   ↓
Stronger Error Penalty
   ↓
More Complex Model
```

---

# 🌀 Kernel Trick

SVM becomes especially powerful when the data is not linearly separable.

The **kernel trick** allows SVM to model non-linear relationships without explicitly constructing all higher-dimensional features.

Conceptually:

```text
Original Space

● ● ●
 ○ ○
● ● ●

      ↓ Kernel Transformation

Higher-Dimensional Space

       ●
      / \
     /   \
    ○     ○

      ↓

Separating Hyperplane
```

---

# 🧩 Common SVM Kernels

## 1. Linear Kernel

```python
SVC(kernel="linear")
```

Useful when classes are approximately linearly separable.

### Advantages

* Simple
* Fast
* Easy to interpret
* Works well for high-dimensional sparse data

---

## 2. Polynomial Kernel

```python
SVC(kernel="poly")
```

Useful when the relationship between variables is polynomial.

The complexity can be controlled using the `degree` parameter.

---

## 3. RBF Kernel

```python
SVC(kernel="rbf")
```

The **Radial Basis Function (RBF)** kernel is one of the most commonly used SVM kernels.

It can model complex non-linear relationships.

---

## 4. Sigmoid Kernel

```python
SVC(kernel="sigmoid")
```

This kernel uses a sigmoid function and can be useful for certain classification problems.

---

# 🎚️ The `gamma` Parameter

The `gamma` parameter determines how far the influence of a single training example reaches.

It is especially important when using the RBF kernel.

### Low Gamma

```text
Low Gamma
    ↓
Large Influence Area
    ↓
Smoother Boundary
    ↓
Possible Underfitting
```

### High Gamma

```text
High Gamma
    ↓
Small Influence Area
    ↓
More Complex Boundary
    ↓
Possible Overfitting
```

Therefore:

```text
C + Gamma + Kernel
        ↓
Model Complexity
```

---

# ⚖️ Why Feature Scaling Matters

Feature scaling is extremely important for SVM.

Suppose a dataset contains:

```text
Age          → 18 to 80

Income       → 20,000 to 2,000,000
```

Without scaling, the larger numerical range can dominate calculations.

Therefore, standardization is typically applied before training.

The standardization formula is:

```text
z = (x - μ) / σ
```

where:

* `x` = original value
* `μ` = mean
* `σ` = standard deviation

Using Scikit-learn:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Important

Never fit the scaler on the test dataset.

Correct:

```python
scaler.fit_transform(X_train)
scaler.transform(X_test)
```

Incorrect:

```python
scaler.fit_transform(X_test)
```

This helps prevent **data leakage**.

---

# 🛠️ Technologies Used

| Technology          | Purpose                         |
| ------------------- | ------------------------------- |
| 🐍 Python           | Programming language            |
| 📓 Jupyter Notebook | Development and experimentation |
| 🔢 NumPy            | Numerical computing             |
| 🐼 Pandas           | Data manipulation               |
| 📊 Matplotlib       | Data visualization              |
| 🎨 Seaborn          | Statistical visualization       |
| 🤖 Scikit-learn     | Machine learning                |

---

# 📦 Main Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
```

---

# 🔄 Complete Machine Learning Pipeline

```text
                    DATASET
                       │
                       ▼
              Data Understanding
                       │
                       ▼
               Data Cleaning
                       │
                       ▼
                    EDA
                       │
                       ▼
              Feature Selection
                       │
                       ▼
              Target Preparation
                       │
                       ▼
             Train-Test Split
                       │
                       ▼
              Feature Scaling
                       │
                       ▼
                 SVM Model
                       │
                       ▼
                  Training
                       │
                       ▼
                 Prediction
                       │
                       ▼
             Model Evaluation
                       │
                       ▼
          Hyperparameter Tuning
```

---

# 🔍 Project Workflow

## 1️⃣ Import Libraries

The first step is importing the required Python libraries.

These libraries provide functionality for:

* Data manipulation
* Mathematical operations
* Visualization
* Preprocessing
* Model training
* Model evaluation

---

## 2️⃣ Load Dataset

The dataset can be loaded using Pandas:

```python
df = pd.read_csv("data.csv")
```

Initial inspection:

```python
df.head()
df.tail()
df.shape
df.columns
df.info()
df.describe()
```

---

# 3️⃣ Data Exploration

Before training the model, the dataset should be understood properly.

Useful checks include:

```python
df.shape
df.isnull().sum()
df.duplicated().sum()
df.dtypes
```

Class distribution can also be checked.

```python
df["target"].value_counts()
```

This helps identify:

* Dataset size
* Missing values
* Duplicate observations
* Feature types
* Class imbalance

---

# 📊 Exploratory Data Analysis

Visualizations can be used to better understand relationships between features.

Typical visualizations include:

### Distribution Plot

```python
sns.histplot(data=df, x="feature")
plt.show()
```

### Box Plot

```python
sns.boxplot(data=df, x="target", y="feature")
plt.show()
```

### Scatter Plot

```python
sns.scatterplot(
    data=df,
    x="feature1",
    y="feature2",
    hue="target"
)

plt.show()
```

### Correlation Heatmap

```python
plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.show()
```

---

# 🧹 Data Preprocessing

Before model training, the data may require:

* Missing-value handling
* Duplicate removal
* Feature selection
* Encoding categorical variables
* Removal of irrelevant columns
* Feature scaling

The objective is to produce a clean and meaningful feature matrix.

---

# 🎯 Feature and Target Separation

The input variables and target variable are separated.

```python
X = df.drop(columns=["target"])
y = df["target"]
```

Where:

```text
X → Features
y → Target
```

---

# 🔀 Train-Test Split

The dataset is divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

### Why?

The model should be evaluated on data that it has never seen during training.

```text
Dataset
   │
   ├────────── Training Data ── 80%
   │
   └────────── Testing Data ─── 20%
```

---

# 📏 Feature Scaling

Standardization is applied after splitting the dataset.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

This ensures that all features operate on a comparable scale.

---

# 🤖 Building the SVM Model

A basic SVM model can be created using:

```python
model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)
```

---

# 🏋️ Model Training

Train the model using the scaled training data:

```python
model.fit(
    X_train_scaled,
    y_train
)
```

At this stage, SVM learns a decision boundary from the training data.

---

# 🔮 Prediction

After training:

```python
y_pred = model.predict(X_test_scaled)
```

The predictions can then be compared with the true test labels.

---

# 📈 Model Evaluation

## Accuracy

```python
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)
```

Formula:

```text
Accuracy =
(TP + TN) / (TP + TN + FP + FN)
```

---

# 🎯 Precision

```text
Precision = TP / (TP + FP)
```

Precision answers:

> Of all observations predicted as positive, how many were actually positive?

---

# 🔎 Recall

```text
Recall = TP / (TP + FN)
```

Recall answers:

> Of all actual positive observations, how many did the model correctly identify?

---

# ⚡ F1 Score

```text
F1 Score =
2 × (Precision × Recall)
/
(Precision + Recall)
```

F1 score is especially useful when both precision and recall matter.

---

# 🧮 Confusion Matrix

```python
cm = confusion_matrix(
    y_test,
    y_pred
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")

plt.show()
```

The confusion matrix contains:

```text
                 Predicted
                 Negative  Positive

Actual Negative     TN        FP

Actual Positive     FN        TP
```

---

# 🧪 Classification Report

```python
print(
    classification_report(
        y_test,
        y_pred
    )
)
```

The classification report typically provides:

* Precision
* Recall
* F1 score
* Support

---

# 🔬 Kernel Comparison

Different kernels can produce very different decision boundaries.

```python
kernels = [
    "linear",
    "poly",
    "rbf",
    "sigmoid"
]
```

Example:

```python
for kernel in kernels:

    model = SVC(
        kernel=kernel
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    y_pred = model.predict(
        X_test_scaled
    )

    score = accuracy_score(
        y_test,
        y_pred
    )

    print(
        f"{kernel}: {score:.4f}"
    )
```

This experiment demonstrates how kernel choice affects classification performance.

---

# 🎛️ Hyperparameter Tuning

SVM provides several important hyperparameters.

| Parameter | Purpose                                 |
| --------- | --------------------------------------- |
| `C`       | Controls error penalty                  |
| `kernel`  | Defines decision function               |
| `gamma`   | Controls influence of training examples |
| `degree`  | Degree of polynomial kernel             |
| `coef0`   | Independent term in certain kernels     |

---

# 🔎 GridSearchCV

One approach to finding better hyperparameters is `GridSearchCV`.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": [
        "linear",
        "rbf",
        "poly"
    ],
    "gamma": [
        "scale",
        "auto"
    ]
}

grid = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(
    X_train_scaled,
    y_train
)
```

Best parameters:

```python
print(
    "Best Parameters:",
    grid.best_params_
)
```

Best cross-validation score:

```python
print(
    "Best CV Score:",
    grid.best_score_
)
```

---

# 🧠 SVM Intuition

The basic intuition can be summarized as:

```text
              Find a boundary
                     ↓
          Separate the classes
                     ↓
           Maximize the margin
                     ↓
             Identify support
                vectors
                     ↓
          Classify new examples
```

For non-linear data:

```text
Non-linear Data
       ↓
Kernel Function
       ↓
Higher-dimensional Representation
       ↓
Linear Separation in New Space
```

---

# 📚 Important SVM Concepts

| Concept        | Meaning                                            |
| -------------- | -------------------------------------------------- |
| Hyperplane     | Decision boundary                                  |
| Margin         | Distance between boundary and closest observations |
| Support Vector | Observation closest to the boundary                |
| `C`            | Error penalty                                      |
| `gamma`        | Influence of a training point                      |
| Kernel         | Function used to model non-linear relationships    |
| RBF            | Common non-linear kernel                           |
| Hard Margin    | Strict separation                                  |
| Soft Margin    | Allows margin violations                           |
| Scaling        | Places features on comparable scales               |

---

# ✅ Advantages of SVM

### 1. Effective in High Dimensions

SVM can perform well when the number of features is large.

### 2. Strong Classification Performance

It can produce powerful decision boundaries for many classification tasks.

### 3. Kernel Flexibility

The kernel trick allows SVM to handle non-linear relationships.

### 4. Good Generalization

The maximum-margin principle helps reduce the risk of poor generalization.

### 5. Useful for Small and Medium Datasets

SVM is particularly practical for many small-to-medium-sized datasets.

---

# ⚠️ Limitations of SVM

### 1. Computational Cost

Training can become expensive on very large datasets.

### 2. Scaling Sensitivity

Feature scaling is often essential.

### 3. Hyperparameter Sensitivity

Poor values of `C` and `gamma` can significantly affect performance.

### 4. Kernel Selection

Choosing the correct kernel may require experimentation.

### 5. Lower Interpretability

Compared with simple linear models, SVM decision-making can be harder to explain.

### 6. Outlier Sensitivity

Extreme observations can influence the decision boundary.

---

# 🚀 Future Improvements

This project can be extended significantly.

### 🔹 Model Improvements

* Cross-validation
* GridSearchCV
* RandomizedSearchCV
* Better feature engineering
* Feature selection
* PCA
* Class-weight optimization

### 🔹 Evaluation Improvements

* ROC Curve
* ROC-AUC
* Precision-Recall Curve
* Cross-validation score
* Learning curves

### 🔹 Model Comparison

Compare SVM against:

```text
Logistic Regression
        ↓
KNN
        ↓
Decision Tree
        ↓
Random Forest
        ↓
Naive Bayes
        ↓
SVM
```

### 🔹 Deployment

The trained model could later be deployed using:

* Streamlit
* Flask
* FastAPI

The model can also be saved using:

```python
import joblib

joblib.dump(
    model,
    "svm_model.pkl"
)
```

---

# 📁 Repository Structure

```text
svm-machine-learning/
│
├── 📓 SVM.ipynb
│
├── 📊 data.csv
│
└── 📖 README.md
```

---

# 💻 Installation

Clone the repository:

```bash
git clone https://github.com/Maganpreet-Singh/svm-machine-learning.git
```

Move into the project:

```bash
cd svm-machine-learning
```

Install the dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
SVM.ipynb
```

---

# 🗺️ Learning Roadmap

This repository can be viewed as a progression:

```text
Python
  ↓
NumPy
  ↓
Pandas
  ↓
Matplotlib / Seaborn
  ↓
Exploratory Data Analysis
  ↓
Data Preprocessing
  ↓
Feature Scaling
  ↓
Supervised Learning
  ↓
Classification
  ↓
Support Vector Machine
  ↓
Kernel Methods
  ↓
Hyperparameter Tuning
  ↓
Model Evaluation
  ↓
Deployment
```

---

# 🏆 Key Takeaways

After completing this project, you should be able to explain:

* What SVM is
* How SVM finds a decision boundary
* Why margins matter
* What support vectors are
* Difference between hard-margin and soft-margin SVM
* How `C` affects the model
* How `gamma` affects the model
* Why kernels are useful
* Difference between linear and non-linear kernels
* Why feature scaling matters
* How to train SVM using Scikit-learn
* How to evaluate a classification model
* How to tune SVM hyperparameters

---

# 📌 Best Practices

```text
✅ Split data before scaling
✅ Fit scaler only on training data
✅ Scale features before SVM
✅ Use cross-validation
✅ Tune C and gamma
✅ Compare kernels
✅ Evaluate beyond accuracy
✅ Inspect confusion matrix
✅ Watch for overfitting
✅ Keep preprocessing reproducible
```

---

# 🧪 Example End-to-End SVM Code

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv("data.csv")

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create model
model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

# Train model
model.fit(
    X_train_scaled,
    y_train
)

# Predictions
y_pred = model.predict(
    X_test_scaled
)

# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"Accuracy: {accuracy:.4f}"
)

# Classification report
print(
    classification_report(
        y_test,
        y_pred
    )
)

# Confusion matrix
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)
```

---

# 🌟 Why This Project Matters

SVM is an important algorithm to learn because it introduces several foundational machine learning concepts at once:

```text
Classification
     +
Optimization
     +
Geometry
     +
Regularization
     +
Kernel Methods
     +
Feature Scaling
     +
Model Evaluation
```

Understanding SVM provides a strong foundation for moving toward more advanced machine learning concepts.

---

# 👨‍💻 Author

## Maganpreet Singh

**B.Tech Computer Science & Engineering Student**

Interested in:

```text
🐍 Python
🤖 Machine Learning
📊 Data Science
🧠 Artificial Intelligence
💻 Software Development
```

### GitHub

👉 https://github.com/Maganpreet-Singh

### Repository

👉 https://github.com/Maganpreet-Singh/svm-machine-learning

---

# ⭐ Support

If this repository helped you understand **Support Vector Machines**, consider giving it a ⭐ on GitHub.

Your support helps motivate further machine learning projects and learning resources.

---

# 📄 License

This project is created for **educational and learning purposes**.

You are free to explore, modify, and extend the implementation for your own learning and experimentation.

---

<div align="center">

### 🚀 Learn → Build → Experiment → Improve

**Machine Learning is not just about using algorithms — it is about understanding why they work.**

⭐ **Keep Learning. Keep Building.**

</div>

# 🩺 Breast Cancer Prediction using Machine Learning

A machine learning classification project that predicts whether a breast tumor is **Benign** or **Malignant** using tumor measurement features.

The project compares multiple classification algorithms, selects a calibrated **Support Vector Machine (SVM)** as the final model, and deploys it through an interactive **Streamlit web application**.

---

## 📌 Project Overview

Breast cancer classification is a binary classification problem where the model predicts one of two outcomes:

- **0 → Benign**
- **1 → Malignant**

In this project, I explored the dataset, performed data preprocessing and exploratory data analysis, trained multiple machine learning models, evaluated their performance using several classification metrics, and selected the best-performing model for deployment.

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
Data Preprocessing
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Final Model Selection
   ↓
Model Saving
   ↓
Streamlit Deployment
```
---

## 🎯 Objective

The main objectives of this project were to:

- Understand and explore the dataset
- Perform data cleaning and preprocessing
- Analyze feature distributions and correlations
- Train multiple classification models
- Compare their performance using different evaluation metrics
- Select the best overall model
- Save the trained model and scaler
- Build an interactive Streamlit application for prediction

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Git
- GitHub
---

## 📊 Dataset

The dataset contains measurements describing characteristics of breast tumors.

The features include:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave Points
- Symmetry
- Fractal Dimension

Each measurement is available in three forms:

- **Mean**
- **Standard Error (SE)**
- **Worst**

This results in **30 numerical features** used for prediction.

### Target

The original diagnosis labels were:

- `B` → Benign
- `M` → Malignant

These were encoded as:

```text
B → 0
M → 1
```
> **Note:** The dataset used in this project was provided by the InternPe team for internship purposes and is not included in this repository.
---

## 🔍 Exploratory Data Analysis

The dataset was explored before model training to understand its structure, distributions, and relationships between features.

The following analysis was performed:

- Dataset inspection
- Checking data types
- Checking missing values
- Checking duplicate records
- Statistical summary
- Target distribution
- Feature distributions
- Outlier analysis using boxplots
- Correlation analysis using a heatmap

### Data Cleaning

The following unnecessary columns were removed:

- `id`
- `Unnamed: 32`

The `id` column was an identifier and was not useful for prediction.

The `Unnamed: 32` column contained missing values and was removed before model training.

---

## ⚙️ Data Preprocessing

### 1. Target Encoding

The diagnosis labels were converted into numerical values:

```text
B → 0
M → 1
```
### 2. Feature and Target Separation

The dataset was separated into:

```text
X → Features
y → Target
```
### 3. Train-Test Split

The dataset was divided into:

```text
80% → Training Data
20% → Testing Data
```
A stratified split was used to maintain the class distribution between the training and testing data.

### 4. Feature Scaling

`StandardScaler` was used to standardize the numerical features.

The scaler was fitted on the training data and then used to transform both the training and testing data.

The same scaler was later saved and used by the Streamlit application when processing new inputs.
---

## 🤖 Machine Learning Models

Four classification models were trained and compared in this project.

### 1. Logistic Regression

Logistic Regression was used as a baseline model for the binary classification task.

### 2. Support Vector Machine (SVM)

SVM stands for **Support Vector Machine**.

An SVM with an **RBF (Radial Basis Function) kernel** was used for classification.

The SVM was calibrated using `CalibratedClassifierCV` with sigmoid calibration to obtain calibrated probability estimates.

### 3. Random Forest

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

### 4. Multi-Layer Perceptron (MLP)

MLP stands for **Multi-Layer Perceptron**.

It is a neural-network-based classification model used for the prediction task.
---

## 📈 Evaluation Metrics

The models were evaluated using the following metrics:

### Accuracy

Measures the overall percentage of predictions that were correct.

### Precision

Measures how many of the samples predicted as positive were actually positive.

### Recall

Measures how many of the actual positive samples were correctly identified.

### F1 Score

Provides a balance between precision and recall.

### ROC-AUC

Measures how well the model distinguishes between the two classes across different classification thresholds.

Using multiple metrics provides a more complete understanding of model performance instead of relying only on accuracy.
---

## 📊 Model Performance Comparison

The models were evaluated on the test set using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 96.49% | 97.50% | 92.86% | 95.12% | 99.60% |
| **SVM** | **97.37%** | **97.56%** | **95.24%** | **96.39%** | **99.44%** |
| Random Forest | **97.37%** | **100.00%** | 92.86% | 96.30% | 99.29% |
| MLP | 96.49% | 100.00% | 90.48% | 95.00% | 99.27% |
---

## 🏆 Final Model Selection

After comparing all four models, **SVM was selected as the final model**.

SVM achieved the highest accuracy along with a strong balance between precision, recall, and F1 Score.

Although Random Forest achieved the same accuracy and higher precision, SVM achieved better recall and a slightly better F1 Score.

### Final SVM Performance

- **Accuracy:** 97.37%
- **Precision:** 97.56%
- **Recall:** 95.24%
- **F1 Score:** 96.39%
- **ROC-AUC:** 99.44%
---

## 🔲 Confusion Matrix

The final SVM produced the following confusion matrix on the test set:

```text
                 Predicted
              Benign  Malignant

Actual Benign     71       1
Actual Malignant   2      40
```
---

## 💾 Model Saving

After selecting SVM as the final model, the trained SVM model and the `StandardScaler` were saved using **Joblib**.

The saved files are:

```text
model/
├── svm_model.pkl
└── scaler.pkl
```
---

## 🌐 Streamlit Web Application

The trained SVM model was deployed using **Streamlit** to create an interactive web application.

The application allows users to:

- Enter the 30 tumor measurement values manually
- Load a predefined Benign test sample
- Load a predefined Malignant test sample
- Run the trained model
- View the prediction result

The application also includes custom CSS styling, a background image, and a medical disclaimer.
---

## 🔄 Prediction Workflow

The application follows this workflow when making a prediction:

```text
User Input
    ↓
30 Tumor Measurements
    ↓
NumPy Array
    ↓
StandardScaler
    ↓
Scaled Input
    ↓
Saved SVM Model
    ↓
Prediction
    ↓
Benign / Malignant
```
---

## 🧪 Quick Test

Entering all 30 tumor measurements manually can be inconvenient while testing the application.

To make testing easier, the application provides two predefined test samples:

### 🟢 Benign Test Sample

Loads a predefined Benign sample into the input fields.

### 🔴 Malignant Test Sample

Loads a predefined Malignant sample into the input fields.

This allows the model to be tested quickly without manually entering all 30 feature values.
---

## 📁 Project Structure

```text
breast-cancer-prediction-ml/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── assets/
│   └── background.jpg
│
├── model/
│   ├── scaler.pkl
│   └── svm_model.pkl
│
└── notebook/
    └── Breast_Cancer_Prediction.ipynb
```
---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aryanpatil-me/breast-cancer-prediction-ml.git
```
### 2. Navigate to the Project Directory

```bash
cd breast-cancer-prediction-ml
```
### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
---

## ▶️ How to Run

After installing the dependencies, run the Streamlit application using:

```bash
streamlit run app.py
```
---

## 📚 Learning Outcomes

Through this project, I learned and practiced:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization
- Feature scaling
- Train-test splitting
- Classification algorithms
- Model evaluation
- Model comparison
- Confusion matrix analysis
- Probability calibration
- Model saving using Joblib
- Streamlit application development
- Git and GitHub

---

## 🚀 Future Improvements

Possible future improvements include:

- Hyperparameter tuning
- More extensive cross-validation
- Feature engineering
- Experimenting with additional classification algorithms
- Improving input validation
- Adding more interactive visualizations
- Deploying the Streamlit application to a cloud platform
---

## ⚠️ Disclaimer

This project was created for **educational and demonstration purposes only**.

The predictions generated by this application should **not** be considered medical advice and should not be used for actual medical diagnosis or treatment decisions.

---

## 👨‍💻 Author

**Aryan Patil**

Machine Learning Enthusiast

**Python • Scikit-learn • Streamlit**

---

⭐ If you found this project interesting, feel free to explore the notebook and application code.

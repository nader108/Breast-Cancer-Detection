# Breast Cancer Detection using Multiple Machine Learning Models

Overview

This project demonstrates the use of multiple machine learning algorithms to classify breast cancer tumors as either malignant or benign. The dataset used is the Breast Cancer Wisconsin (Diagnostic) Data Set. Various models, including Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Decision Tree, Random Forest, XGBoost, and Naive Bayes, are trained to predict whether a tumor is malignant or benign based on features such as the radius, texture, perimeter, area, and other properties of cell nuclei.

Dataset

The dataset used is the Breast Cancer Wisconsin (Diagnostic) Data Set, consisting of 569 entries and 32 feature columns. The target variable (diagnosis) is binary, with "M" representing malignant tumors and "B" for benign tumors. The dataset includes measurements such as radius, texture, smoothness, and compactness of cell nuclei.

Objective

Build and compare multiple machine learning models for predicting breast cancer diagnoses.
Evaluate the models using metrics like accuracy, precision, recall, f1-score, and ROC AUC.
Technologies Used
Python: For data processing and model building.
Pandas: For data manipulation and cleaning.
NumPy: For numerical operations.
Scikit-Learn: For implementing machine learning models and evaluation metrics.
XGBoost: For advanced boosting algorithms.
Matplotlib: For data visualization.

Key Steps

1. Data Preprocessing:
Loaded the Breast Cancer Wisconsin (Diagnostic) Data Set.
Cleaned the dataset by removing unnecessary columns and handling missing values.
Encoded the target variable diagnosis as binary values (1 for malignant, 0 for benign).
2. Model Building:
Logistic Regression: Used for binary classification of cancer type.
K-Nearest Neighbors (KNN): Classified the data based on the nearest neighbors.
Support Vector Machine (SVM): Applied SVM for classification.
Decision Tree Classifier: A non-linear classifier used for modeling.
Random Forest Classifier: An ensemble method that builds multiple decision trees.
XGBoost: A gradient boosting algorithm known for high performance.
Naive Bayes: Applied Gaussian Naive Bayes for classification.
3. Model Evaluation:
Each model was evaluated using classification metrics like accuracy, precision, recall, f1-score, and ROC AUC.
Error Rate vs K was plotted for KNN.
Confusion Matrix was plotted for understanding model performance.
4. Ensemble Learning:
Built an Ensemble Model using Voting Classifier combining Decision Tree, SVM, and Random Forest.
The final model was trained and predictions were made on the test set.
5. Hyperparameter Tuning:
Optimized the hyperparameters using GridSearchCV for models like Logistic Regression and XGBoost.
Results
Logistic Regression:

Accuracy: 95.8%
F1-score: 93.8%
Precision: 91.8%
Recall: 95.7%
ROC AUC: 0.9940
K-Nearest Neighbors (KNN):

Optimal K: 20
Error rate vs K was plotted, showing the relationship between K and model performance.
SVM:

Recall: 85.1%
Decision Tree:

Accuracy: 90%
F1-score: 89%
Recall: 94%
Confusion Matrix and Feature Importances were visualized.
Random Forest:

Built and evaluated the Random Forest model.
XGBoost:

Model trained and performance evaluated.
Naive Bayes:

Model trained using Gaussian Naive Bayes and evaluated.
Ensemble Model (Voting Classifier):

F1-score: 93.4%
Visualizations
ROC Curve: For model evaluation across different thresholds.
Confusion Matrix: To understand true positives, false positives, true negatives, and false negatives.
Feature Importances: For Decision Tree and Random Forest to understand which features are most important.
Hyperparameter Tuning
GridSearchCV was used to find optimal parameters for Logistic Regression, SVM, and XGBoost.
Best hyperparameters for Logistic Regression: C=2, max_iter=150.
XGBoost model was optimized with learning_rate=0.3, max_depth=6, and n_estimators=100.
Installation
To run this project locally, clone the repository and install the required dependencies.

bash
Copy code
git clone https://github.com/yourusername/breast-cancer-detection.git
cd breast-cancer-detection
pip install -r requirements.txt
Usage
Run the Jupyter notebook to explore the analysis, model training, evaluation, and visualizations.

Conclusion
This project compares multiple machine learning models for breast cancer detection. The Logistic Regression model performed exceptionally well, but ensemble methods like Random Forest and XGBoost also showed promising results. The Ensemble Model outperformed individual models with a F1-score of 93.4%.

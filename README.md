Mobile Review Sentiment Classification Project
---------------------------------------------

1. Project Overview
This project builds a complete machine learning pipeline to classify mobile phone review sentiment into three classes: Positive, Neutral, and Negative.

The workflow includes:
- Data preprocessing
- Feature engineering
- Class balancing (SMOTE)
- Feature selection (SelectKBest)
- Dimensionality reduction experiments (PCA, LDA, KernelPCA on a subset)
- Model training and hyperparameter tuning (RandomizedSearchCV)
- Final model evaluation and comparison

The following models are compared:
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier

The main goal is to identify which model provides the best predictive performance while demonstrating a full, reproducible ML workflow.

2. Dataset
- Filename: Mobile Reviews Sentiment.csv
- Problem type: Multi-class classification (sentiment)

Key features used:
- Categorical: brand, model
- Numeric:
  - price_usd
  - battery_life_rating, camera_rating, performance_rating, design_rating, display_rating
- Target: sentiment (Positive / Neutral / Negative)

Data cleaning steps:
- Rows with missing sentiment labels were removed.
- An additional feature was engineered:
  - avg_spec_rating = mean of the five technical ratings (battery, camera, performance, design, display).

3. Preprocessing Pipeline
The preprocessing pipeline includes:
- Handling missing values:
  - Numeric features imputed with the median
  - Categorical features imputed with the most frequent value
- Scaling numeric features with StandardScaler
- One-hot encoding categorical variables (brand, model)
- Creating avg_spec_rating
- Splitting data into train and test sets (stratified 80/20 split)
- Encoding target labels using LabelEncoder (Negative, Neutral, Positive → integers)

To address class imbalance, SMOTE is applied only to the training set to create a balanced training dataset and enhance learning for the minority sentiment classes.

4. Feature Selection
One-hot encoding of brand and model leads to a high-dimensional feature space.

To reduce noise and training time:
- SelectKBest with the ANOVA F-test (f_classif) is used.
- The top K = 30 features (or min(K, total_features) if fewer) are retained.

This keeps the most informative features while making models more efficient and less prone to overfitting.

5. Models Trained
Four supervised classifiers are tuned and evaluated:
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier

Hyperparameter tuning details:
- RandomizedSearchCV
- 3-fold cross-validation
- Scoring: f1_macro (Macro F1-score)
- Tuning is performed on a random subset of the balanced training data, then the best configuration is retrained on the full balanced set.

Each model is trained using the feature-selected data and evaluated on the original (unbalanced) test set.

6. Evaluation Metrics
Model performance is evaluated using:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-Score

A comparative results table and a bar plot of F1 scores are generated to visualize and compare model performance.

7. Key Findings
- LightGBM achieves the best overall performance (highest Accuracy and Macro F1).
- XGBoost also performs very well and is close to LightGBM.
- Random Forest is strong and stable, providing good performance with less tuning.
- Decision Tree serves as a simple, interpretable baseline but does not match ensemble models.
- The engineered feature avg_spec_rating improves both interpretability and performance by summarizing technical quality.
- SMOTE effectively balances the training classes and improves recall and F1 for the minority (Neutral/Negative) classes.
- Feature selection via SelectKBest reduces dimensionality with little or no loss in performance.

8. Conclusion
This project demonstrates a full, end-to-end machine learning workflow:
- From raw CSV data
- Through preprocessing, feature engineering, and class balancing
- To model selection, hyperparameter tuning, and evaluation

LightGBM emerges as the best-performing model in this setting, offering strong generalization and robust performance across all sentiment classes.

The pipeline is designed to be modular and extensible. Possible extensions include:
- Adding text-based features from review content (NLP)
- Performing more extensive hyperparameter searches
- Deploying the best model as an API or service
- Experimenting with deep learning or multi-modal models (text + structured data)

9. Requirements
Python version: 3.8+ (recommended)

Key Python libraries:
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- lightgbm
- matplotlib
- joblib

Install with:
    pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm matplotlib joblib

10. How to Run
1. Place Mobile Reviews Sentiment.csv in the project working directory.
2. Open and run the notebook (e.g., Group4-notebook.ipynb) or the corresponding Python script in order:
   - Data loading and preprocessing
   - Feature engineering and balancing
   - Feature selection and dimensionality reduction (optional exploratory cells)
   - Model training, tuning, and evaluation
3. After execution, inspect:
   - Printed metrics (Accuracy, Precision, Recall, F1)
   - Comparison tables
   - F1 score bar plots
4. Modify hyperparameters, feature selection settings, or models as desired for further experimentation.

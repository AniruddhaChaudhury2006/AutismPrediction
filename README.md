# Autism Prediction Model

This project aims to develop and evaluate machine learning models for predicting Autism Spectrum Disorder (ASD) based on various features. The notebook covers data loading, preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition. Early detection and intervention can significantly improve outcomes. This project leverages a dataset containing various behavioral and demographic features to build a predictive model for ASD.

## Dataset

The dataset `autism.csv` contains 800 entries with 22 features. Key features include:
- `A1_Score` to `A10_Score`: Scores from an autism screening questionnaire.
- `age`: Age of the individual.
- `gender`: Gender of the individual.
- `ethnicity`: Ethnic background.
- `jaundice`: Whether the individual had jaundice at birth.
- `austim`: Whether a family member has autism.
- `contry_of_res`: Country of residence.
- `used_app_before`: Whether an app was used for screening before.
- `result`: Score from a screening test.
- `relation`: Relationship of the tester to the individual.
- `Class/ASD`: The target variable, indicating the presence (1) or absence (0) of ASD.

## Data Preprocessing

The following preprocessing steps were performed:
1.  **Load Data**: The `autism.csv` file was loaded into a pandas DataFrame.
2.  **Data Cleaning**: 
    - The `age` column was converted to integer type.
    - `ID` and `age_desc` columns were dropped as they were not relevant for modeling.
    - Inconsistent country names (`Viet Nam` to `Vietnam`, `AmericanSamoa` to `United States`, `Hong Kong` to `China`) were standardized.
    - Missing values represented by '?' in `ethnicity` and `relation` columns were replaced with 'Others'.
    - `relation` categories were simplified by grouping 'Relative', 'Parent', and 'Health care professional' into 'Others'.
3.  **Encoding Categorical Features**: All object-type categorical columns (`gender`, `ethnicity`, `jaundice`, `austim`, `contry_of_res`, `used_app_before`, `relation`) were converted into numerical representations using `LabelEncoder`.
4.  **Outlier Handling**: Outliers in `age` and `result` columns were replaced with their respective medians using a custom function `replace_outliers_with_median`.
5.  **Target-Feature Split**: The dataset was split into features (`x`) and the target variable (`y`, which is `Class/ASD`).
6.  **Train-Test Split**: The data was split into training (80%) and testing (20%) sets using `train_test_split`.
7.  **Handling Class Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data to address the imbalance in the `Class/ASD` target variable, ensuring that both classes have an equal number of samples.

## Exploratory Data Analysis (EDA)

Key EDA steps included:
-   **Descriptive Statistics**: `df.describe()` and `df.info()` were used to understand data types, non-null counts, and basic statistics.
-   **Unique Values**: Unique values for all non-numerical columns were inspected to identify inconsistencies and prepare for encoding.
-   **Distribution Plots**: Histograms with KDE were generated for `age` and `result` to visualize their distributions, along with mean and median indicators.
-   **Box Plots**: Box plots for `age` and `result` were created to identify outliers.
-   **Outlier Detection**: The Interquartile Range (IQR) method was used to quantify outliers in `age` and `result`.
-   **Categorical Count Plots**: Count plots were generated for all categorical features to visualize their distributions.
-   **Correlation Heatmap**: A heatmap was used to visualize the correlation matrix of all features, showing relationships between variables.

## Model Training and Evaluation

Three classification models were considered:
-   Decision Tree Classifier (`DecisionTreeClassifier`)
-   Random Forest Classifier (`RandomForestClassifier`)
-   XGBoost Classifier (`XGBClassifier`)

### Cross-Validation
Initial evaluation of models was done using 5-fold cross-validation on the SMOTE-resampled training data to assess baseline performance.

### Hyperparameter Tuning (Randomized Search)
`RandomizedSearchCV` was used to find optimal hyperparameters for each model:
-   **Decision Tree**: Tuned `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
-   **Random Forest**: Tuned `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `bootstrap`.
-   **XGBoost**: Tuned `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.

### Best Model Selection
The model with the highest cross-validation accuracy from the randomized search was selected as the `best_model`.

## Results

The best performing model was the `RandomForestClassifier` with the following hyperparameters:
```
RandomForestClassifier(bootstrap=False, max_depth=20, n_estimators=50, random_state=42)
```

The evaluation on the test set (`x_test`, `y_test`) yielded the following metrics:
-   **Accuracy Score**: 0.81875
-   **Confusion Matrix**:
    ```
    [[108  16]
     [ 13  23]]
    ```
-   **Classification Report**:
    ```
                   precision    recall  f1-score   support

               0       0.89      0.87      0.88       124
               1       0.59      0.64      0.61        36

        accuracy                           0.82       160
       macro avg       0.74      0.75      0.75       160
    weighted avg       0.82      0.82      0.82       160
    ```

The best model was saved as `best_model.pkl`.

## Installation

To run this notebook, you will need the following Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imblearn
```

## Usage

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Open the Jupyter/Colab notebook**: Launch the notebook in your preferred environment.
3.  **Run all cells**: Execute the cells sequentially to reproduce the analysis and model training.

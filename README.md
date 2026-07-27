# Hypertension Medication Habit Analysis

Factors associated with medication habits (fasting vs. postprandial administration) among community-dwelling hypertensive patients — a cross-sectional study with univariable and multivariable logistic regression

## Background

This study investigates factors associated with medication habits (fasting vs. postprandial administration) among community-dwelling hypertensive patients, based on data from the National Basic Public Basic Public Health Service Program in Guangzhou, China (2026). A cross-sectional design was employed, using univariable analysis (χ² tests) and multivariable logistic regression to identify factors independently associated with medication habits, followed by comprehensive model credibility checks.

## Methods

Study design: Cross-sectional analysis
Primary outcome: Medication habit (fasting = 0, postprandial = 1)
Statistical methods:
Baseline characteristics table (Table 1): χ² tests for categorical variables, Welch's t-test for continuous variables
Univariable analysis: χ² tests for categorical variables, with automatic exclusion of sparse categories (threshold < 20)
Multivariable logistic regression: full-variable model with automatic detection and exclusion of complete separation variables
Model credibility checks: Hosmer-Lemeshow goodness-of-fit test, ROC curve & AUC, multicollinearity (VIF), Link Test for model specification, residual analysis & outlier detection, comprehensive rating
Software: Python (Statsmodels, Scipy, Scikit-learn)

## Repository Structure

├── data_analysis_20260727.py          
├── requirements.txt            
└── README.md

## Requirements

pip install pandas numpy statsmodels scipy scikit-learn openpyxl

## License

MIT License

## Citation

If you use this code, please cite the corresponding publication.

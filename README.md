# Hypertension Medication Habit Analysis

Community-dwelling hypertensive patients' medication habits: associated factors and blood pressure control — a cross-sectional study with logistic regression and sensitivity analyses.

## Background

This study investigates factors associated with medication habits (fasting vs. postprandial administration) among community-dwelling hypertensive patients and their association with blood pressure control, based on data from the National Basic Public Health Service Program in Guangzhou, China (2026).

## Methods

- **Study design**: Cross-sectional analysis of 723 hypertensive patients
- **Primary outcome**: Medication habit (fasting = 1, postprandial = 0)
- **Statistical methods**: Univariable and multivariable logistic regression, propensity score matching (PSM), inverse probability weighting (IPW), E-value sensitivity analysis
- **Software**: Python (Statsmodels, Scipy)

## Key Findings

- Male sex was independently associated with fasting medication habit (adjusted OR = 1.608, 95% CI: 1.177–2.197)
- Left-arm SBP showed a negative association with fasting habit (OR = 0.821 per 10 mmHg), but robustness was limited (E-value = 1.746 < sex effect OR = 1.942)
- Medication habit was not significantly associated with blood pressure control categories

## Repository Structure

├── analysis_script.py          # Main statistical analysis
├── visualization_script.py     # Publication-quality figures
├── requirements.txt            # Python dependencies
└── README.md

## Requirements

pip install pandas numpy statsmodels scipy matplotlib seaborn openpyxl

## License

MIT License

## Citation

If you use this code, please cite the corresponding publication.

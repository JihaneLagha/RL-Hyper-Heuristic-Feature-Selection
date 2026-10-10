# Reinforcement-Learning-Based Hyper-Heuristic for Feature Selection in Disease Prediction

## Project Overview

This project investigates a reinforcement-learning-based hyper-heuristic for feature selection in disease prediction.

The RL controller dynamically selects among four low-level feature-selection operators:

- ADD
- REMOVE
- SWAP
- FLIP

The approach is evaluated on two healthcare datasets:

1. CDC Diabetes Health Indicators
2. UCI Cleveland Heart Disease

The RL method is compared with:

- All-feature baseline
- Random heuristic selection
- Fixed heuristic strategies
- SelectKBest
- L1-based feature selection
- Recursive Feature Elimination (RFE)

Statistical analysis includes paired Wilcoxon signed-rank tests and Holm multiple-comparison correction.

## Project Structure

```text
Emerging Topics in AI Project/
|
+-- 01_data/
|   +-- archive.zip
|   +-- heart+disease.zip
|
+-- 02_scripts/
|   +-- 01_dataset_inspection.py
|   +-- 02_baseline_model.py
|   +-- 03_simple_feature_selection.py
|   +-- 04_feature_heuristics.py
|   +-- 05_subset_evaluation.py
|   +-- 06_qlearning_hyperheuristic.py
|   +-- 07_repeated_rl_runs.py
|   +-- 08_random_hyperheuristic_baseline.py
|   +-- 09_fixed_heuristic_baselines.py
|   +-- 11_30seed_baseline_comparison.py
|   +-- 12_statistical_comparison.py
|   +-- 13_paired_statistical_comparison.py
|   +-- 14_multiple_comparison_correction.py
|   +-- 15_heart_dataset_inspection.py
|   +-- 16_heart_baseline_model.py
|   +-- 17_heart_feature_selection.py
|   +-- 18_heart_qlearning_hyperheuristic.py
|   +-- 19_heart_repeated_rl.py
|   +-- 20_heart_statistical_analysis.py
|   +-- 21_diabetes_conventional_baselines.py
|   +-- 22_diabetes_repeated_final_evaluation.py
|   +-- 23_diabetes_final_statistical_comparison.py
|   +-- 24_heart_conventional_repeated_baselines.py
|   +-- 25_heart_final_method_comparison.py
|   +-- 26_final_results_consolidation.py
|   +-- project_paths.py
|
+-- 03_results/
|   +-- diabetes/
|   +-- heart_disease/
|   +-- statistical_tests/
|
+-- 04_figures/
|   +-- diabetes/
|   +-- heart_disease/
|   +-- cross_dataset/
|
+-- 05_report/
|
+-- 06_presentation/
|
+-- README.md
+-- requirements.txt
+-- .gitignore
```

## Datasets

### CDC Diabetes Health Indicators

The CDC Diabetes Health Indicators dataset is used as the primary case study.

Main characteristics:

- 253,680 observations
- 21 input features
- Binary diabetes target
- No missing values in the selected dataset
- Imbalanced class distribution

The binary target contains approximately:

- 86.1% non-diabetes cases
- 13.9% diabetes cases

Because of the class imbalance, the Logistic Regression models use balanced class weighting in the Diabetes experiments.

### UCI Cleveland Heart Disease

The UCI Cleveland Heart Disease dataset is used as the secondary validation dataset.

Main characteristics:

- 303 observations
- 13 input features
- Original target values from 0 to 4
- Binary target created using `num > 0`
- 6 missing values
- Approximately balanced binary classes

Missing values are handled using median imputation.

## Feature-Selection Problem

For each dataset, a feature subset is represented as a binary vector.

A value of:

- `1` means the feature is selected.
- `0` means the feature is excluded.

The search spaces are:

### Diabetes

21 candidate features:

```text
2^21 = 2,097,152 possible feature subsets
```

### Heart Disease

13 candidate features:

```text
2^13 = 8,192 possible feature subsets
```

The objective is to identify compact feature subsets while retaining strong predictive performance.

## Reinforcement-Learning Hyper-Heuristic

The proposed approach uses Q-learning as a hyper-heuristic controller.

Instead of directly selecting individual features, the RL agent learns which low-level feature-selection operator to apply during the search.

The available actions are:

### ADD

Adds one currently unselected feature.

### REMOVE

Removes one currently selected feature.

### SWAP

Removes one selected feature and replaces it with an unselected feature.

### FLIP

Toggles the selection state of one feature.

The RL state represents the approximate size of the current feature subset using three states:

- SMALL
- MEDIUM
- LARGE

The controller uses epsilon-greedy exploration and updates the Q-table according to the reward obtained after applying each heuristic.

## Machine-Learning Classifier

Logistic Regression is used as the primary predictive model during feature-subset evaluation.

Standardization is applied using `StandardScaler`.

For the Diabetes dataset, Logistic Regression uses:

```text
class_weight = balanced
```

because the target is imbalanced.

For the Heart Disease dataset, median imputation is performed before scaling and classification.

## Optimization Objective

The optimization objective balances predictive performance and subset compactness:

```text
Objective = ROC-AUC - 0.02 * (Selected Features / Total Features)
```

The objective therefore rewards high ROC-AUC while applying a small penalty for selecting more features.

The penalty coefficient used throughout the main experiments is:

```text
lambda = 0.02
```

## Compared Methods

The RL hyper-heuristic is evaluated against several alternative approaches.

### Full-Feature Baseline

Uses all available input features.

### Random Hyper-Heuristic

Randomly chooses among the available low-level heuristic operators without learning.

### Fixed Heuristic Strategies

The following fixed strategies are evaluated:

- ADD
- REMOVE
- SWAP
- FLIP

Each fixed strategy repeatedly applies only one type of low-level heuristic.

### SelectKBest

Filter-based feature selection using the ANOVA F-statistic.

### L1 Feature Selection

Embedded feature selection based on L1-regularized Logistic Regression.

### Recursive Feature Elimination

RFE recursively removes features based on Logistic Regression coefficients.

## Evaluation Metrics

The experiments evaluate:

- ROC-AUC
- Accuracy
- Precision
- Recall / Sensitivity
- F1-score
- Specificity
- Number of selected features
- Feature reduction percentage
- Performance-compactness objective

ROC-AUC is used as the primary predictive-performance metric.

## Statistical Analysis

Repeated experimental runs are evaluated using paired statistical testing.

The main statistical procedure uses:

- Paired Wilcoxon signed-rank tests
- Holm correction for multiple comparisons

The paired design compares methods evaluated using the same random seeds and data splits.

A corrected significance threshold of:

```text
p < 0.05
```

is used.

## Final Diabetes Evaluation

The final Diabetes comparison is based on repeated evaluation across 10 independent data splits.

| Method | Mean ROC-AUC | Std ROC-AUC | Mean Features | Mean Reduction | Mean Objective |
|---|---:|---:|---:|---:|---:|
| All Features | 0.8230 | 0.0014 | 21.00 | 0.0% | 0.8030 |
| L1 | 0.8224 | 0.0013 | 11.00 | 47.6% | 0.8119 |
| RFE | 0.8221 | 0.0016 | 9.80 | 53.3% | 0.8128 |
| RL | 0.8210 | 0.0016 | 13.80 | 34.3% | 0.8079 |
| SelectKBest | 0.8178 | 0.0015 | 8.00 | 61.9% | 0.8102 |

### Diabetes Interpretation

The full-feature model achieved the highest mean ROC-AUC at approximately 0.8230.

The RL hyper-heuristic achieved approximately:

- ROC-AUC: 0.8210
- 13.8 selected features on average
- 34.3% feature reduction

The difference in ROC-AUC between RL and the full-feature model was approximately:

```text
-0.0019
```

Therefore, RL retained predictive performance very close to the complete feature set while substantially reducing the number of predictors.

However, RFE and L1 achieved stronger overall performance-compactness trade-offs.

RFE achieved the highest mean optimization objective while using fewer than half of the original variables on average.

RL significantly outperformed SelectKBest in ROC-AUC, while SelectKBest achieved greater feature reduction.

## Final Diabetes Statistical Findings

After Holm correction:

- RL had significantly lower ROC-AUC than the all-feature model.
- RL used significantly fewer features than the all-feature model.
- RL achieved a significantly better performance-compactness objective than the all-feature model.
- L1 achieved significantly higher ROC-AUC and objective values than RL.
- RFE achieved significantly higher ROC-AUC and objective values than RL.
- RL achieved significantly higher ROC-AUC than SelectKBest.
- SelectKBest achieved a significantly better compactness-oriented objective than RL because it selected considerably fewer features.

These results indicate that RL provides a useful compromise between predictive performance and feature reduction, although it is not the strongest overall method on the Diabetes dataset.

## Final Heart Disease Evaluation

The final Heart Disease comparison is based on 30 repeated train-validation-test splits.

| Method | Mean ROC-AUC | Std ROC-AUC | Mean Features | Mean Reduction | Mean Objective |
|---|---:|---:|---:|---:|---:|
| All Features | 0.9032 | 0.0268 | 13.00 | 0.0% | 0.8832 |
| L1 | 0.8926 | 0.0281 | 8.00 | 38.5% | 0.8803 |
| RFE | 0.8911 | 0.0314 | 9.07 | 30.3% | 0.8772 |
| RL | 0.8791 | 0.0373 | 8.03 | 38.2% | 0.8667 |
| SelectKBest | 0.8905 | 0.0315 | 7.73 | 40.5% | 0.8786 |

### Heart Disease Interpretation

The RL hyper-heuristic selected approximately:

```text
8.03 of 13 features
```

on average, corresponding to approximately:

```text
38.2% feature reduction
```

However, its mean ROC-AUC was approximately:

```text
0.8791
```

compared with:

```text
0.9032
```

for the complete 13-feature model.

The mean difference was therefore approximately:

```text
-0.0241 ROC-AUC
```

The RL method also showed greater variability across repeated splits.

This indicates that aggressive feature selection on the much smaller Heart Disease dataset is less stable and more susceptible to validation-set variability.

## Final Heart Disease Statistical Findings

After Holm correction:

### RL vs All Features

RL showed:

- significantly lower ROC-AUC
- significantly fewer selected features
- significantly lower performance-compactness objective

### RL vs L1

RL showed:

- significantly lower ROC-AUC
- no significant difference in feature count
- significantly lower performance-compactness objective

### RL vs RFE

The differences in:

- ROC-AUC
- feature count
- objective

were not statistically significant after Holm correction.

### RL vs SelectKBest

- ROC-AUC difference was not statistically significant after correction.
- Feature-count difference was not significant.
- RL achieved a significantly lower combined objective.

## Heart Disease Feature-Selection Frequency

Across the 30 repeated RL experiments, the most frequently selected features were approximately:

| Feature | Selection Frequency |
|---|---:|
| sex | 80.0% |
| ca | 76.7% |
| cp | 76.7% |
| trestbps | 73.3% |
| chol | 63.3% |
| thal | 63.3% |
| slope | 63.3% |
| restecg | 60.0% |
| thalach | 60.0% |
| age | 56.7% |
| exang | 50.0% |
| oldpeak | 43.3% |
| fbs | 36.7% |

This suggests that the RL search is not completely random and repeatedly identifies certain variables as useful, even though the compact subsets do not consistently generalize as well as the full feature set.

## Cross-Dataset Findings

The two datasets demonstrate different behavior.

### Diabetes

The large Diabetes dataset provides a relatively stable search and evaluation environment.

RL:

- reduced the feature set by approximately one-third
- preserved ROC-AUC very close to the full-feature model
- achieved a better performance-compactness objective than using all features

### Heart Disease

The much smaller Heart Disease dataset produced greater variability.

RL:

- achieved substantial feature reduction
- experienced a larger predictive-performance loss
- produced less stable subsets across repeated splits

This suggests that RL-based feature selection is sensitive to:

- dataset size
- validation-set stability
- available training information
- search-space evaluation noise

## Main Findings

The main findings of the project are:

- A reinforcement-learning hyper-heuristic can successfully navigate a feature-selection search space by dynamically choosing among multiple low-level heuristics.
- RL can produce substantially smaller feature subsets.
- On the large Diabetes dataset, RL preserves predictive performance well while reducing feature count.
- The RL approach is competitive but does not universally outperform conventional feature-selection techniques.
- RFE and L1 provide particularly strong performance-compactness trade-offs on the Diabetes dataset.
- On the small Cleveland Heart Disease dataset, RL shows greater performance variability and weaker generalization.
- Feature-selection effectiveness depends strongly on dataset characteristics and validation stability.
- Statistical testing is necessary because small differences in mean ROC-AUC should not automatically be interpreted as meaningful improvements.

## Main Conclusion

The study demonstrates that reinforcement-learning-based hyper-heuristics are a viable approach for feature-selection search in disease-prediction tasks.

The method is capable of dynamically selecting search operators and discovering compact feature subsets without relying on a single fixed feature-selection strategy.

However, the experiments also demonstrate that RL-based feature selection is not universally superior to established methods.

On the Diabetes dataset, RL achieves a useful compromise between predictive performance and feature reduction, but RFE and L1 achieve stronger overall optimization results.

On the smaller Heart Disease dataset, RL feature selection is less stable and generally produces greater predictive-performance loss.

The overall findings therefore support the use of RL hyper-heuristics as an adaptive feature-selection strategy while highlighting the importance of dataset size, validation stability, conventional baselines, and rigorous statistical evaluation.

## Reproducibility

Install the required Python packages using:

```powershell
python -m pip install -r requirements.txt
```

The project uses the following main Python libraries:

- NumPy
- pandas
- scikit-learn
- SciPy
- statsmodels
- Matplotlib

Project paths are centralized in:

```text
02_scripts/project_paths.py
```

Datasets are stored in:

```text
01_data/
```

Python scripts are stored in:

```text
02_scripts/
```

Experimental results are stored in:

```text
03_results/
```

Final figures are stored in:

```text
04_figures/
```

The research report is stored in:

```text
05_report/
```

The project presentation is stored in:

```text
06_presentation/
```

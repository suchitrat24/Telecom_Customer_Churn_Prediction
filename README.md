# No-Churn Telecom Customer Churn Prediction

## Project Overview

This client project analyzes customer data for No-Churn Telecom, an established European telecom operator, and predicts which customers are likely to leave for a competitor, using data science and machine learning techniques.

The project was completed as part of the **DataMites AI Engineer Program (Internship Project)**.

Even after reducing tariffs and promoting more offers, the company's churn rate was well above 10%. The primary objective was to identify the variables that influence customers to leave, build a model that gives every customer a **Churn Risk Score** and a new **CHURN-FLAG** (YES = 1 / NO = 0), and give the retention, email campaign and customer care teams practical suggestions. The analysis covers **4,617 customers and 20 columns**.

## Project Objectives

- Perform data cleaning and exploratory data analysis (EDA).
- Identify the variables that influence customers to churn.
- Remove redundant columns and engineer total usage and billing features.
- Build and compare multiple classification models on an imbalanced target.
- Tune hyperparameters and evaluate advanced ensembles.
- Create honest Churn Risk Scores and a CHURN-FLAG for every customer.
- Provide data-driven suggestions to the retention, campaign and customer care teams.

## Technologies & Libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

## Project Workflow

### 1. Data Sourcing and Cleaning

- Checked structure, data types, nulls and duplicates (no null values and no duplicate rows).
- Converted the text target (`'True.'` / `'False.'`) to 1 / 0.
- Created **TOTAL MINS**, **TOTAL CALLS** and **TOTAL CHARGE** columns, plus grouping columns used only for plots.
- Converted `State`, `International Plan` and `VMail Plan` with Label Encoding and scaled features with `StandardScaler` for Logistic Regression and KNN.

### 2. Exploratory Data Analysis (EDA)
- Analyzed distributions of usage minutes, customer service calls, voice mail messages and total charge.
- Examined churn rates by international plan, voice mail plan, customer service calls, day minutes, total charge, account length, area code and state.
- Used chi-square tests to check whether differences between states, area codes and plans were real.
- Used histograms, relational plots, pair plots, boxplots, line plots, bar charts and correlation heatmaps.

### 3. Outlier Analysis
Used the Interquartile Range (IQR) method and boxplots to identify extreme values.

Key findings:
- Most columns had only 23 to 107 outliers, and all values were realistic.
- The 368 customers with 4+ service calls churned at 50.8%, the 27 customers with extreme Day Mins at 51.9%, and the 38 customers with extreme TOTAL CHARGE at 57.9%, against 14.2% on average.
- All outliers were kept because they are exactly the customers the company needs to find.

### 4. Feature Selection
- Dropped the four **Charge** columns, because each one is Minutes x a fixed price (correlation 1.00). Their combined effect is kept in `TOTAL CHARGE`.
- Dropped `TOTAL MINS` (correlation 0.89 with `TOTAL CHARGE`, which has the stronger link with churn).
- Dropped `Area Code` (no relation with churn, chi-square p = 0.67) and `State` (44-149 customers per state, and adding it did not improve the score).
- Final model used 15 features covering usage, calls, plans, customer service calls and total charge.

### 5. Machine Learning Model Development

This project uses supervised binary classification to predict whether a customer will churn. The predicted probability becomes the Churn Risk Score.

The following algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN) Classifier
4. Random Forest Classifier
5. Gradient Boosting Classifier

Class imbalance was handled with a stratified train/test split (80/20) and balanced class weights.

### 6. Model Evaluation and Comparison

Because only 14.2% of customers churned, a model that always predicts "no churn" would be 86% accurate and useless. Models were therefore evaluated using:
- ROC-AUC
- Precision, Recall, and F1 Score
- 5-fold stratified cross-validation

| Model | Test ROC-AUC | Precision | Recall | F1 | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | **0.920** | 0.958 | **0.870** | 0.912 | **0.923** |
| Random Forest | 0.914 | **0.966** | **0.870** | **0.916** | 0.918 |
| Decision Tree | 0.911 | 0.933 | 0.847 | 0.888 | 0.916 |
| KNN | 0.882 | 0.848 | 0.298 | 0.441 | 0.886 |
| Logistic Regression | 0.812 | 0.360 | 0.763 | 0.489 | 0.824 |

### 7. Hyperparameter Tuning and Advanced Ensembles
- **GridSearchCV** on Random Forest (`n_estimators` x `max_depth`) showed that shallow trees work best (depth 4, CV AUC 0.927). Deeper trees overfit, and the number of trees made almost no difference.
- **RandomizedSearchCV** on Random Forest and HistGradientBoosting.
- **Soft Voting** and **Stacking** ensembles.

| Model | Test ROC-AUC | PR-AUC | Precision | Recall | F1 | Churners in top 20% | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting (random search) | **0.946** | **0.917** | 0.974 | 0.863 | 0.915 | 88.5% | **0.925** |
| Random Forest (grid search) | 0.912 | 0.894 | 0.782 | **0.878** | 0.827 | 87.8% | 0.924 |
| Gradient Boosting (default) | 0.920 | 0.893 | 0.958 | 0.870 | 0.912 | 87.8% | 0.923 |
| Random Forest (random search) | 0.918 | 0.900 | 0.982 | 0.840 | 0.905 | 87.8% | 0.921 |
| Soft Voting | 0.930 | 0.911 | **0.991** | 0.870 | **0.927** | **89.3%** | 0.920 |
| Stacking | 0.930 | 0.911 | 0.974 | 0.870 | 0.919 | **89.3%** | 0.920 |

All tuned models and ensembles scored between **0.918 and 0.925** cross-validated ROC-AUC, so tuning moved the score by less than 0.01. The ensembles did not beat the best single model because the combined models miss the same churners. The limit is the information in the data rather than the algorithm.

**Best Performing Model: HistGradientBoosting Classifier (tuned)**

The tuned HistGradientBoosting achieved the highest cross-validated ROC-AUC (0.925) and the highest test ROC-AUC (0.946). On the test set of 924 customers it found 113 of 131 churners (86%) with 97% precision (only 3 false alarms out of 793 loyal customers).

### 8. Churn Risk Score and CHURN-FLAG

Every customer was scored with 5-fold out-of-fold predictions, so each score comes from a model that never saw that customer (out-of-fold ROC-AUC 0.924). The CHURN-FLAG cut-off was chosen where F1 is highest (0.68).

| Risk band | Customers | Actual churn rate | Share of all churners |
|---|---:|---:|---:|
| High Risk (CHURN-FLAG = YES) | 560 | 99.5% | 84.9% |
| Low Risk (CHURN-FLAG = NO) | 4,057 | 2.4% | 15.1% |

Below the cut-off the score could not rank customers any further (ROC-AUC 0.499 inside that group), so only two risk bands were used. All 560 flagged customers receive **HIGH** ticket priority, which is 12.1% of the customer base. Scores, flags and priorities were saved to `No_Churn_Customer_Risk_Scores.csv`.

## Key Findings

- The total bill had the largest effect: customers with a total charge **above 75** churned **74.4%** of the time (313 customers).
- Churn stayed at about 11% for 0-3 customer service calls, then rose to **44.4%** at 4 calls, **61.8%** at 5 and **62.2%** at 6+.
- International plan customers churned **41.7%**, compared with **11.3%** without the plan. Every plan holder who made 2 or fewer international calls (74 customers) or used more than 13 international minutes (76 customers) left.
- Voice mail plan customers were more loyal (**7.8%** vs **16.5%**). Among customers without voice mail, heavy day users churned **58.7%**, but among those with voice mail they churned only **7.2%**.
- Permutation importance ranked TOTAL CHARGE first (0.39 drop in AUC), followed by CustServ Calls, International Plan and VMail Plan.
- Account length and area code made almost no difference to churn.

## Retention and Customer Care Suggestions

- Offer a flat-rate or capped day-calls plan to customers whose bill goes above 70, and bundle voice mail into offers for heavy users.
- Treat the 4th customer service call as an alarm: auto-escalate to a senior agent, mark the ticket HIGH priority, and track first-call resolution.
- Offer low-use international plan holders a plan removal or pay-as-you-go option, and give heavy international users a cheaper per-minute rate or bundle.
- Send lucrative email offers only to CHURN-FLAG = YES customers and match the offer to the reason for leaving.
- Re-score customers every month, since usage, bills and service calls change monthly.
- The High Risk group is 12.1% of customers but **13.5% of monthly revenue**, so retaining them protects high-value revenue.
- Collect network quality, complaint type, contract end date and competitor offers to find the 15.1% of churners the current data cannot explain.

## Challenges and Solutions

Some of the main challenges encountered included:

- **Target stored as text:** Stripped the trailing full stop and mapped `'True.'` / `'False.'` to 1 / 0.
- **Redundant columns:** Replaced the four Charge columns with a single `TOTAL CHARGE` column.
- **Imbalanced target:** Used stratified splitting, balanced class weights, and ROC-AUC/recall/precision/F1 instead of accuracy.
- **Threshold-type patterns:** Used tree-based models, which capture sudden jumps that Logistic Regression cannot.
- **Overlapping risk factors:** Compared each factor inside the same customer group before making suggestions.
- **Honest risk scores:** Used out-of-fold predictions so no customer was scored by a model trained on them.

## Disclaimer

This project was completed for educational and portfolio purposes as part of the DataMites AI Engineer Program. The model results are based on historical customer data and should not be interpreted as a validated production churn prediction system.

## Author

**Suchitra Thanumalayan**

Data Science | Machine Learning | Artificial Intelligence

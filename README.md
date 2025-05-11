# Predict Premium Conversion – XYZ

## Project Goal
This project develops a predictive model for **Website XYZ**, a music-listening social networking platform using a freemium business model. The goal is to predict **which free users are likely to convert to premium subscribers** within 6 months after being targeted by a marketing campaign.

By accurately identifying likely adopters, XYZ can:
- Maximize ROI on marketing campaigns
- Personalize outreach to high-potential users
- Reduce spending on uninterested users

---

## Dataset Overview
The dataset `XYZData.csv` contains **41,540 user records** from a past campaign, with:
- `1,540` adopters (3.7%)  
- `25` features including demographics, listening behavior, friend stats, and 3-month activity deltas

---

## Business Metric Selection
Due to the high class imbalance and business need to identify most potential adopters, **Recall** was chosen as the primary evaluation metric. It emphasizes minimizing false negatives (missed opportunities) — crucial for targeted campaign success.

---

## Methodology

### Preprocessing
- Removed ID column
- Converted categorical variables (`adopter`, `male`, `good_country`)
- Checked and handled missing data
- Scaled and normalized continuous features

### Data Splitting
- 70% Train / 15% Validation / 15% Test split
- Applied **SMOTE** to training set to balance adopter classes

### Feature Selection
- Variable importance using:
  - Decision Tree (`rpart`)
  - Random Forest
  - Stepwise Logistic Regression
- Evaluated multicollinearity and correlation structure
  - Pearson and Point Biserial Correlations
  - Heatmaps for visual insight

### Models Explored
- **Decision Tree** (depth-limited)
- **Random Forest** (tuned with importance analysis)
- **Stepwise Logistic Regression**
- **Final Logistic Regression** (on refined features)

### Evaluation
- Confusion Matrix
- Accuracy, Recall, F1 Score, AUC
- ROC Curve (Validation + Test)
- Final model tested and validated on holdout data

  ## Collaborators
- Raman Chowdhury
- Dharmpalsinh
- Ruth

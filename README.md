# 6103-sp-25-Patil
# Prediction of Ship Type and Course Over Ground (COG) from AIS Dataset
## DATS 6103 Data Mining – Final Project

**Student:**  
- Mayur Patil  

**Instructor:** Sushovan Majhi  
**Semester:** Fall 2025  

---

## Repository Structure
text
#AIS-COG-Prediction/
#├── Model.ipynb
#├── Q&A.pdf
#├── README.md
#└── data_preprocessing.py


# Features:
#   Numerical:
#       - speed (sog)
#       - course over ground (cog)
#       - heading
#       - draught
#       - width
#       - length
#
#   Categorical:
#       - ship type
#       - navigational status
#
# -------------------------------------------------------------
# Project Timeline
#
# Phase 1: Data Preparation (Nov 22–24)
#   - Load dataset
#   - Exploratory Data Analysis (EDA)
#   - Handle missing values
#   - Remove duplicates
#   - Merge rare ship type categories
#
# Phase 2: Feature Engineering (Nov 23–29)
#   - Outlier removal
#   - Scaling / normalization
#   - One-hot encoding of ship type
#   - Random Forest feature importance
#   - VIF (multicollinearity check)
#   - PCA / SVD for dimensionality analysis
#
# Phase 3: Modeling & Presentation (Nov 30–Dec 2)
#   - COG regression modeling
#   - Evaluation metrics & statistical tests
#   - Plots & visualizations
#   - Final presentation slides

# 6103-sp-25-Patil
# Prediction of Ship Type and Course Over Ground (COG) from AIS Dataset
## DATS 6103 Data Mining – Final Project

**Student:**  
- Mayur Patil  

**Instructor:** Sushovan Majhi  
**Semester:** Fall 2025  

---

## Repository Structure
```text
AIS-COG-Prediction/
├── Proposal/
│   └── project_proposal.pdf
├── Data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
├── Code/
│   ├── notebooks/
│   │   ├── 01_data_preprocessing.ipynb
│   │   ├── 02_feature_engineering.ipynb
│   │   ├── 03_regression_model.ipynb
│   │   └── 04_model_evaluation.ipynb
│   ├── scripts/
│   │   ├── preprocess.py
│   │   ├── feature_selection.py
│   │   ├── train_regression.py
│   │   └── evaluate_model.py
│   ├── utils/
│   ├── results/
│   │   ├── figures/
│   │   ├── models/
│   │   └── metrics/
│   ├── requirements.txt
│   └── README.md
└── Presentation/
    └── Final_Presentation_Slides.pdf
```



## Data Source
The dataset used in this project is AIS (Automatic Identification System) maritime tracking data.

- **Source:** Denmark Maritime Authority – AIS Open Access Database  
- **Region:** Kattegat Strait  
- **Period:** January 1 – March 10, 2022  
- **Total Records:** ~358,351 AIS messages  
- **Features Include:**  
  - Numerical: speed, heading, draught, width, length  
  - Categorical: ship type, navigational status  



## Project Timeline

| Phase | Dates | Tasks |
|-------|-------|-------|
| **Phase 1: Data Preparation** | Nov 16–22 | Load dataset, EDA, handle missing values, remove duplicates, merge rare categories |
| **Phase 2: Feature Engineering** | Nov 23–29 | Outlier removal (K-Means), scaling, encoding, Random Forest importance, VIF, PCA/SVD |
| **Phase 3: Modeling & Presentation** | Nov 30–Dec 2 | Regression modeling, evaluation metrics, statistical tests, plots, presentation slides |

Categorical: ship type, navigational status

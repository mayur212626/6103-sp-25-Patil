# 6103-sp-25-Patil
Prediction of Ship Type and Course Over Ground (COG) from AIS Dataset
DATS 6103 – Data Mining – Final Project

Student:

Mayur Patil

Instructor: Sushovan Majhi
Presentation Date: December 2, 2024
Semester: Fall 2024

DATS6103 – Data Mining | Final Project | Fall 2024

Project Overview

This project focuses on developing a regression-based analytical system to predict a vessel’s Course Over Ground (COG) and analyze ship type characteristics using Automatic Identification System (AIS) data.

The project includes:

Comprehensive data preprocessing
Handling missing values, duplicates, outliers, categorical encoding, and scaling

Feature engineering & dimensionality reduction
Using Random Forest, VIF, PCA, and SVD

Regression modeling for COG prediction
Linear Regression + Stepwise Backward Regression

Statistical & performance evaluation
R², MSE, AIC, BIC, t-test, F-test, confidence intervals

Goal: Build a reliable model to support maritime navigation analytics and vessel behavior understanding.

Repository Structure
AIS-COG-Prediction/
├── Proposal/
│   └── project_proposal.pdf               # Your project proposal
├── Data/
│   ├── raw/                               # Original AIS data
│   ├── cleaned/                           # After preprocessing
│   └── processed/                         # Final modeling data
├── Code/
│   ├── notebooks/                         # Jupyter Notebooks (EDA & modeling)
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

Data Source

Dataset used:

Denmark Maritime Authority – AIS Open Data

Region: Kattegat Strait

Period: Jan 1 – Mar 10, 2022

Records: ~358,351 entries

Attributes:

Numerical: speed, heading, draught, width, length, etc.

Categorical: ship type, navigational status

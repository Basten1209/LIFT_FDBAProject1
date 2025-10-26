# Introducing LIFT (Lasso-Integrated Factor Thresholding): A Unified Framework for High-Dimensional Covariance Estimation
This study proposes an Integrated covariance estimation framework for high dimensional financial data that unifies sparse selection, structural regularization, and latent factor modeling. 

Keywords : High-dimensionality, LASSO, POET, Hierarchical Clustering, Robust Estimation, Portfolio Optimization.

This project was carried out as part of "IMEN891M : ST: Financial Big Data Analysis" (Prof. Minseok Shin).

## Repo Structure Explanation
```
.
├── LIFT_Model.py
├── daily_~.csv, .pkl
├── covanalyzer.py
├── covanalyzer2.py
├── data/
│   ├── data_preprocessing.R
│   └── raw/
├── explanation/
├── plots/
├── presentation/
└── result/
```
- `LIFT_Model.py`: Core Python implementation that integrates lasso selection with factor thresholding.
- `daily_~.csv, .pkl` : Datas in the intermediate course after script execution (asset selection, covariance matrix, etc.)
- `covanalyzer.py` · `covanalyzer2.py`: Exploratory scripts for alternative covariance analyses and sensitivity checks.
- `data/`: Stores preprocessing scripts and both processed and raw market datasets.
- `explanation/`: Contains narrative documentation that walks through analytical decisions and findings.
- `plots/`: Collects generated figures summarizing covariance structures, clustering, and performance metrics.
- `presentation/`: Hosts final presentation materials shared with the course.
- `result/`: Houses generated outputs such as summary tables, figures, and Python helpers for reporting.

## Collaborator

| Name | Contributions |
| --- | --- |
| Woohyeok Choi (Lead) | Coordinated the overall progress of the project, actively participating in topic selection and model design. |
| Seungjun Oh | Mainly contributed to model implementation and data preprocessing. |
| Isac Johnsson | Led the report writing and refinement process, supporting data collection and presentation preparation. |

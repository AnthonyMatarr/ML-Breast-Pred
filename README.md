# Development, Validation, and Deployment of a Machine Learning Risk Calculator for 30-Day Complications after mastectomy and delayed breast reconstruction/revision

## Description
This project implements Logistic Regression, XGBoost, LightGBM, Neural Network, and Stacked Generalization models to predict post-operative complications (Surgical-related, Medical-related, Mortality, Unplanned Reoperation, Venous Thromboembolism) in mastectomy and delayed breast reconstruction/revision patients found in the National Surgical Quality Improvement Program (NSQIP) dataset spanning 2008-2024.

## Associated Risk Calculator
A web application was developed to deploy calibrated XGBoost models for Surgical, Medical, Unplanned Reoperation, Venous Thromboembolism, and LightGBM for Mortality. The interface can be found [here](https://pro-breast.streamlit.app/).
The app may also be run locally with command `uv run streamlit run app/base_app.py`, once all [Installation Steps](#installation) are completed.


### Features
- Select any available outcome from the sidebar
- Input patient values into appropriate fields (all features used as model inputs)
- View results:
  - Risk bin allocation
  - Feature contribution via regularized SHAP explanation values
  - Calibrated risk probability
  - Percentile of model output relative to all test cohort patients
  - View imputed numerical values as appropriate

## Project layout
### Included directories
- `notebooks/`: end-to-end machine learning workflows 
  - data cleaning, preprocessing, tuning, training, evaluation, feature importance, and figure/table generation
  - assumes access to files in `data/` and `models/`

- `src/`: reusable functions shared across the project
- `app/`: all code related to interface
  - `base_app.py`: source code for calculator interface
  - `.streamlit/`: Interface styling
  - `deployment_prep.ipynb`: helper file to copy relevant data into `app/`
  - `display_functions.py`: helper file containing modules displayed on interface

### Omitted directories
- ```data/```: raw + processed data, preprocessing pipelines
- ```models/```: raw trained and calibrated models for each outcome
- ```logs/```: logged data produced during model tuning
- ```results/```: figures and tables

## Installation

### Prerequisites
Ensure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your system.

### Steps
1. Clone this repository with HTTPS or SSH:

- Using HTTPS (**recommemnded for simplicity**):
```
git clone https://github.com/AnthonyMatarr/ML-Breast-Pred.git
```
- Using SSH:
```
git clone git@github.com:AnthonyMatarr/ML-Breast-Pred.git
```
2. Navigate to project directory
```
cd ML-Breast-Pred
```
3. Ensure data/object integrity and consistency
```
git fsck --full
```
4. Sync environment from pyproject.toml and uv.lock files:
```
uv sync --locked
```

## Usage
1. Adjust BASE_PATH in src/config.py to the absolute path to the root directory of ML-Breast-Pred, for example:
```
BASE_PATH = Path("/Users/<user_name>/Downloads/ML-Breast-Pred")
```
2. Run notebooks
  - Notebooks are numbered by stage, but assuming necessary data is available, can be run on their own
  - **NOTE**: Due to OS/architecture differences and solver choices, despite a consistent random state/seed used throughout the project, minor numerical deviations from the manuscript may occur in model tuning/training/evaluation and SHAP values

    
## Custom Modifications
- [MLstakit](https://github.com/Brritany/MLstatkit) was forked and slightly altered, with some code appended to `MLstatkit/metrics.py` and `MLstatkit/ci.py` to add bin event rate, ICI, and Brier functionality.
- This change should be consistent once the repo is cloned and `uv sync --locked` is run, however to view these changes or ensure their consistency, the forked repo can be found [here](https://github.com/AnthonyMatarr/MLstatkit), or in the project directory at:
```
/.venv/lib/python3.12/site-packages/MLstatkit/
```
## License: MIT
- Code licensed under MIT
- No patient data are included


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import SEED
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
from shutil import rmtree
import joblib


def remove_prefix(df):
    X = df.copy()
    X.columns = X.columns.str.replace(r"^\w+__", "", regex=True)
    return X


def clip_and_round_asa(X):
    """
    Round ASA values to nearest int, clip to [1, 4], and cast to int.
    Expects a 2D array-like (n_samples, n_features).
    """
    X = np.asarray(X, dtype=float)
    X = np.rint(X)  # round to nearest integer
    X = np.clip(X, 1, 4)  # bound to 1–4
    return X.astype(int)


class BMICalculatorArray(BaseEstimator, TransformerMixin):
    def __init__(self, height_idx, weight_idx):
        self.height_idx = height_idx
        self.weight_idx = weight_idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        height = X[:, self.height_idx]
        weight = X[:, self.weight_idx]
        bmi = (weight * 703) / (height**2)
        # Remove height and weight columns
        mask = np.ones(X.shape[1], dtype=bool)
        mask[[self.height_idx, self.weight_idx]] = False
        X_new = X[:, mask]
        # Append BMI as last column
        X_new = np.column_stack([X_new, bmi])
        return X_new.astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        # Remove height and weight, add BMI
        if input_features is None:
            input_features = [
                f"num_{i}" for i in range(self.height_idx + self.weight_idx + 1)
            ]
        input_features = list(input_features)
        # Remove height and weight
        features = [
            f
            for i, f in enumerate(input_features)
            if i not in [self.height_idx, self.weight_idx]
        ]
        features.append("BMI")
        return np.array(features)


def transform_export_data(
    X, y, outcome_name, preprocessor, data_path=None, pipeline_path=None
):
    """
    Split, preprocess, and export train/val/test datasets for a given outcome.

    Performs stratified train-val-test split (70-15-15), fits the preprocessor on
    training data, transforms all splits, converts columns to numeric types, and
    optionally exports the processed datasets and fitted preprocessor to disk.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix containing predictor variables for the full dataset.
    y : pd.Series
        Target variable (binary outcome labels) corresponding to X.
    outcome_name : str
        Name identifier for the outcome (e.g., 'asp', 'mort'). Used for file naming
        and directory structure when saving outputs.
    preprocessor : sklearn.pipeline.Pipeline or sklearn.compose.ColumnTransformer
        Scikit-learn preprocessing pipeline to fit on training data and transform
        all splits. Must implement fit(), transform(), and get_feature_names_out().
    data_path : pathlib.Path or str, optional
        Base directory path where processed train/val/test parquet and Excel files
        will be saved. If None, data is not saved to disk. Default: None.
    pipeline_path : pathlib.Path or str, optional
        Directory path where the fitted preprocessor will be saved as a compressed
        joblib file. If None, preprocessor is not saved to disk. Default: None.

    Returns
    -------
    dict
        Dictionary containing six DataFrames/Series with keys:
        - 'X_train': pd.DataFrame - Preprocessed training features
        - 'y_train': pd.Series - Training labels
        - 'X_val': pd.DataFrame - Preprocessed validation features
        - 'y_val': pd.Series - Validation labels
        - 'X_test': pd.DataFrame - Preprocessed test features
        - 'y_test': pd.Series - Test labels

    Notes
    -----
    - Train-val-test split uses 70-15-15 proportions with stratification by outcome.
    - The preprocessor is fit only on training data to prevent data leakage.
    - All feature columns are converted to numeric types; conversion failures are logged.
    - Column name prefixes (e.g., from ColumnTransformer) are removed via remove_prefix().
    - If output paths exist, existing files/directories are overwritten with a warning.
    - Saved files use parquet format for features (efficient storage) and Excel for labels.

        Warnings
    --------
    UserWarning
        Raised when overwriting existing data or preprocessor files.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.compose import ColumnTransformer
    >>>
    >>> preprocessor = ColumnTransformer([
    ...     ('scaler', StandardScaler(), ['AGE', 'BMI'])
    ... ])
    >>>
    >>> data_splits = transform_export_data(
    ...     X=features_df,
    ...     y=labels_series,
    ...     outcome_name='mortality',
    ...     preprocessor=preprocessor,
    ...     data_path=Path('data/processed'),
    ...     pipeline_path=Path('models/preprocessors')
    ... )
    >>> print(data_splits['X_train'].shape)
    """
    print("\t Splitting data...")
    ##Get train set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    ##Get val + test set
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    print("\t Fitting preprocessor...")
    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out()
    print("\t Transforming train, val, and test sets...")
    X_train_transformed = np.array(preprocessor.transform(X_train))
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_train_transformed = remove_prefix(X_train_transformed)

    X_val_transformed = np.array(preprocessor.transform(X_val))
    X_val_transformed = pd.DataFrame(X_val_transformed, columns=feature_names)
    X_val_transformed = remove_prefix(X_val_transformed)

    X_test_transformed = np.array(preprocessor.transform(X_test))
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
    X_test_transformed = remove_prefix(X_test_transformed)

    # Reset index
    X_train_transformed.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_val_transformed.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)
    X_test_transformed.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    print("\t Making all columns numeric...")
    ## Make all columns numeric for ML models
    for col in X_train_transformed.columns:
        try:
            X_train_transformed[col] = pd.to_numeric(X_train_transformed[col])
        except Exception as e:
            print(f"Column {col} failed: {e}")

    for col in X_val_transformed.columns:
        try:
            X_val_transformed[col] = pd.to_numeric(X_val_transformed[col])
        except Exception as e:
            print(f"Column {col} failed: {e}")

    for col in X_test_transformed.columns:
        try:
            X_test_transformed[col] = pd.to_numeric(X_test_transformed[col])
        except Exception as e:
            print(f"Column {col} failed: {e}")

    ### Save processed data ###
    if data_path:
        print("\t Saving processed data...")
        data_path = data_path / outcome_name
        if data_path.exists():
            warnings.warn(f"Over-writing tabular data at path: {data_path}")
            rmtree(data_path)
        data_path.mkdir(exist_ok=False, parents=True)

        ## Save transformed data
        X_train_transformed.to_parquet(data_path / "X_train.parquet")
        y_train.to_excel(data_path / "y_train.xlsx")
        X_val_transformed.to_parquet(data_path / "X_val.parquet")
        y_val.to_excel(data_path / "y_val.xlsx")
        X_test_transformed.to_parquet(data_path / "X_test.parquet")
        y_test.to_excel(data_path / "y_test.xlsx")
    ### Save fitted preprocessor/pipeline ###
    if pipeline_path:
        print("\t Saving pipeline...")
        preprocessor_path = pipeline_path / f"{outcome_name}_pipeline.joblib"
        if preprocessor_path.exists():
            warnings.warn(f"Over-writing tabular data at path: {data_path}")
            preprocessor_path.unlink()
        preprocessor_path.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(preprocessor, preprocessor_path, compress=3)
    return {
        "X_train": X_train_transformed,
        "y_train": y_train,
        "X_val": X_val_transformed,
        "y_val": y_val,
        "X_test": X_test_transformed,
        "y_test": y_test,
    }

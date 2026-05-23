import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import SEED
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
from shutil import rmtree
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    MinMaxScaler,
    FunctionTransformer,
)
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from src.data_utils import get_feature_lists


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


def get_pipeline(num_cols, nom_cols, ord_cols, bin_cols):
    ################# Numerical pipeline #################
    # ====>Impute, calculate BMI, then scale
    height_idx = num_cols.index("HEIGHT")
    weight_idx = num_cols.index("WEIGHT")
    # Only age is ~normal --> best to use this over StandardScaler()
    num_pipeline = Pipeline(
        [
            (
                "imputer",
                IterativeImputer(
                    estimator=None,  # default = BayesianRidge
                    initial_strategy="median",
                    max_iter=10,
                    sample_posterior=False,  # deterministic
                ),
            ),
            ("bmi", BMICalculatorArray(height_idx=height_idx, weight_idx=weight_idx)),
            ("scaler", MinMaxScaler()),
        ]
    )
    ################# Ordinal pipeline #################
    # ==============> Separate imputer/encoder for ASA
    asa_col = ["ASACLAS"]
    # asa_pipeline = Pipeline([("encoder", OrdinalEncoder(categories=[[1, 2, 3, 4]]))])

    asa_pipeline = Pipeline(
        steps=[
            # 1. Imputation on the ASA column
            (
                "imputer",
                IterativeImputer(
                    estimator=None,  # default = BayesianRidge
                    initial_strategy="median",
                    max_iter=10,
                    sample_posterior=False,  # deterministic
                ),
            ),
            # 2. Round to nearest integer, cast to int
            (
                "round_to_int",
                FunctionTransformer(
                    clip_and_round_asa,
                    feature_names_out="one-to-one",
                ),
            ),
            # 3. Ordinal encoding just in case
            (
                "encoder",
                OrdinalEncoder(categories=[[1, 2, 3, 4]]),
            ),
        ]
    )

    # ==============> Separate encoder for all other ordinals (0, 1, 2+)
    other_ordinal_cols = [col for col in ord_cols if col != "ASACLAS"]
    num_other_ordinal = len(other_ordinal_cols)
    other_ordinal_pipeline = Pipeline(
        [
            (
                "encoder",
                OrdinalEncoder(
                    categories=[["0", "1", "2+"]]
                    * num_other_ordinal  # Repeat for each column
                ),
            )
        ]
    )
    ################# Nominal pipeline #################
    # =========> One-hot encode
    nom_pipeline = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])

    ################# Combine all preprocessing #################
    preprocessor = ColumnTransformer(
        [
            ("num", num_pipeline, num_cols),
            ("cat", nom_pipeline, nom_cols),
            ("ord_asa", asa_pipeline, asa_col),
            ("ord_other", other_ordinal_pipeline, other_ordinal_cols),
            ("bin", "passthrough", bin_cols),
        ]
    )
    return preprocessor


def transform_export_data_dev(
    df,
    x_cols,
    target_col_name,
    tst_yr,
    include_yrs,
    data_path=None,
    pipeline_path=None,
):
    """
    Split development (2008-2023) data into train-val-test (70-15-15)
    for a given outcome

    Fits the preprocessor on training data, transforms all splits, converts columns to numeric types,
    and optionally exports the processed datasets and fitted preprocessor to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix containing predictor variables for the full dataset, along with
        all target variables. Columns will be further subset with `x_cols`
        Assumes this includes 2024 data (and removes it)
    x_cols: list[str]
        Names of predictor variables
    target_col_name : str
        Name of target column in X
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

    Warnings
    --------
    UserWarning
        Raised when overwriting existing data or preprocessor files.
    """
    df_dev = df[df["OPERYR"].isin(include_yrs)]
    df_test = df[df["OPERYR"] == tst_yr]
    df_exclude = df[(df["OPERYR"] != tst_yr) & (~df["OPERYR"].isin(include_yrs))]
    assert len(df_dev) + len(df_test) + len(df_exclude)
    df_X_dev = df_dev[x_cols].copy()
    X_test = df_test[x_cols].copy()
    df_y_dev = df_dev[target_col_name].copy()
    y_test = df_test[target_col_name].copy()

    # split into train-temp sets (80-20)
    X_train, X_val, y_train, y_val = train_test_split(
        df_X_dev, df_y_dev, train_size=0.8, random_state=SEED, stratify=df_y_dev
    )

    ## Get processor
    feat_list_dict = get_feature_lists(X_train)
    preprocessor = get_pipeline(
        num_cols=feat_list_dict["numerical_cols"],
        nom_cols=feat_list_dict["nominal_cols"],
        ord_cols=feat_list_dict["ordinal_cols"],
        bin_cols=feat_list_dict["binary_cols"],
    )
    ## Fit processor on train
    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out()
    ## Transform all
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
        data_path = data_path / target_col_name
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
        preprocessor_path = pipeline_path / f"{target_col_name}_pipeline.joblib"
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


def transform_export_data_full(
    df,
    x_cols,
    target_col_name,
    data_path=None,
    pipeline_path=None,
):
    """
    Split, preprocess, and export train/val/test datasets for a given outcome.

    Splits into train (OperYr: 2008-2021), val (OperYr 2022-2023) and evaluation (OpYr: 2024) data.

    Fits the preprocessor on training data, transforms all splits, converts columns to numeric types,
    and optionally exports the processed datasets and fitted preprocessor to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix containing predictor variables for the full dataset, along with
        all target variables. Columns will be further subset with `x_cols`
    x_cols: list[str]
        Names of predictor variables
    target_col_name : str
        Name of target column in X
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

    Warnings
    --------
    UserWarning
        Raised when overwriting existing data or preprocessor files.
    """
    df_sub = df[df["OPERYR"] >= 2014]
    train_years = list(range(2014, 2022))  # 2014-2021
    train_set = df_sub[df_sub["OPERYR"].isin(train_years)]
    val_set = df_sub[df_sub["OPERYR"].isin([2022, 2023])]  # 2022-2023
    test_set = df_sub[df_sub["OPERYR"] == 2024]  # 2024
    assert len(train_set) + len(val_set) + len(test_set) == len(df_sub)

    X_train = train_set[x_cols].copy()
    y_train = train_set[target_col_name].copy()

    X_val = val_set[x_cols].copy()
    y_val = val_set[target_col_name].copy()

    X_test = test_set[x_cols].copy()
    y_test = test_set[target_col_name].copy()
    ## Will be the same for every outcome, but initialize here for simplicity
    test_ids = test_set["CASEID"].copy()

    ## Get processor
    feat_list_dict = get_feature_lists(X_train)
    preprocessor = get_pipeline(
        num_cols=feat_list_dict["numerical_cols"],
        nom_cols=feat_list_dict["nominal_cols"],
        ord_cols=feat_list_dict["ordinal_cols"],
        bin_cols=feat_list_dict["binary_cols"],
    )
    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out()

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
    test_ids.reset_index(drop=True, inplace=True)

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
        data_path = data_path / target_col_name
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
        test_ids.to_excel(data_path / "test_ids.xlsx")

    ### Save fitted preprocessor/pipeline ###
    if pipeline_path:
        preprocessor_path = pipeline_path / f"{target_col_name}_pipeline.joblib"
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

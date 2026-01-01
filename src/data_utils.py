import pandas as pd
import numpy as np
from src.config import BASE_PATH
import joblib

# Although not used, need import here for loading calibrated models
from src.nn_model import load_nn_clf


def import_raw_data_dict(import_dir, verbose=False):
    data_dict = {}
    for file in import_dir.iterdir():
        file_name = file.stem
        file_ext = file.suffix
        if file.is_dir():
            continue
        if file_ext == ".parquet":
            if verbose:
                print(f"Working on file: {file_name}...")
            file_import = pd.read_parquet(file)
        else:
            print(f"\t File extension must be parquet got {file_ext} instead.")
            print("\t Skipping file...")
            continue
        data_dict[file_name] = file_import
    assert len(data_dict) == 17
    return data_dict


def get_feature_lists(df):
    """
    Classify all features/columns in a dataframe as one of:
        binary, nominal, ordinal, categorical using the number of unique values
        in that column
    """
    ## Initialize feature lists w/ exceptions to rules
    binary_cols = []
    # nominal_cols = ["DIABETES", "SURGINDICD", "ANESTHES", "RACE"]
    nominal_cols = []
    ordinal_cols = ["ASACLAS"]
    numerical_cols = []
    for col in df.columns:
        ## If column already classified, skip
        if col in (binary_cols + nominal_cols + numerical_cols + ordinal_cols):
            continue
        unique_vals = df[col].unique()
        n_unique = len(unique_vals)
        if n_unique == 2:
            binary_cols.append(col)
        elif n_unique == 3:
            if "1" in [str(val) for val in unique_vals]:
                ordinal_cols.append(col)
            else:
                nominal_cols.append(col)
        elif n_unique <= 10:
            nominal_cols.append(col)
        else:
            numerical_cols.append(col)
        ## Ensure all columns classified
    return {
        "binary_cols": list(binary_cols),
        "numerical_cols": list(numerical_cols),
        "nominal_cols": list(nominal_cols),
        "ordinal_cols": list(ordinal_cols),
    }


def get_data(outcome_folder, file_dir=BASE_PATH / "data" / "processed/"):
    """
    For a given outcome, get X/y train, validation, and testing data
    """
    file_path = file_dir / outcome_folder
    return {
        "X_train": pd.read_parquet(file_path / "X_train.parquet"),
        "y_train": pd.read_excel(file_path / "y_train.xlsx", index_col=0),
        "X_val": pd.read_parquet(file_path / "X_val.parquet"),
        "y_val": pd.read_excel(file_path / "y_val.xlsx", index_col=0),
        "X_test": pd.read_parquet(file_path / "X_test.parquet"),
        "y_test": pd.read_excel(file_path / "y_test.xlsx", index_col=0),
    }


def get_models(model_prefix_list, outcome, file_dir=BASE_PATH / "models"):
    """
    For a given outcome, get all models that predict that outcome
    """
    model_dict = {}
    for model_name in model_prefix_list:
        model = joblib.load(file_dir / outcome / f"{model_name}.joblib")
        model_dict[model_name] = model
    return model_dict

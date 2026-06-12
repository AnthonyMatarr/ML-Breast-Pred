from src.config import SEED
from src.data_utils import log

import argparse
import copy
import joblib
from pathlib import Path

import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def import_data(import_path, in_dim=None):
    suffix = import_path.suffix
    ## MODELS
    if suffix == ".joblib":
        imp_data = joblib.load(import_path)
    elif suffix == ".pt":
        from src.nn_model import load_nn_clf

        imp_data = load_nn_clf(data_path=import_path, in_dim=in_dim, device="cpu")
    elif suffix == ".parquet":
        imp_data = pd.read_parquet(import_path)
    elif suffix == ".csv":
        imp_data = pd.read_csv(import_path, index_col=0)
    elif suffix == ".tsv":
        imp_data = pd.read_csv(import_path, index_col=0, sep="\t")
    elif suffix == ".xlsx":
        imp_data = pd.read_excel(import_path, index_col=0)
    else:
        raise ValueError(
            f"Unrecognized file: {import_path.name} with suffix: {import_path.suffix}"
        )
    return imp_data


def get_ohe_cols(df):
    """
    Extract original column names and categories from one-hot encoded features.

    Parses column names in a dataframe to identify and group one-hot encoded features
    by their original categorical variable names. Assumes one-hot encoded columns follow
    the naming convention: '<ORIGINAL_NAME>_<CATEGORY_VALUE>'.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing one-hot encoded columns with underscore-delimited names.

    Returns
    -------
    dict of {str: list of str}
        Maps original categorical column names to lists of their encoded category values.
        For example:
        {
            'SEX': ['MALE', 'FEMALE'],
            'RACE': ['WHITE', 'BLACK', 'ASIAN'],
        }

    Notes
    -----
    - Columns without underscores are assumed to be non-encoded and are skipped
    """
    ohe_dict = {}
    for col in df.columns:
        col_split = col.split("_")
        if len(col_split) == 1:
            continue
        col_name = col_split[0]
        instance_name = "_".join(col_split[1:])

        if col_name in ohe_dict.keys():  # if already in dict append
            ohe_dict[col_name].append(instance_name)
        else:  # if not in dict add
            ohe_dict[col_name] = [instance_name]
    return ohe_dict


def combine_encoded(shap_values, name, mask, return_original=True):
    """
    Combine SHAP values for one-hot encoded features into a single feature importance score.

    Aggregates SHAP values from multiple one-hot encoded columns (e.g., SEX_MALE, SEX_FEMALE)
    into a single SHAP value representing the original categorical feature (e.g., SEX).
    This enables interpretation of categorical features as unified entities rather than
    separate binary indicators.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP Explanation object containing values, feature names, and metadata for all features.
    name : str
        Name for the combined categorical feature (e.g., 'SEX' for SEX_MALE and SEX_FEMALE).
    mask : array-like of bool
        Boolean mask indicating which features to combine. True values mark the one-hot
        encoded columns belonging to the same categorical feature.
    return_original : bool, default=True
        If True, returns both the combined Explanation and the original subset Explanation.
        If False, returns only the combined Explanation.

    Returns
    -------
    shap.Explanation or tuple of (shap.Explanation, shap.Explanation)
        If return_original=True:
            (combined_sv, original_subset_sv) where:
            - combined_sv: Explanation with one-hot features combined into single feature
            - original_subset_sv: Explanation containing only the encoded features
        If return_original=False:
            combined_sv: Explanation with combined features

    Notes
    -----
    - SHAP values are summed across encoded columns to create the combined importance
    - The display data shows which specific category was active for each instance
    - Original feature data is encoded as integers representing category indices
    - All metadata (base_values, instance_names, etc.) is preserved in the output

    Implementation Details
    ----------------------
    Adapted from: https://gist.github.com/peterdhansen/ca87cc1bfbc4c092f0872a3bfe3204b2

    The combination process:
    1. Extracts SHAP values for masked (encoded) features
    2. Sums SHAP values across encoded columns
    3. Reconstructs feature data to show active category
    4. Concatenates non-encoded features with new combined feature
    5. Preserves all SHAP Explanation metadata
    """
    mask = np.array(mask)
    mask_col_names = np.array(shap_values.feature_names, dtype="object")[mask]
    sv_name = shap.Explanation(
        shap_values.values[:, mask],
        feature_names=list(mask_col_names),
        data=shap_values.data[:, mask],
        base_values=shap_values.base_values,
        display_data=shap_values.display_data,
        instance_names=shap_values.instance_names,
        output_names=shap_values.output_names,
        output_indexes=shap_values.output_indexes,
        lower_bounds=shap_values.lower_bounds,
        upper_bounds=shap_values.upper_bounds,
        main_effects=shap_values.main_effects,
        hierarchical_values=shap_values.hierarchical_values,
        clustering=shap_values.clustering,
    )

    new_data = (sv_name.data * np.arange(sum(mask))).sum(axis=1).astype(int)  # type: ignore
    svdata = np.concatenate(
        [shap_values.data[:, ~mask], new_data.reshape(-1, 1)], axis=1
    )

    if shap_values.display_data is None:
        svdd = shap_values.data[:, ~mask]
    else:
        svdd = shap_values.display_data[:, ~mask]

    svdisplay_data = np.concatenate(
        [svdd, mask_col_names[new_data].reshape(-1, 1)], axis=1
    )

    # Handle multi-class (3D) vs binary/regression (2D) SHAP arrays
    if len(shap_values.values.shape) == 3:  # Multi-class case
        # Sum encoded features while preserving class dimension
        new_values = sv_name.values.sum(axis=1, keepdims=True)  # type: ignore
        svvalues = np.concatenate([shap_values.values[:, ~mask, :], new_values], axis=1)
    else:  # Binary/regression case
        new_values = sv_name.values.sum(axis=1)  # type: ignore
        svvalues = np.concatenate(
            [shap_values.values[:, ~mask], new_values.reshape(-1, 1)], axis=1
        )

    svfeature_names = list(np.array(shap_values.feature_names)[~mask]) + [name]

    sv = shap.Explanation(
        svvalues,
        base_values=shap_values.base_values,
        data=svdata,
        display_data=svdisplay_data,
        instance_names=shap_values.instance_names,
        feature_names=svfeature_names,
        output_names=shap_values.output_names,
        output_indexes=shap_values.output_indexes,
        lower_bounds=shap_values.lower_bounds,
        upper_bounds=shap_values.upper_bounds,
        main_effects=shap_values.main_effects,
        hierarchical_values=shap_values.hierarchical_values,
        clustering=shap_values.clustering,
    )
    if return_original:
        return sv, sv_name
    else:
        return sv


def get_raw_shap(
    model,
    model_name,
    background_vals,
    explanation_vals,
    seed=SEED,
):
    if model_name in ["lgbm", "xgb"]:
        explainer = shap.TreeExplainer(
            model=model,
            data=background_vals,
            feature_perturbation="interventional",
            model_output="probability",
            feature_names=background_vals.columns.tolist(),
        )
        shap_raw = explainer(explanation_vals)
    elif model_name in ["lr"]:
        explainer = shap.LinearExplainer(
            model,
            background_vals,
            feature_names=background_vals.columns.tolist(),
            seed=seed,
        )
        shap_raw = explainer(explanation_vals)
    elif model_name in ["nn", "stack"]:
        ## Suppress feat names warning from SVC and Stack
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        explainer = shap.KernelExplainer(
            model=model.predict_proba,
            data=background_vals,
            feature_names=background_vals.columns.tolist(),
        )
        # batch explain for intermediate logging of progress
        explanations = []
        n = len(explanation_vals)
        batch_size = 320
        for i, start in enumerate(range(0, n, batch_size), start=1):
            end = min(start + batch_size, n)
            batch = explanation_vals.iloc[start:end]
            shap_batch = explainer(
                batch,
                silent=False,  # True for progress bar
                l1_reg=0,  # will sometimes (stack) produce all-zero SHAP estimates w/o this
            )
            explanations.append(shap_batch)
            log(f"KernelExplainer progress: {end}/{n} rows ({end/n:.1%})")
        # Concat explanations
        shap_raw = explanations[0]
        for e in explanations[1:]:
            shap_raw = shap.Explanation(
                values=np.vstack([shap_raw.values, e.values]),
                data=np.vstack([shap_raw.data, e.data]),
                base_values=shap_raw.base_values,
                feature_names=shap_raw.feature_names,
                display_data=(
                    np.vstack([shap_raw.display_data, e.display_data])
                    if shap_raw.display_data is not None
                    else None
                ),
                output_names=shap_raw.output_names,
                output_indexes=None,
                instance_names=None,
            )
    else:
        raise ValueError(
            f"Expected model_name to be one of ['lgbm', 'xgb', 'nn', 'lr', 'svc', 'stack']; got {model_name} instead!"
        )
    return shap_raw


def generate_MAV(shap_vals, feat_order, result_path):
    """
    Generate and export mean absolute SHAP value (MASV) importance table.

    Computes both absolute and relative mean absolute SHAP values for each feature,
    providing a quantitative measure of feature importance. The relative MASV shows
    each feature's contribution as a percentage of total model explanation.

    Parameters
    ----------
    shap_vals : shap.Explanation
        SHAP Explanation object containing values for all samples and features.
        May be 2D or 3D array (will be reformatted internally).
    feat_order : list of str
        Desired ordering of features in the output table. Must contain exactly
        the same features as shap_vals.feature_names (order can differ).
    model_name : str
        Name of the model being analyzed. Used as the Excel sheet name when exporting.
    result_path : pathlib.Path
        Path to Excel file where MASV table will be written.

    Raises
    ------
    AssertionError
        If feat_order and shap_vals.feature_names contain different features.
        Prints the differences before raising.
    AssertionError
        If relative MASV percentages don't sum to approximately 100% (tolerance: 0.1%).

    Notes
    -----
    - **MASV**: Mean Absolute SHAP Value - average of |SHAP| across all samples
    - **Relative MASV**: Percentage contribution relative to sum of all MASVs
    - Relative MASV percentages always sum to 100% (within numerical tolerance)
    - Features are reordered according to feat_order in the output
    - Excel export uses append mode if file exists, enabling multi-model comparison
    - If all SHAP values are zero (rare edge case), function exits with warning

    Output Table Columns
    --------------------
    - **Feature**: Feature name
    - **MASV**: Mean absolute SHAP value (raw importance score)
    - **Relative_ MASV**: Percentage of total explanation (0-100%)
    """

    ## HELPER FUNCTION
    def get_vals_to_plot(shap_vals):
        """
        Reformat SHAP values for plotting by handling different model output structures.

        Converts SHAP Explanation objects with varying dimensionality (2D vs 3D arrays)
        into a standardized 2D format suitable for SHAP visualization functions. Handles
        binary classification models that output either single or dual class predictions,
        as well as multi-class models.

        Parameters
        ----------
        shap_vals : shap.Explanation
            SHAP Explanation object from shap.Explainer(). May contain:
            - 2D array [samples, features]: Standard format for many models
            - 3D array [samples, features, classes]: Multi-output or multi-class format

        Returns
        -------
        shap.Explanation
            Reformatted SHAP Explanation with 2D values array [samples, features],
            ready for plotting with shap.plots functions.
        """
        if len(shap_vals.values.shape) == 3:  # 3D array
            if (
                shap_vals.values.shape[2] == 1
            ):  # Binary classification with single output
                shap_vals_to_plot = shap_vals[:, :, 0]
            elif (
                shap_vals.values.shape[2] >= 2
            ):  # Binary with two outputs or multi-class
                shap_vals_to_plot = shap_vals[:, :, 1]  # Use positive class
            else:
                shap_vals_to_plot = shap_vals.mean(axis=2)  # Fallback
        else:  # 2D array
            shap_vals_to_plot = shap_vals
        return shap_vals_to_plot

    feat_names = shap_vals.feature_names
    try:
        assert set(feat_names) == set(feat_order)
    except AssertionError:
        log(set(feat_names) - set(feat_order))
        log(set(feat_order) - set(feat_names))
        raise AssertionError("Feature names and feature order do not match")
    shap_to_plot = get_vals_to_plot(shap_vals)
    shap_df = pd.DataFrame(shap_to_plot.values, columns=feat_names)
    # Get absolute avg + relative abs avg
    absolute_mean_shap = shap_df.abs().mean().reset_index()
    absolute_mean_shap.columns = ["Feature", "MASV"]
    sum_vals = absolute_mean_shap["MASV"].sum()
    if sum_vals == 0:
        raise Exception("All generated SHAP values are 0. Exiting...")
    absolute_mean_shap["Relative_ MASV"] = np.round(
        (100 * absolute_mean_shap["MASV"] / absolute_mean_shap["MASV"].sum()), 2
    )
    # Ensure logic makes sense
    assert np.isclose(
        absolute_mean_shap["Relative_ MASV"].sum(), 100, atol=0.1
    ), f"Sum is instead {absolute_mean_shap['Relative_ MASV'].sum()}"
    # Reorder
    absolute_mean_shap["Feature"] = absolute_mean_shap["Feature"].astype(str)

    absolute_mean_shap_reordered = (
        absolute_mean_shap.set_index("Feature").loc[feat_order].reset_index()
    )
    ######################## Export ########################
    if result_path.exists():
        result_path.unlink()
    result_path.parent.mkdir(exist_ok=True, parents=True)
    absolute_mean_shap_reordered.to_excel(result_path)


def get_stratified_background(X, y, n_background, random_state=SEED):
    """
    Returns a stratified subsample of X with the same event rate as the full dataset.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data to subsample from
    y : pd.Series or np.ndarray
        Labels used for stratification
    n_background : int
        Number of background samples to return
    random_state : int
        Random seed for reproducibility
    """
    frac = n_background / len(X)
    _, X_background, _, _ = train_test_split(
        X,
        y,
        test_size=frac,
        stratify=y,
        random_state=random_state,
    )
    return X_background


def get_shap_single_model(
    *_,
    model_name,
    feat_order,
    model_path,
    explanation_vals_path,
    background_vals_path,
    seed=SEED,
    n_background=None,
):
    """
    Generate SHAP values for a single model-outcome combination.

    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    # ==============> Load data + models
    explanation_vals = import_data(explanation_vals_path)
    background_vals = import_data(background_vals_path)
    in_dim = explanation_vals.shape[1]
    model = import_data(model_path, in_dim=in_dim)
    log("\t Data + model loaded!")
    # ==============> RAW SHAP
    if model_name in ["nn", "stack"]:
        # Subset
        log(f"\t Subsampling to {n_background} features...")
        background_y = import_data(background_vals_path.with_name("y_train.xlsx"))
        background_vals = get_stratified_background(
            background_vals, background_y, n_background, random_state=seed
        )
        assert len(background_vals) == n_background
    shap_raw = get_raw_shap(model, model_name, background_vals, explanation_vals, seed)
    log("\t SHAP VALS computed!")
    # ===========> Fix OHE
    ohe_dict = get_ohe_cols(explanation_vals)
    ohe_cols = ohe_dict.keys()
    raw_feat_order = []
    for col in feat_order:
        if col in ohe_cols:
            for sub_col in ohe_dict[col]:
                raw_feat_order.append(f"{col}_{sub_col}")
        else:
            raw_feat_order.append(col)
    # shap_old = copy.deepcopy(shap_raw)
    shap_combined = copy.deepcopy(shap_raw)
    for col_name in ohe_cols:
        mask = [n.startswith(f"{col_name}_") for n in shap_combined.feature_names]
        if not any(mask):
            continue
        shap_combined, _ = combine_encoded(
            shap_combined,
            col_name,
            mask,
        )
    log("\t OHE combined!")
    return shap_combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--outcome_name", required=True)
    parser.add_argument("--feat_order", type=str, required=True)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--result_path", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--n_background", required=False, type=int, default=None)
    args = parser.parse_args()
    ## Deal with paths
    vals_to_explain_path = args.data_dir / args.outcome_name.upper() / "X_test.parquet"
    background_vals_path = args.data_dir / args.outcome_name.upper() / "X_train.parquet"
    # Clean up feat order input
    feat_order = [f.strip() for f in args.feat_order.split(",")]
    # Run SHAP
    log(f"Starting for outcome: {args.outcome_name}; model: {args.model_name}")
    shap_combined = get_shap_single_model(
        model_name=args.model_name,
        feat_order=feat_order,
        model_path=args.model_path,
        explanation_vals_path=vals_to_explain_path,
        background_vals_path=background_vals_path,
        n_background=args.n_background,
        seed=args.seed,
    )
    # Generate MAV + export
    generate_MAV(
        shap_combined,
        feat_order,
        result_path=args.result_path,
    )
    log("\t MAV computed, DONE!")


if __name__ == "__main__":
    main()

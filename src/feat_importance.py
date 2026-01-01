import warnings
import copy
import logging
from filelock import FileLock
from src.config import SEED

warnings.filterwarnings("ignore", category=UserWarning)
import shap
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import Bunch
from datetime import datetime


def get_ohe_cols(df):
    """
    Extract original column names and categories from one-hot encoded features.

    Parses column names in a dataframe to identify and group one-hot encoded features
    by their original categorical variable names. Assumes one-hot encoded columns follow
    the naming convention: 'ORIGINAL_NAME_CATEGORY_VALUE'.

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
            'RACE_NEW': ['WHITE', 'BLACK', 'ASIAN'],
        }

    Notes
    -----
    - Columns without underscores are assumed to be non-encoded and are skipped
    - Special cases are excluded from grouping:
      * Columns starting with 'ETHNICITY_'
      * Columns starting with 'PARTIAL GLOSSECTOMY (HEMIGLOSSECTOMY_'
      * Columns starting with 'COMPOSITE_'
      * Columns starting with 'LOCAL_'
    - For columns starting with 'RACE_NEW_', the original name is 'RACE_NEW'
    - For all other encoded columns, the original name is the prefix before the first underscore
    """
    ohe_dict = {}
    for col in df.columns:
        col_split = col.split("_")
        if len(col_split) == 1 or col_split[0] == "ETHNICITY":
            continue
        else:
            col_name = col_split[0]
            instance_name = col_split[1]
        if col_name in ohe_dict.keys():
            ohe_dict[col_name].append(instance_name)
        else:
            ohe_dict[col_name] = [instance_name]
    return ohe_dict


######## Combine one-hot-encoded #######
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
        if shap_vals.values.shape[2] == 1:  # Binary classification with single output
            # DNN
            shap_vals_to_plot = shap_vals[:, :, 0]
        elif shap_vals.values.shape[2] >= 2:  # Binary with two outputs or multi-class
            shap_vals_to_plot = shap_vals[:, :, 1]  # Use positive class
        else:
            shap_vals_to_plot = shap_vals.mean(axis=2)  # Fallback
    else:  # 2D array
        # LightGBM, SVC, KNN, Stack, LR-Nomogram
        shap_vals_to_plot = shap_vals
    return shap_vals_to_plot


def generate_MAV(shap_vals, feat_order, model_name, result_path=None):
    """
    Generate and optionally export mean absolute SHAP value (MASV) importance table.

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
    result_path : pathlib.Path, optional
        Path to Excel file where MASV table will be written. If the file exists,
        the new sheet is appended. If None, results are not exported.

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
    feat_names = shap_vals.feature_names
    try:
        assert set(feat_names) == set(feat_order)
    except AssertionError:
        print(set(feat_names) - set(feat_order))
        print(set(feat_order) - set(feat_names))
        raise AssertionError("Feature names and feature order do not match")
    shap_to_plot = get_vals_to_plot(shap_vals)
    shap_df = pd.DataFrame(shap_to_plot.values, columns=feat_names)
    absolute_mean_shap = shap_df.abs().mean().reset_index()
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
    ######################## Display + Export ########################
    if result_path:
        result_path.parent.mkdir(exist_ok=True, parents=True)
        with pd.ExcelWriter(
            result_path,
            engine="openpyxl",
            mode="a" if result_path.exists() else "w",
        ) as writer:
            absolute_mean_shap_reordered.to_excel(
                writer, sheet_name=model_name, index=True
            )


def get_shap_single_model(
    *_,
    model,
    model_name,
    outcome_name,
    feat_order,
    explanation_path,
    background_path,
    log_path,
    result_path=None,
):
    """
    Generate SHAP values for a single model-outcome combination.

    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    ######################## Initialize logger ########################
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logging.getLogger("shap").setLevel(logging.WARNING)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        filemode="w",
        force=True,
    )
    logging.info(f"Starting SHAP for outcome: {outcome_name}, model: {model_name}")
    ######################## Import expl/background (subset test/train) sets ########################
    background_vals = pd.read_parquet(background_path)
    explanation_vals = pd.read_parquet(explanation_path)
    ######################## GET RAW SHAP VALUES ########################
    if model_name in ["xgb", "lgbm"]:
        explainer = shap.TreeExplainer(
            model=model,
            data=background_vals,
            feature_perturbation="interventional",
            model_output="probability",
            feature_names=background_vals.columns.tolist(),
        )
        shap_raw = explainer(explanation_vals)
    elif model_name == "lr":
        explainer = shap.LinearExplainer(
            model,
            background_vals,
            feature_names=background_vals.columns.tolist(),
            seed=SEED,
        )
        shap_raw = explainer(explanation_vals)
    elif model_name in ["nn", "stack"]:
        explainer = shap.KernelExplainer(
            model=model.predict_proba,
            data=background_vals,
            feature_names=background_vals.columns.tolist(),
        )
        # shap_raw = explainer(explanation_vals)
        explanations = []
        n = len(explanation_vals)
        batch_size = 64
        for i, start in enumerate(range(0, n, batch_size), start=1):
            end = min(start + batch_size, n)
            batch = explanation_vals.iloc[start:end]
            shap_batch = explainer(batch)
            explanations.append(shap_batch)
            logging.info(f"KernelExplainer progress: {end}/{n} rows ({end/n:.1%})")
        # shap_raw = shap.Explanation.concatenate(explanations)
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
            f"Expected model_name to be one of ['xgb', 'lgbm', 'nn', 'lr', 'stack']; got {model_name} instead!"
        )

    logging.info("SHAP values calculated")
    ######################## Deal with one-hot encoded ########################
    ohe_dict = get_ohe_cols(explanation_vals)
    ohe_cols = ohe_dict.keys()

    raw_feat_order = []
    for col in feat_order:
        if col in ohe_cols:
            for sub_col in ohe_dict[col]:
                raw_feat_order.append(f"{col}_{sub_col}")
        else:
            raw_feat_order.append(col)

    shap_old = copy.deepcopy(shap_raw)
    for col_name in ohe_cols:
        shap_combined, _ = combine_encoded(
            shap_old, col_name, [col_name in n for n in shap_old.feature_names]  # type: ignore
        )
        shap_old = shap_combined

    logging.info("OHE features combined")
    ######################## Generate + export MAV table ########################
    if result_path:
        # raw_path = result_path / "raw" / f"{outcome_name}.xlsx"
        combined_path = result_path / "combined" / f"{outcome_name}.xlsx"
    else:
        # raw_path = None
        combined_path = None

    # generate_MAV(shap_raw, raw_feat_order, model_name=model_name, result_path=raw_path)
    generate_MAV(
        shap_combined, feat_order, model_name=model_name, result_path=combined_path
    )
    logging.info("Computation complete!")


##########################################################################################
############################## PERMUTAION ##############################


def batch_perm_imp(
    estimator,
    X,
    y,
    log_path,
    n_repeats,
    scoring,
    random_state,
    n_jobs,
    batch_size,
):
    """
    Compute permutation importance in batches, logging progress.

    Returns a Bunch with the same attributes as sklearn.permutation_importance:
    - importances: (n_features, n_repeats)
    - importances_mean: (n_features,)
    - importances_std: (n_features,)
    """
    # Prepare log file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Clear old log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path.write_text(
        f"[{timestamp}] \nStarting permutation importance...\nTest size: {len(y)} \nTotal iterations: {n_repeats} \n"
    )

    def log(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")

    rng = np.random.RandomState(random_state)

    n_done = 0
    importances_list = []
    n_features = None
    while n_done < n_repeats:
        cur_repeats = min(batch_size, n_repeats - n_done)

        # Generate a fresh seed per batch so permutations differ
        batch_seed = rng.randint(0, 2**31 - 1)
        res = permutation_importance(
            estimator=estimator,
            X=X,
            y=y,
            n_repeats=cur_repeats,
            random_state=batch_seed,
            n_jobs=n_jobs,
            scoring=scoring,
        )
        if n_features is None:
            n_features = res.importances.shape[0]

        importances_list.append(res.importances)
        n_done += cur_repeats

        log(f"Progress: {n_done}/{n_repeats} iterations completed.")
    # Concatenate over repeats axis
    importances = np.concatenate(importances_list, axis=1)  # (n_features, n_repeats)

    # Recompute mean/std across all repeats
    importances_mean = importances.mean(axis=1)
    importances_std = importances.std(axis=1, ddof=1)

    return Bunch(
        importances=importances,
        importances_mean=importances_mean,
        importances_std=importances_std,
    )


def plot_perm_single_model(
    model_name,
    model,
    outcome_name,
    X,
    y,
    log_path,
    n_repeats=100,
    result_path=None,
    show_output=False,
    scoring="roc_auc",
    rand_state=SEED,
    batch_size=5,
    n_perm_jobs=1,
):
    """
    Calculate and plot permutation feature importance using bar charts.

    Measures decrease in model scoring (auroc by default) when each feature is randomly shuffled
    """
    result = batch_perm_imp(
        estimator=model,
        X=X,
        y=y,
        log_path=log_path,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=rand_state,
        n_jobs=n_perm_jobs,
        batch_size=batch_size,
    )
    ## Sort results (most important at the top)
    sorted_feats = pd.Series(result.importances_mean, index=X.columns).sort_values(
        ascending=True
    )

    ## Create plot
    fig, ax = plt.subplots(figsize=(10, 12))
    sorted_feats.plot.barh(xerr=result.importances_std, ax=ax)
    ax.set_title(
        f"{model_name}-{outcome_name} {scoring.upper()} Permutation Importance over {n_repeats} iterations"
    )
    ax.set_xlabel(f"Decrease in mean {scoring}")
    fig.tight_layout()
    if result_path:
        result_path.mkdir(exist_ok=True, parents=True)
        ## FIGURE
        fig_path = result_path / f"{outcome_name}.pdf"
        if fig_path.exists():
            fig_path.unlink()
        plt.savefig(fig_path, bbox_inches="tight")
        if show_output:
            plt.show()
        plt.close()
        ## TABLE
        table_path = result_path / f"{outcome_name}.xlsx"
        if table_path.exists():
            table_path.unlink()
        # Build results table
        df_results = pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )

        # Sort so matches bar plot
        df_results = df_results.sort_values("importance_mean", ascending=True)

        # Save to Excel
        df_results.to_excel(table_path, index=False)


##########################################################################################

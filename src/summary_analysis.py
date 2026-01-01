from src.data_utils import get_feature_lists
import sigfig
import warnings
import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from scipy.stats.contingency import odds_ratio
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from tqdm.notebook import tqdm


def add_dummy_rows(*_, df, col, result_list, header_ORs, header_p):
    """
    Creates dummy rows to be added to summary/analysis df in the event that a given feature should not be populated with values

    Parameters
    ---------
    df: pandas DataFrame
        Tabular dataframe containing both predictor and outcome variables
    col: string
        Name of the column for which dummy rows will be generated
    result_list: list[dict{}]
        List of dictionaries. Each dict pertains to a single row once converted into a dataframe
    header_ORs: string
        Header for Odds Ratios column in final summary/analysis dataframe
    header_p: string
        Header for p-value column in final summary/analysis dataframe
    """
    entries = sorted(df[col].unique())
    for entry in entries:
        result_list.append(
            {
                "Feature": f"{col.upper()} {entry}",
                header_ORs: "",
                header_p: "",
            }
        )
    return result_list


def format_p_val(p_val):
    """
    Re-formats p-values if <0.0001, and rounds otherwise. Converts to string
    """
    if p_val < 0.0001:
        return "<0.0001"
    elif p_val <= 0.05:
        return str(sigfig.round(p_val, sigfigs=2))
    else:
        return str(round(p_val, 1))


def generate_fish_list(df, outcome_dict, binary_cols, verbose=True):
    """
    Generates dictionary specifying, for each outcome, which features require fishers exact test due to low expected frequencies

    Returns
    -------
    Dictionary with format:
    {
        <outcome_name> str: List[<feat_name> str]
    }
    """
    fish_dict = {}
    for outcome_name, outcome in outcome_dict.items():
        if verbose:
            print(f"{outcome_name}...")
        fish_dict[outcome_name] = []
        for col in df:
            if col in binary_cols:
                contingency_table = pd.crosstab(df[col], outcome)
                _, _, _, expected_frequencies = chi2_contingency(contingency_table)
                if (expected_frequencies < 5).any():  # type: ignore
                    fish_dict[outcome_name].append(col)
                    if verbose:
                        print(col)
                        print(expected_frequencies)
                        print("-" * 20)
    return fish_dict


def generate_summary_column(
    df_impute,
    og_df,
    outcome,
    outcome_type,
    all_categories,
    outcome_sub_cols,
    feature_dict,
):
    """
    Generates a single column for the summary table. Usually one of [all, only_positive_instances, only_neg_instances]

    Parameters
    ----------
    df_impute: pd.DataFrame
        Dataframe for which summary values are to be generated. Usually a fully processed (imputed) dataset (excludes target variable).
    og_df: pd.DataFrame
        Original dataframe (before imputation), used only to extract missingness from numerical values
    outcome: pd.Series
    outcome_type: str
    all_categories:
    outcome_sub_cols: dict
        Dictionary detailing raw columns used to create general columns
        Format:
        {
            <gen_col_name> str: list[<col_name> str],
        }
    feature_dict: dict
        Dictionary detailing the type of feature each column is (excluding target)
        Format:
        {
            <col_type> str: list[<col_name> str],
        }
    """
    binary_cols = feature_dict["binary_cols"]
    numerical_cols = feature_dict["numerical_cols"]
    nominal_cols = feature_dict["nominal_cols"]
    ordinal_cols = feature_dict["ordinal_cols"]

    total_entries = len(df_impute)
    header = f"{outcome}-{outcome_type}  (n={total_entries})"
    summary_list = []

    for col in tqdm(
        df_impute.columns, desc=f"{outcome_type} summary column", leave=False
    ):  # Loop through all columns
        summary_list.append({"Feature": f"{col.upper()}", header: ""})
        if col in binary_cols + nominal_cols + ordinal_cols:  # Categorical
            # Get counts and percentages, add blank row for variable name
            counts = df_impute[col].value_counts()
            percentages = np.round(
                df_impute[col].value_counts(normalize=True) * 100, decimals=1
            )
            # if col == "OTHSESHOCK":
            #     print(counts)
            #     print(percentages)
            #     for val in summary_list[-3:]:
            #         print(val)
            #     print('-'*20)
            # For each instance in feature, add counts/percentages to df
            for entry in all_categories[col]:
                summary_list.append(
                    {
                        "Feature": f"{col.upper()} {entry}",
                        # Get value count, if not existent, replace with 0
                        header: f"{counts.get(entry, 0)} ({percentages.get(entry, 0.0)})",
                    }
                )
                # if col == "OTHSESHOCK":
                #     print(counts.get(entry, 0))
                #     print(percentages.get(entry, 0.0))
            # if col == "OTHSESHOCK":
            #     for val in summary_list[-3:]:
            #         print(val)
            #     print('*'*50)
        elif col in numerical_cols:  # Numerical
            # Get mean, stdev, quantiles
            avg = np.mean(df_impute[col])
            std = np.std(df_impute[col])
            quantiles = np.round(
                df_impute[col].quantile([0.25, 0.5, 0.75]).values.tolist(), 1
            )
            summary_list.append(
                {
                    "Feature": col.upper() + ", Avg ± (SD) -- [25%, 50%, 75%]",
                    header: f"{avg:.1f} ± {std:.1f} -- {quantiles}",
                }
            )
            ## Add missingness
            n_missing = og_df[col].isna().sum()
            pct_missing = np.round(n_missing / total_entries * 100, 1)
            summary_list.append(
                {
                    "Feature": f"{col.upper()} Missing (% missing)",
                    header: f"{n_missing} ({pct_missing})",
                }
            )
        else:
            print(col)
            raise ValueError
    return pd.DataFrame(summary_list)


def generate_summary_table(
    *_,
    X_df_final,
    X_df_og,
    outcome_data,
    outcome_name,
    all_categories,
    outcome_sub_cols,
):
    """
    Parent/main function used to generate summary tables. Generates one column at a time (all, positives, negatives) and appends together.

    Parameters
    X_df_final: pandas dataframe
        Final tabular dataframe containing data (usually imputed) to be summarized
        Excludes target variable
    X_df_og: pandas dataframe
        Original tabular dataframe without any imputations used to calculate missingness
        Excludes target variable
    outcome_data: pandas Series
        Pandas series containing binary target variable
    outcome_name: str
        Specifies name of target variable
    all_categories: dict
        Dictionary mapping feature names to all instances of that feature
        ex)
            {
                'SEX' : ['male', 'female']
            }
    outcome_sub_cols: dict
        Dictionary mapping summay/analysis-specific general features to the sub-features in the df
        that were aggregated to create that outocme
        ex)
        any positive in "SUPINFEC","WNDINFD", "ORGSPCSSI","DEHIS" is a positive for Surgical_Outcome
            {
               "Surgical_Outcome": ["SUPINFEC",
                                    "WNDINFD",
                                    "ORGSPCSSI",
                                    "DEHIS"]
            }
    Returns
    -------
    combined_df: pandas dataframe
        Columns:
            All/Total entries, Negative entries, Positive Entries
        Rows:
            Feature Names (+ feature instances if categorical)
        Values:
            Numerical Features: Mean +/- SD; [25%, 50%, 75%] quartiles
    Raises
    ------
    ValueError:
        If positional arguments are given
    """
    ## Ensure outcome is binary and numeric
    if outcome_data.dtype != "int64":
        raise TypeError(f"Outcome data must be binary int64, got {outcome_data.dtype}")

    feature_dict = get_feature_lists(X_df_final)
    feature_dict["numerical_cols"].remove("OPERYR")
    feature_dict["ordinal_cols"].append("OPERYR")

    if outcome_name in X_df_final.columns:
        # If outcome already in df
        concat_df = X_df_final.copy()
        og_concat = X_df_og.copy()
    else:
        # If outcome not already in df, concat
        concat_df = pd.concat([X_df_final, outcome_data], axis=1)
        og_concat = pd.concat([X_df_og, outcome_data], axis=1)

    non_outcome_df = (
        concat_df[concat_df[outcome_name] == 0].drop(outcome_name, axis=1).copy()
    )
    non_outcome_df_og = (
        og_concat[og_concat[outcome_name] == 0].drop(outcome_name, axis=1).copy()
    )
    outcome_df = (
        concat_df[concat_df[outcome_name] == 1].drop(outcome_name, axis=1).copy()
    )
    outcome_df_og = (
        og_concat[og_concat[outcome_name] == 1].drop(outcome_name, axis=1).copy()
    )
    # Get total, neg-outcome, and pos-outcome summary tables
    total_col = generate_summary_column(
        X_df_final,
        X_df_og,
        outcome_name,
        "Total",
        all_categories,
        outcome_sub_cols,
        feature_dict,
    )
    non_outcome_col = generate_summary_column(
        non_outcome_df,
        non_outcome_df_og,
        outcome_name,
        "Negative",
        all_categories,
        outcome_sub_cols,
        feature_dict,
    )
    outcome_col = generate_summary_column(
        outcome_df,
        outcome_df_og,
        outcome_name,
        "Positive",
        all_categories,
        outcome_sub_cols,
        feature_dict,
    )
    ## Combine by feature column
    total_col = total_col.set_index("Feature")

    non_outcome_col = non_outcome_col.set_index("Feature")

    outcome_col = outcome_col.set_index("Feature")

    combined_df = pd.concat([total_col, non_outcome_col, outcome_col], axis=1)

    return combined_df


def get_analysis_df(*_, df, outcome_data, outcome_name, outcome_sub_cols, fish_dict):
    """
    Returns a dataframe containing p-values, odds ratios, and 95% CIs for a given outcome

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe for which analysis values are being run on. Usually just the base tabular data (excluding any target variables)
    outcome_data: pd.Series
        Pandas Series containing labels for a given outcome
    outcome_name: str
        String specifying the name of the given outcome
    outcome_sub_cols: dict
        Dictionary detailing raw columns used to create general columns
        Format:
        {
            <gen_col_name> str: list[<col_name> str],
        }
    fish_dict: dict
        Dictionary containing, for each outcome, what features require fishers exact test. The return value of a call to generate_fish_list().
    Raises:
    ------
    ValueError:
        If positional arguments are given
    """
    if _ != tuple():
        raise ValueError("This function does not accept positional arguments!")
    ## Set func globals
    feature_dict = get_feature_lists(df)
    binary_cols = feature_dict["binary_cols"]
    numerical_cols = feature_dict["numerical_cols"]
    nominal_cols = feature_dict["nominal_cols"]
    ordinal_cols = feature_dict["ordinal_cols"]
    header_ORs = "Odds Ratios (95% CI)"
    header_p = "P-Value"
    ## Make Year an ordinal variable instead of numerical --> just for this portion of proj
    numerical_cols.remove("OPERYR")
    ordinal_cols.append("OPERYR")
    ##Analysis (p-vals + ORs)
    result_list = []
    if outcome_name in df.columns:
        full_df = df.copy()
    else:
        full_df = pd.concat([df, outcome_data], axis=1)
    for col in tqdm(
        df.columns, desc="Analysis", leave=False
    ):  # Loop through all columns
        ## Feature is either the outcome or directly used to generate outcome
        if col in outcome_sub_cols[outcome_name] or outcome_name == col:
            result_list.append(
                {"Feature": f"{col.upper()}", header_ORs: "---", header_p: "---"}
            )
            result_list = add_dummy_rows(
                df=full_df,
                col=col,
                result_list=result_list,
                header_ORs=header_ORs,
                header_p=header_p,
            )
            continue
        elif col in binary_cols:
            contingency_table = pd.crosstab(df[col], outcome_data)
            if (contingency_table == 0).any().any():
                contingency_table += 1
            if col in fish_dict[outcome_name]:
                ## Fishers Exact for p-vals if expected freq < 5
                _, p_value = fisher_exact(contingency_table)
                result = odds_ratio(contingency_table, kind="conditional")
            else:
                _, p_value, _, _ = chi2_contingency(contingency_table)
                result = odds_ratio(contingency_table, kind="sample")
            p_value = format_p_val(p_value)
            or_estimate = result.statistic
            ci_low, ci_high = result.confidence_interval(confidence_level=0.95)
            odds_conf = f"{or_estimate:.4f} ({ci_low:.4f}, {ci_high:.4f})"
            result_list.append(
                {"Feature": col.upper(), header_ORs: odds_conf, header_p: p_value}
            )
            ##Append empty rows just to ensure index matches with summary
            result_list = add_dummy_rows(
                df=full_df,
                col=col,
                result_list=result_list,
                header_ORs=header_ORs,
                header_p=header_p,
            )
        elif col in numerical_cols:
            ### Mann-Whitney U test for p-vals ###
            group1 = full_df[full_df[outcome_name] == 0][col]
            group2 = full_df[full_df[outcome_name] == 1][col]
            _, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
            p_value = format_p_val(p_value)
            ### Log Regresion for ORs and CIs ###
            X = sm.add_constant(full_df[col])
            y = full_df[outcome_name].values
            model = sm.Logit(y, X).fit(disp=0)
            or_estimate = np.exp(model.params[col])
            conf_int = model.conf_int().loc[col]
            ci_lower = np.exp(conf_int[0])
            ci_upper = np.exp(conf_int[1])
            logit_p = model.pvalues[col]
            logit_p = format_p_val(logit_p)
            odds_conf = f"{or_estimate:.4f} ({ci_lower:.4f}, {ci_upper:.4f})"
            list_adds = [
                {
                    "Feature": col.upper(),
                    header_ORs: odds_conf,
                    header_p: p_value,
                },
                # Extra row to match summary df
                {
                    "Feature": col.upper() + ", Avg ± (SD) -- [25%, 50%, 75%]",
                    header_ORs: "",
                    header_p: "",
                },
                # Extra row to match summary df
                {
                    "Feature": col.upper() + " Missing (% missing)",
                    header_ORs: "",
                    header_p: "",
                },
            ]
            result_list.extend(list_adds)
        elif col in nominal_cols + ordinal_cols:
            result_list.append({"Feature": col.upper(), header_ORs: "", header_p: ""})
            y = full_df[outcome_name].values
            ### One-hot encode, excluding entry with highest frequency as a reference-> run log regression for p-val and ORs (CI)
            # List of possible entries in a given column, sorted from low to HIGH
            entries = sorted(full_df[col].unique())
            # Create subset onehot-encoded temporary df
            temp_df = full_df[[col]]
            temp_df = pd.get_dummies(
                temp_df, columns=[col], drop_first=False, dtype=int
            )
            # If nominal, drop highest freq entry and make that reference
            if col in nominal_cols:
                val_counts = full_df[col].value_counts()
                max_freq_idx = val_counts.idxmax()  # Entry with the highest frequency
                drop_col = f"{col}_{max_freq_idx}"
            # If ordinal, drop the entry with the lowest value and make that reference
            else:
                # Ensure no perfect sep in reference instance
                ct = pd.crosstab(df[col], y)
                zero_entries = ct[ct[1] == 0].index.tolist()
                for i in range(len(entries)):
                    drop_col = f"{col}_{entries[i]}"
                    if entries[i] not in zero_entries:
                        break
            temp_df.drop(drop_col, axis=1, inplace=True)
            X = sm.add_constant(temp_df)
            ## Identify problematic entries
            ct = pd.crosstab(df[col], y)
            zero_entries = ct[ct[1] == 0].index.tolist()
            bad_entry_list = [
                f"{col}_{entry}"
                for entry in zero_entries
                if f"{col}_{entry}" != drop_col and f"{col}_{entry}" in X.columns  # type: ignore
            ]
            if bad_entry_list:
                X = X.drop(bad_entry_list, axis=1)  # type: ignore
            model = None
            or_estimates = None
            conf_ints = None
            p_values = None
            try:  # Try to run Logit
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    model = sm.Logit(y, X).fit(disp=0)
                    or_estimates = np.exp(model.params)
                    conf_ints = model.conf_int()
                    p_values = model.pvalues
            except (
                ConvergenceWarning,
                RuntimeWarning,
                ValueError,
                KeyError,
                OverflowError,
            ) as e:  # Except any issues with logit
                # Log issue
                print(f"Failed to fit model for {col}: {type(e).__name__}: {e}")
                # Mark all entries as unstable
                for entry in entries:
                    entry_name = f"{col}_{entry}"
                    result_list.append(
                        {
                            "Feature": f"{col.upper()} {entry}",
                            header_ORs: "UNSTABLE",
                            header_p: "UNSTABLE",
                        }
                    )
                continue  # Skip to next column
            if model is not None:
                for entry in entries:  # Loop through each possible entry of the feature
                    entry_name = f"{col}_{entry}"  # Reformat to allow for indexing ex) 1.0 --> SITE_1.0
                    if entry_name == drop_col:
                        result_list.append(
                            {
                                "Feature": f"{col.upper()} {entry}",
                                header_ORs: "Reference",
                                header_p: "Reference",
                            }
                        )

                    elif entry_name in bad_entry_list:
                        result_list.append(
                            {
                                "Feature": f"{col.upper()} {entry}",
                                header_ORs: "UNSTABLE",
                                header_p: "UNSTABLE",
                            }
                        )
                    # For BOTH: If not designated reference, get stat vals
                    else:
                        p_val_specific = format_p_val(p_values[entry_name])
                        or_estimate = or_estimates.loc[entry_name]
                        ci_lower = np.exp(conf_ints.loc[entry_name, 0])
                        ci_upper = np.exp(conf_ints.loc[entry_name, 1])
                        odds_conf = (
                            f"{or_estimate:.4f} ({ci_lower:.4f}, {ci_upper:.4f})"
                        )
                        result_list.append(
                            {
                                "Feature": f"{col.upper()} {entry}",
                                header_ORs: odds_conf,
                                header_p: p_val_specific,
                            }
                        )

    results_df = pd.DataFrame(result_list)
    return results_df.set_index("Feature")

import sigfig
import warnings
import numpy as np
import pandas as pd

from src.feat_eng import log

from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from scipy.stats.contingency import odds_ratio

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def impute_numerical(df, impute_cols, seed):
    """
    Impute numeric cols
    """
    ## Impute
    imputer = IterativeImputer(
        estimator=None,  # default = BayesianRidge
        initial_strategy="median",
        imputation_order="random",
        max_iter=10,
        sample_posterior=False,  # deterministic
        random_state=seed,
    )
    df_impute = df.copy()
    imputed_values = imputer.fit_transform(df[impute_cols])
    df_impute[impute_cols] = imputed_values
    assert df_impute.isna().sum().sum() == 0
    return df_impute


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


def get_binary_analysis(
    df, bin_col, outcome_name, fish_dict, bin_cat_dict, header_ORs, header_p
):
    """
    Calculates/formats ORs, 95% CIs, and p-values for binary variables in relation to outcome_name

    Params
    -----
    bin_cat_dict: dict{bin_col: {1:str(entry_name), 0: str(entry_name)}}
        For a given binary column, maps 1/0 encoding back to original entry name
    fish_dict: dict{outcome_name: list[str]}
        Maps outcomes to a list of binary cols that require fisher exact test bc of low (<5) expeced freq
    """
    contingency_table = pd.crosstab(df[bin_col], df[outcome_name])
    if (contingency_table == 0).any().any():
        contingency_table += 1
    ## P-VALUE ##
    if bin_col in fish_dict[outcome_name]:
        ## Fishers Exact for p-vals if expected freq < 5
        _, p_value = fisher_exact(contingency_table)
        or_kind = "conditional"
    else:
        _, p_value, _, _ = chi2_contingency(contingency_table)
        or_kind = "sample"
    p_value = format_p_val(p_value)
    ## ODDS RATIOs ##
    result = odds_ratio(contingency_table, kind=or_kind)
    or_estimate = result.statistic
    ci_low, ci_high = result.confidence_interval(confidence_level=0.95)
    or_ci = f"{or_estimate:.2f} ({ci_low:.2f}, {ci_high:.2f})"
    #### ADD TO LIST ####
    if bin_col in bin_cat_dict.keys():
        cat_dict = bin_cat_dict[bin_col]
        ## Add 1 header row, 1 row for "1" entry, and 1 row for "0" entry (ref)
        res_sub_list = [
            {"Feature": bin_col.upper(), header_ORs: "", header_p: ""},
            {
                "Feature": f"{bin_col.upper()}___{cat_dict[1]}",
                header_ORs: or_ci,
                header_p: p_value,
            },
            {
                "Feature": f"{bin_col.upper()}___{cat_dict[0]}",
                header_ORs: "Reference",
                header_p: "Reference",
            },
        ]
    else:
        res_sub_list = [
            {
                "Feature": f"{bin_col.upper()}___{bin_col.upper()}",
                header_ORs: or_ci,
                header_p: p_value,
            }
        ]
    return res_sub_list


def get_numeric_analysis(df, outcome_name, num_col, header_ORs, header_p):
    """
    Calculates/formats ORs, 95% CIs, and p-values for continuous numeric variables in relation to outcome_name
    """
    ### Mann-Whitney U test for p-vals ###
    group1 = df[df[outcome_name] == 0][num_col]  # neg patients
    group2 = df[df[outcome_name] == 1][num_col]  # pos patients
    _, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
    p_value = format_p_val(p_value)
    ### Log Regresion for ORs and CIs ###
    X = sm.add_constant(df[num_col])
    y = df[outcome_name].values
    model = sm.Logit(y, X).fit(disp=0)
    or_estimate = np.exp(model.params[num_col])
    conf_int = model.conf_int().loc[num_col]
    ci_lower = np.exp(conf_int[0])
    ci_upper = np.exp(conf_int[1])
    # logit_p = model.pvalues[col] # can get p from logit but will use Mann-Whitney U
    # logit_p = format_p_val(logit_p)
    odds_conf = f"{or_estimate:.2f} ({ci_lower:.2f}, {ci_upper:.2f})"
    ### ADD TO ROWS ####
    res_sub_list = [
        {
            "Feature": f"{num_col.upper()}, median (IQR)",
            header_ORs: odds_conf,
            header_p: p_value,
        },
        # add to match summary
        {
            "Feature": f"{num_col.upper()}___Unknown (imputed)",
            header_ORs: "---",
            header_p: "---",
        },
    ]
    return res_sub_list


def get_ref_bad_entries(df, col, nominal_cols, entries, y):
    """
    Given a nominal or ordinal column, extract reference column and any zero-seperated columns

    Reference is chosen as:
        - Ordinal: Lowest-rank value
        - Nominal: Entry with highest frequency
    """
    # Get perf seperated entries
    ct = pd.crosstab(df[col], y)
    zero_entries = ct[ct[1] == 0].index.tolist()

    # If nominal, drop highest freq entry and make that reference
    if col in nominal_cols:
        ref_col = df[col].value_counts().idxmax()  # Entry with the highest frequency
        assert ref_col not in zero_entries
    # If ordinal, drop the entry with the lowest value and make that reference
    else:
        # use first entry that isnt perf seperated as reference (drop)
        for entry in entries:
            if entry not in zero_entries:
                ref_col = entry
                break
    return zero_entries, ref_col


def mark_unstable(col, entries, header_ORs, header_p):
    """
    If a given feature is found to be unstable, replace statistical values with "UNSTABLE"
    """
    res_sub_list = []
    for entry in entries:
        res_sub_list.append(
            {
                "Feature": f"{col.upper()}___{entry}",
                header_ORs: "UNSTABLE",
                header_p: "UNSTABLE",
            }
        )
    return res_sub_list


def get_cat_analysis(
    col,
    df,
    outcome_name,
    nominal_cols,
    header_ORs,
    header_p,
):
    """
    1) One-hot encode, excluding:
        Nominal: entry with highest frequency as a reference
        Ordinal: lowest-value entry (assuming no perfect seperation)
    2) attempt to id problematic/unstable entries and remove/mark unstable
    3) run log regression for p-val and ORs (CI)
        - returns early if logit fails (w/ all entries marked unstable)
    4) Return updated result_list
    """
    result_list = []
    y = df[outcome_name].values
    # List of possible entries in a given column, sorted from low to HIGH
    entries = sorted(df[col].unique())
    bad_entry_list, ref_col = get_ref_bad_entries(
        df=df, col=col, nominal_cols=nominal_cols, entries=entries, y=y
    )
    # Create subset onehot-encoded temporary df
    temp_df = pd.get_dummies(df[col], columns=[col], drop_first=False, dtype=int)
    temp_df.drop(ref_col, axis=1, inplace=True)
    if len(bad_entry_list) > 0:
        temp_df.drop(bad_entry_list, axis=1, inplace=True)
    X = sm.add_constant(temp_df)
    model = None
    or_estimates = None
    conf_ints = None
    p_values = None
    # Try to run Logit
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            model = sm.Logit(y, X).fit(disp=0)
            or_estimates = np.exp(model.params)
            conf_ints = model.conf_int()
            p_values = model.pvalues
    # Except any issues with logit
    except (
        ConvergenceWarning,
        RuntimeWarning,
        ValueError,
        KeyError,
        OverflowError,
    ) as e:
        # Log the specific issue
        print(f"Failed to fit model for {col}: {type(e).__name__}: {e}")
        # Mark all entries as unstable and return
        unstable_list = mark_unstable(
            col=col, entries=entries, header_ORs=header_ORs, header_p=header_p
        )
        result_list += unstable_list
        return result_list  # Skip to next column

    # Loop through each possible entry of the feature
    for entry in entries:
        if entry == ref_col:
            result_list.append(
                {
                    "Feature": f"{col.upper()}___{entry}",
                    header_ORs: "Reference",
                    header_p: "Reference",
                }
            )

        elif entry in bad_entry_list:
            result_list.append(
                {
                    "Feature": f"{col.upper()}___{entry}",
                    header_ORs: "UNSTABLE",
                    header_p: "UNSTABLE",
                }
            )
        # If not designated reference, get stat vals
        else:
            p_val_specific = format_p_val(p_values[entry])
            or_estimate = or_estimates.loc[entry]
            ci_lower = np.exp(conf_ints.loc[entry, 0])
            ci_upper = np.exp(conf_ints.loc[entry, 1])
            odds_conf = f"{or_estimate:.2f} ({ci_lower:.2f}, {ci_upper:.2f})"
            result_list.append(
                {
                    "Feature": f"{col.upper()}___{entry}",
                    header_ORs: odds_conf,
                    header_p: p_val_specific,
                }
            )
    return result_list


def get_analysis_df(
    df,
    outcome_name,
    outcome_sub_cols,
    fish_dict,
    feature_dict,
    bin_cat_dict,
    all_categories,
):
    """
    For a given outcome, calculate ORs (95% CIs) and p-values for all features in the full df.

    Does not attempt to calculate values if a feature is the outcome or if it was used to create the outcome
    """
    result_list = []
    header_ORs = f"{outcome_name} OR (95% CI)"
    header_p = f"{outcome_name} p-val"
    for col in df.columns:
        if col in feature_dict["binary_cols"]:
            if col in outcome_sub_cols[outcome_name] or col == outcome_name:
                result_list.append(
                    {
                        "Feature": f"{col.upper()}___{col.upper()}",
                        header_ORs: "---",
                        header_p: "---",
                    }
                )
                continue
            bin_res_list = get_binary_analysis(
                df=df,
                bin_col=col,
                outcome_name=outcome_name,
                fish_dict=fish_dict,
                bin_cat_dict=bin_cat_dict,
                header_ORs=header_ORs,
                header_p=header_p,
            )
            result_list += bin_res_list
        elif col in feature_dict["numerical_cols"]:
            num_res_list = get_numeric_analysis(
                df=df,
                outcome_name=outcome_name,
                num_col=col,
                header_ORs=header_ORs,
                header_p=header_p,
            )
            result_list += num_res_list
        elif col in feature_dict["nominal_cols"] + feature_dict["ordinal_cols"]:
            ## Append to match
            result_list.append({"Feature": col.upper(), header_ORs: "", header_p: ""})
            if col in outcome_sub_cols[outcome_name] or col == outcome_name:
                ## Add fillers
                result_list += [
                    {
                        "Feature": f"{col.upper()}___{val}",
                        header_ORs: "---",
                        header_p: "---",
                    }
                    for val in all_categories[col]
                ]
                continue
            cat_results = get_cat_analysis(
                col=col,
                df=df,
                outcome_name=outcome_name,
                nominal_cols=feature_dict["nominal_cols"],
                header_ORs=header_ORs,
                header_p=header_p,
            )
            result_list += cat_results
        else:
            raise ValueError(f"Uncategorized column: {col}")
        ## Add dummy cols for any missed rows

    return pd.DataFrame(result_list).set_index("Feature")


def get_summary_col(
    header_prefix, df, all_categories, bin_cat_dict, num_missing_dict, feature_dict
):
    """
    Generates summary stats for a given subset of a df (denoted by header_prefix)
        - counts (%) for cat and median [25%, 75%] for numeric vars
    """
    binary_cols = feature_dict["binary_cols"]
    numerical_cols = feature_dict["numerical_cols"]
    nominal_cols = feature_dict["nominal_cols"]
    ordinal_cols = feature_dict["ordinal_cols"]
    total_entries = len(df)
    header = f"{header_prefix}  (n={total_entries})"
    summary_list = []
    for col in df.columns:
        if col not in numerical_cols:
            counts = df[col].value_counts()
            percentages = df[col].value_counts(normalize=True) * 100
            if col in list(bin_cat_dict.keys()) + nominal_cols + ordinal_cols:
                # only add header for cat var not in binary yes/no
                summary_list.append({"Feature": f"{col.upper()}", header: ""})
                if col in list(bin_cat_dict.keys()):
                    # replace 0/1 w/ actual var names
                    change_dict = bin_cat_dict[col]
                    counts = {change_dict[k]: v for k, v in counts.items()}
                    percentages = {change_dict[k]: v for k, v in percentages.items()}
            for entry in all_categories[col]:
                if col in binary_cols and col not in bin_cat_dict.keys():
                    # only record for binary YES under name of col
                    if entry == 0:
                        continue
                    else:
                        entry_name = col
                else:
                    entry_name = entry
                count_val = counts.get(entry, 0)
                percent_val_raw = percentages.get(entry, 0.0)
                if count_val > 0 and percent_val_raw < 0.1:
                    percent_val = "<0.1"
                else:
                    percent_val = round(percent_val_raw, 1)
                summary_list.append(
                    {
                        "Feature": f"{col.upper()}___{entry_name}",
                        # Get value count, if not existent, replace with 0
                        header: f"{count_val} ({percent_val})",
                    }
                )

        else:  ## ADD IQR + missing for numerical
            quantiles = np.round(df[col].quantile([0.25, 0.5, 0.75]).values.tolist(), 1)
            summary_list.append(
                {
                    "Feature": col.upper() + ", median (IQR)",
                    header: f"{quantiles[1]} ({quantiles[0]}-{quantiles[2]})",
                }
            )
            summary_list.append(
                {
                    "Feature": f"{col.upper()}___Unknown (imputed)",
                    header: num_missing_dict[col],
                },
            )
    return pd.DataFrame(summary_list).set_index("Feature")


def generate_fish_list(df, outcome_list, binary_cols):
    """
    Generates dictionary specifying, for each outcome,
    which features require fishers exact test due to low expected frequencies
    """
    fish_dict = {}
    for outcome_name in outcome_list:
        fish_dict[outcome_name] = []
        for col in binary_cols:
            contingency_table = pd.crosstab(df[col], df[outcome_name])
            _, _, _, expected_frequencies = chi2_contingency(contingency_table)
            if (expected_frequencies < 5).any():  # type: ignore
                fish_dict[outcome_name].append(col)
    return fish_dict


def get_summary_analysis_table(
    outcome_list,
    df,
    all_categories,
    bin_cat_dict,
    num_missing_dict,
    outcome_sub_cols,
    feature_dict,
    verbose=False,
):
    """
    Aggregates summary cols with statistical analysis cols for each outcome,
    and concats into one large df
    """
    if verbose:
        log("Starting run...")
    ## Get binary cols w/ low expected freq
    fish_dict = generate_fish_list(
        df=df,
        outcome_list=list(outcome_sub_cols.keys()),
        binary_cols=feature_dict["binary_cols"],
    )
    if verbose:
        log("Fish list generated, getting all patient summary...")
    ## Create all col
    all_col = get_summary_col(
        header_prefix="All patients",
        df=df,
        all_categories=all_categories,
        bin_cat_dict=bin_cat_dict,
        num_missing_dict=num_missing_dict,
        feature_dict=feature_dict,
    )
    col_list = [all_col]
    if verbose:
        log(f"Starting summary/analysis per outcome...")
    for idx, outcome_name in enumerate(outcome_list):
        if verbose:
            log(f"{outcome_name}\t({idx}/{len(outcome_list)})")
        ## Get col for each of pos and neg for this outcome
        pos_df = df[df[outcome_name] == 1]
        neg_df = df[df[outcome_name] == 0]
        assert len(pos_df) + len(neg_df) == len(df)
        pos_col = get_summary_col(
            header_prefix=f"{outcome_name}-Y",
            df=pos_df,
            all_categories=all_categories,
            bin_cat_dict=bin_cat_dict,
            num_missing_dict=num_missing_dict,
            feature_dict=feature_dict,
        )
        neg_col = get_summary_col(
            header_prefix=f"{outcome_name}-N",
            df=neg_df,
            all_categories=all_categories,
            bin_cat_dict=bin_cat_dict,
            num_missing_dict=num_missing_dict,
            feature_dict=feature_dict,
        )
        analysis_cols = get_analysis_df(
            df=df,
            outcome_name=outcome_name,
            outcome_sub_cols=outcome_sub_cols,
            fish_dict=fish_dict,
            feature_dict=feature_dict,
            bin_cat_dict=bin_cat_dict,
            all_categories=all_categories,
        )
        ## All dfs must have same index (Feature)
        col_list += [pos_col, neg_col, analysis_cols]
    # Concat into one large table
    sum_anlys_table = pd.concat(col_list, axis=1).reset_index(drop=False)
    sum_anlys_table["Feature"] = sum_anlys_table["Feature"].str.replace(
        r".*___", "", regex=True
    )
    return sum_anlys_table

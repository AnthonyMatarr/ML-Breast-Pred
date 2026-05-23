import warnings
import argparse
from pathlib import Path
import pandas as pd
from src.feat_eng import (
    get_feat_groups,
    log,
    tune_train_model,
    calibrate_model,
    eval_model,
    export_data,
)
from src.data_utils import get_data
from src.config import BASE_PATH
from optuna.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)


##########################################################################################
##########################################################################################
##########################################################################################
def is_monotonic(df):
    low_val = float(df.loc["Low", "lift"])
    med_val = float(df.loc["Moderate", "lift"])
    high_val = float(df.loc["High", "lift"])
    return low_val < med_val < high_val


def get_max_auprc(df):
    lift_list = df["test_ap (lift)"].apply(lambda r: float(r.split(" ")[1].strip("()")))
    return max(lift_list)


def get_max_auroc(df):
    auroc_list = df["test_auroc"].astype(float)
    return max(auroc_list)


def get_best_lift_bin(bin_dir, num_bins):
    best_lift = 0
    for iter_dir in bin_dir.iterdir():
        bin_file = pd.read_csv(
            iter_dir / f"bin_report_{num_bins}.tsv", index_col=0, sep="\t"
        )
        if not is_monotonic(bin_file):
            continue
        high_lift = float(bin_file.loc["High", "lift"])
        if high_lift > best_lift:
            best_lift = high_lift
    return best_lift


def is_viable(
    iter_val,
    best_auroc,
    best_auprc,
    best_bin_lift,
    metric_df,
    bin_df,
    threshold_perc,
):
    ## Get current values
    reduc_num = int(iter_val.split("_")[1])
    cur_row = metric_df[metric_df["reduction"] == reduc_num]

    cur_auprc = float(cur_row["test_ap (lift)"].iloc[0].split(" ")[1].strip("()"))
    cur_auroc = float(cur_row["test_auroc"].iloc[0])
    cur_bin_lift = float(bin_df.loc["High", "lift"])

    ## Compare
    does_increase = is_monotonic(bin_df)
    auprc_good = cur_auprc > (best_auprc * (1 - threshold_perc))
    aurpc_good = cur_auroc > (best_auroc * (1 - threshold_perc))
    bin_lift_good = cur_bin_lift > (best_bin_lift * (1 - threshold_perc))
    return does_increase and auprc_good and aurpc_good and bin_lift_good


def get_rank_df(perm_df, all_feats, outcome):
    n_feats = len(perm_df)
    rank_list = [round((n_feats - i + 1) / n_feats, 3) for i in range(1, n_feats + 1)]
    rank_col_name = f"Rank_{outcome}"
    rank_df = pd.DataFrame({"Feature": perm_df["Feature"], rank_col_name: rank_list})
    for feat in all_feats:
        if feat in rank_df["Feature"].to_list():
            continue
        rank_df.loc[len(rank_df)] = {"Feature": feat, rank_col_name: 0}
    assert len(rank_df) == rank_df["Feature"].nunique() == 50
    return rank_df


def select_iter(outcome, yr_rng, cohort, num_bins, iter_list, threshold_perc):
    """
    Returns iteration of minimum features that meets following requirements

    - Monotonic increase in event rate lift for risk bins
    - AUPRC >= AUPRC_max * (1-threshold_perc)
    - AUROC >= AUROC_max * (1-threshold_perc)
    - High risk lift >= high_risk_lift_max * (1-threshold_perc)

    Starts with lowest feature iteration and returns if viable

    Params
    -----
    outcome: str
        Outcome feature reduction results are being analyzed
    yr_rng: str
        Year range of data used for feature reduction run
    cohort: str
        One of ['all', 'cancer']
    num_bins: int
        One of [3,4]
    iter_list: list[str]
        List of iterations to loop over, in reverse order
    threshold_perc: float
        Maximum % difference in some metric to that of the max for AUPRC, AUROC, and high risk bin lift

    Returns
    ------
    Viable iteration w/ the smallest feature set
    """
    base_dir = BASE_PATH / "feat_eng" / cohort / outcome / yr_rng
    metric_df = pd.read_csv(base_dir / "metrics.tsv", sep="\t", index_col=0)
    best_auprc = get_max_auprc(metric_df)
    best_auroc = get_max_auroc(metric_df)
    best_bin_lift = get_best_lift_bin(bin_dir=base_dir / "bins", num_bins=num_bins)
    for iter_val in iter_list:
        bin_df = pd.read_csv(
            base_dir / "bins" / iter_val / f"bin_report_{num_bins}.tsv",
            index_col=0,
            sep="\t",
        )
        viable_flag = is_viable(
            iter_val=iter_val,
            best_auroc=best_auroc,
            best_auprc=best_auprc,
            best_bin_lift=best_bin_lift,
            metric_df=metric_df,
            bin_df=bin_df,
            threshold_perc=threshold_perc,
        )
        if viable_flag:
            return iter_val
    print(best_auprc)
    print(best_auroc)
    print(best_bin_lift)
    return None


##########################################################################################
##########################################################################################
##########################################################################################


def get_cols_from_groups(selected_groups, feature_groups):
    """
    Map a list of feature group names to their constituent column names.
    Silently skips any group name not found in feature_groups.
    """
    cols = []
    for group in selected_groups:
        cols += feature_groups.get(group, [])
    return cols


def filter_to_feature_set(X, selected_groups, feature_groups):
    """
    Subset a DataFrame to only the columns belonging to selected_groups.
    Silently skips columns not present in X (e.g. already dropped upstream).
    """
    cols = get_cols_from_groups(selected_groups, feature_groups)
    cols_present = [c for c in cols if c in X.columns]
    return X[cols_present]


def run_single_pass(
    outcome,
    cohort,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    selected_groups,
    n_cv_folds,
    num_optuna_trials,
    seed,
    n_cv_jobs,
    export_dir,
):
    """
    Single tune -> train -> calibrate -> eval pass for one outcome
    using a fixed feature set defined by selected_groups.

    Mirrors one iteration of run_iter_feat_red() without the
    reduction loop or permutation importance step.
    """
    feat_groups = get_feat_groups(is_cancer=(cohort == "cancer"))
    ## Subset to selected feature set
    X_tr = filter_to_feature_set(X_train, selected_groups, feat_groups)
    X_v = filter_to_feature_set(X_val, selected_groups, feat_groups)
    X_te = filter_to_feature_set(X_test, selected_groups, feat_groups)
    log(f"Feature set size: {X_tr.shape[1]} columns")

    ## Tune + train
    log("Running tuner...")
    model, best_params, cv_score = tune_train_model(
        X_train=X_tr,
        y_train=y_train,
        X_val=X_v,
        y_val=y_val,
        n_trials=num_optuna_trials,
        n_cv_folds=n_cv_folds,
        seed=seed,
        n_jobs=n_cv_jobs,
        study_name=f"{outcome}_confirmation",
    )
    log(f"Model best_iteration: {model.best_iteration_}")

    ## Calibrate
    log("Calibrating...")
    cal_model = calibrate_model(
        X=X_v,
        y=y_val,
        n_splits=n_cv_folds,
        seed=seed,
        model=model,
        n_cv_jobs=n_cv_jobs,
    )

    ## Evaluate
    log("Getting metrics...")
    metric_dict = eval_model(
        X_train=X_tr,
        y_train=y_train,
        X_val=X_v,
        y_val=y_val,
        X_test=X_te,
        y_test=y_test,
        model=cal_model,
        export_dir=export_dir / "bins",
    )

    ## Export scalar metrics
    import pandas as pd

    metric_df = pd.DataFrame([metric_dict])
    export_data(data_to_export=metric_df, export_path=export_dir / "metrics.tsv")
    return metric_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single tune/train/calibrate/eval pass for a fixed feature set"
    )
    parser.add_argument("--outcome", type=str)
    parser.add_argument("--cohort", type=str)
    parser.add_argument("--imp_dir", type=Path)
    parser.add_argument("--export_dir", type=Path)
    parser.add_argument(
        "--selected_groups",
        type=str,
        nargs="+",
        help="Feature group names from FEATURE_GROUPS to include",
    )
    parser.add_argument("--n_cv_folds", type=int)
    parser.add_argument("--num_optuna_trials", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n_cv_jobs", type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    log("Importing data...")
    data_dict = get_data(args.outcome, file_dir=args.imp_dir)
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"].values.ravel()
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"].values.ravel()
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"].values.ravel()

    log(f"Starting confirmation pass for outcome: {args.outcome}")
    run_single_pass(
        outcome=args.outcome,
        cohort=args.cohort,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        selected_groups=args.selected_groups,
        n_cv_folds=args.n_cv_folds,
        num_optuna_trials=args.num_optuna_trials,
        seed=args.seed,
        n_cv_jobs=args.n_cv_jobs,
        export_dir=args.export_dir / args.outcome,
    )
    log("DONE!")


if __name__ == "__main__":
    main()

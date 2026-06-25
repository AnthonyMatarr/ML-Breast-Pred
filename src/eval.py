# eval.py
from src.config import SEED
from src.data_utils import get_data, export_data, log

from pathlib import Path
import joblib
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from MLstatkit import Bootstrapping

BIN_NAME_DICT = {
    3: ["Low", "Moderate", "High"],
    4: ["Very Low", "Low", "Moderate", "High"],
}

# Used for risk bin plot
Y_MAX_DICT = {
    "SERIOUS": 27.5,
    "ANY": 31.0,
    "PNEUMO": 4.0,
    "CARDIAC_COMP": 1.4,
    "VTE": 2.25,
    "SEPSIS": 2.5,
    "SSI": 15.0,
    "UTI": 2.0,
    "RENAL": 1.1,
    "UNPLNREOP": 20.0,
    "MORT": 1.6,
}


def get_logspace_thresholds(y_proba, n_bins, lower=1e-5, upper=None):
    """
    Returns thresholds spaced evenly on a log scale within [lower, upper], gracefully handling zero predictions.
    """
    if upper is None:
        upper = np.percentile(y_proba, 99)
    # Avoid log(0) by setting very small lower bound
    lo = max(lower, np.min(y_proba[y_proba > 0]))
    hi = upper
    # Make log-spaced edges
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    thresholds = edges[1:-1]
    return thresholds


def get_threshold_str(bin_idx, thresholds, n_bins):
    """
    Formats bin threshold string
    """
    if bin_idx == 0:
        threshold_str = f"[0%, {thresholds[0]:.2%})"
    elif bin_idx == n_bins - 1:
        threshold_str = f"[{thresholds[-1]:.2%}, 100%]"
    else:
        threshold_str = f"[{thresholds[bin_idx-1]:.2%}, {thresholds[bin_idx]:.2%})"
    return threshold_str


def get_event_rate(n, in_bin_labels, in_bin_probs, n_bootstraps, seed):
    """
    Compute the observed event rate for a single risk bin with a bootstrap CI

    Falls back to the plain mean (and NaN CI) when the bin is empty, has a
    single class, or the bootstrap fails.

    Parameters
    ----------
    n : int
        Number of patients allocated to the bin.
    in_bin_labels : np.ndarray
        True binary labels for patients in the bin.
    in_bin_probs : np.ndarray
        Predicted probabilities for patients in the bin (passed to
        ``Bootstrapping`` but unused by the ``event_rate`` metric).
    n_bootstraps : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.
    """

    n_unique_classes = len(np.unique(in_bin_labels))
    if n > 0 and n_unique_classes > 1:
        try:
            event_rate_boot, ci_lower, ci_upper = Bootstrapping(
                in_bin_labels,
                in_bin_probs,  # this not used but need to pass
                metric_str="event_rate",
                n_bootstraps=n_bootstraps,
                random_state=seed,
                show_progress=False,
            )
        except RuntimeError:
            # Fallback if bootstrap fails
            event_rate_boot = float(in_bin_labels.mean())
            ci_lower = np.nan
            ci_upper = np.nan
    else:
        # Not enough data or only one class
        event_rate_boot = float(in_bin_labels.mean()) if n > 0 else np.nan
        ci_lower = np.nan
        ci_upper = np.nan
    event_rate = float(in_bin_labels.mean()) if n > 0 else np.nan
    event_rate_ci_str = f"{event_rate*100:.1f} ({ci_lower*100:.1f}-{ci_upper*100:.1f})"
    return event_rate_ci_str, event_rate


def get_bin_metrics(y_true, y_proba, thresholds, n_bootstraps, n_bins, seed):
    """
    Compute per-bin risk-stratification metrics for a set of predictions.

    Digitizes predicted probabilities into ``n_bins`` using ``thresholds``, then
    for each bin computes allocation counts, positive counts, bootstrapped event
    rate, lift over the cohort base rate, mean prediction, and the bin's
    threshold range.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_proba : np.ndarray
        Predicted probabilities
    thresholds : array-like
        Interior bin edges (length of n_bins - 1)
    n_bootstraps : int
        Bootstrap iterations for event-rate CIs
    n_bins : int
        Number of risk bins
    seed : int
        Random seed for reproducibility
    """

    bin_dict = {}
    bin_names = BIN_NAME_DICT[n_bins]
    thresholds = np.asarray(thresholds, dtype=float).flatten()
    bin_indices = np.digitize(y_proba, thresholds, right=False)  # 0,1,...,n_bins-1
    tot_n = len(y_true)
    tot_n_pos = int(np.sum(y_true, axis=0))
    tot_event_rate = tot_n_pos / tot_n
    for b in range(n_bins):
        ## Get labels + probs of allocated to this bin
        mask = bin_indices == b
        n = int(mask.sum())
        in_bin_labels = y_true[mask]
        in_bin_probs = y_proba[mask]
        # ================= Populate bin dict =================
        bin_name = bin_names[b]
        ## Total patients in bin (% of test cohort)
        perc_all_pat = n / tot_n
        n_perc = f"{n} ({perc_all_pat*100:.1f})"
        # Total pos patients in bin (% of all pos in test)
        in_bin_n_pos = int(np.sum(in_bin_labels, axis=0))
        perc_pos = float(in_bin_n_pos / tot_n_pos) if tot_n_pos > 0 else np.nan
        n_perc_pos = f"{in_bin_n_pos} ({perc_pos*100:.1f})"

        ## Event rate w/ CIs
        event_rate_ci_str, event_rate = get_event_rate(
            n=n,
            in_bin_labels=in_bin_labels,
            in_bin_probs=in_bin_probs,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )
        ## Lift
        lift = float(event_rate / tot_event_rate) if n > 0 else np.nan
        ## thresholds
        threshold_str = get_threshold_str(
            bin_idx=b, thresholds=thresholds, n_bins=n_bins
        )
        ## mean model output
        mean_output = float(in_bin_probs.mean()) if n > 0 else np.nan
        bin_dict[bin_name] = {
            "for_table": {
                "Total No. allocated (% of all patients)": n_perc,
                "Positive No. allocted (% of all positives)": n_perc_pos,
                "Event rate, % (95% CI)": event_rate_ci_str,
                "Lift": round(lift, 2),
                "Mean prediction, %": round(mean_output * 100, 2),
                "Threshold, %": threshold_str.replace("%", ""),
            },
            "for_plot": {
                "total_allocated": n,
                "event_rate": event_rate,
                "mean_pred": mean_output,
                "threshold": threshold_str,
            },
        }
    return bin_dict


def plot_risk_bar_dot(bin_report_dict, n_bins, title, y_max=1.0):
    """
    Create risk stratification plot with bar graph showing observed event
    rates and overlaid mean predictions per bin.
    """
    ## Label bins w/ thresholds
    event_rates = []
    bins_labels = []
    mean_preds = []
    counts = []
    bin_names = BIN_NAME_DICT[n_bins]
    for bin_name in bin_names:
        cur_dict = bin_report_dict[bin_name]["for_plot"]
        threshold_str = cur_dict["threshold"]
        bins_labels.append(f"{bin_name}\n{threshold_str}")
        event_rates.append(cur_dict["event_rate"] * 100)
        mean_preds.append(cur_dict["mean_pred"] * 100)
        counts.append(cur_dict["total_allocated"])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(n_bins), event_rates, color="C0", alpha=0.7, label="Event Rate (%)")
    ax.plot(
        range(n_bins), mean_preds, "o-", color="C1", label="Avg. Predicted Risk (%)"
    )
    ax.set_title(title)
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bins_labels, rotation=0)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.linspace(0, y_max + (y_max / 10), 5))
    ax.set_ylabel("% With Outcome / Mean Prediction (%)")
    ax.set_xlabel("Risk Bin")
    ax.legend(loc="upper left")
    # n=XXXX at bottom of bar
    for i, n in enumerate(counts):
        ax.text(
            i,
            0.0,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="k",
        )
    fig.tight_layout()
    return fig, ax


def get_roc_stats(y_true, y_proba, n_bootstraps=5000, seed=SEED, show_progress=False):
    """
    Get AUROC w/ CIs, determine threshold for hard predictions
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc, lower_CI, upper_CI = Bootstrapping(
        y_true,
        y_proba,
        random_state=seed,
        metric_str="roc_auc",
        n_bootstraps=n_bootstraps,
        show_progress=show_progress,
    )
    auc_string = f"{auc:.3f} ({lower_CI:.3f}-{upper_CI:.3f})"

    ## Youden's J to determine threshold ##
    pr_dif = tpr - fpr
    optimal_idx = np.argmax(pr_dif)
    optimal_threshold = thresholds[optimal_idx]
    return {
        "roc_stats": {
            "fpr": fpr,
            "tpr": tpr,
        },
        "auc_string": auc_string,
        "threshold_raw": optimal_threshold,
        "threshold_round": round(optimal_threshold, 3),
    }


def get_pr_stats(y_true, y_proba, n_bootstraps=5000, seed=SEED, show_progress=False):
    """
    Compute the precision-recall curve, bootstrapped AUPRC, and AUPRC lift.
    """

    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    ap, lower_CI, upper_CI = Bootstrapping(
        y_true,
        y_proba,
        random_state=seed,
        metric_str="average_precision",
        n_bootstraps=n_bootstraps,
        show_progress=show_progress,
    )

    ap_string = f"{ap:.3f} ({lower_CI:.3f}-{upper_CI:.3f})"
    ap_lift = ap / float(np.mean(y_true))
    return {
        "prc_stats": {
            "precision": precision,
            "recall": recall,
        },
        "ap_string": ap_string,
        "ap_lift": round(ap_lift, 1),
    }


def plot_PRC(recall, precision, ap_string, set_type, event_rate, model_name):
    """
    Plot a single precision-recall curve with a base-rate baseline.

    Parameters
    ----------
    recall, precision : np.ndarray
        Curve coordinates.
    ap_string : str
        Formatted AUPRC string for the legend.
    set_type : str
        Cohort label ("train"/"val"/"test"), shown in title and legend.
    event_rate : float
        Base positive rate, drawn as the random-classifier baseline.
    model_name : str
        Model label for the title.
    """

    fig, ax = plt.subplots(figsize=(12, 8))
    # Get baseline
    ax.hlines(
        event_rate,  # all the same
        0,
        1,
        color="gray",
        linestyle="--",
        label="Random Classifier",
    )
    ax.plot(
        recall,
        precision,
        lw=4,
        label=f"{set_type.capitalize()} AUPRC = {ap_string}",
    )
    ax.set_xlim([0.0, 1.0]) # type: ignore
    ax.set_ylim([0.0, 1.0]) # type: ignore
    ## Add meta
    ax.set_title(
        f"{model_name} {set_type} PR Curve",
        fontweight="semibold",
        fontsize=25,
    )
    ax.set_xlabel("Recall", fontsize=21, fontweight=550)
    ax.set_ylabel("Precision", fontsize=21, fontweight=550)
    ax.legend(loc="upper right", prop={"size": 19, "weight": 550})
    fig.tight_layout()
    return fig, ax


def plot_ROC(data_dict, model_name):
    """
    Plot ROC curve (train, val, test all in single plot)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
    for set_type, set_dict in data_dict.items():
        fpr = set_dict["roc_stats"]["fpr"]
        tpr = set_dict["roc_stats"]["tpr"]
        ax.plot(fpr, tpr, lw=4, label=f"{set_type} AUROC = {set_dict['auc_string']}")

    ax.set_xlim([0.0, 1.0]) # type: ignore
    ax.set_ylim([0.0, 1.05]) # type: ignore
    ax.set_xlabel("False Positive Rate", fontsize=21, fontweight=550)
    ax.set_ylabel("True Positive Rate", fontsize=21, fontweight=550)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_title(f"{model_name} ROC", fontweight="semibold", fontsize=25)
    ax.legend(loc="lower right", prop={"size": 19, "weight": 550})
    fig.tight_layout()
    return fig, ax


def get_discrimination_str(
    *_,
    y_true,
    y_proba,
    metric_str,
    threshold,
    n_bootstraps=5000,
    random_state=SEED,
    bin_thresholds=None,
    show_progress=False,
):
    """
    Calculate a value and 95% CI for a given metric using MLStakit.Bootstrapping

    Parameters
    ----------
    y_true: numpy.ndarray
        True binary class labels
    y_proba: numpy.ndarray
        Continues predicted probabilities
    metric_str: str
        Specify the metric type to get values for
    threshold: float
        Threshold value to use for converting probabilities into hard labels
    n_bootstraps: Optional int; defaults to 5000
        Number of iterations to run bootstrap method for
    random_state: Optional int; defaults to SEED from src.config
        Controls determinism

    Returns
    -------
    final_str: String of format
        '<metric_val> (<ci_lower>, <ci_upper>)'
    Raises
    ------
    ValueError:
        -If positional arguments are passed
        -If an unaccepted str type is passed. Must be one of:
            'f1', 'accuracy', 'recall', 'precision', 'roc_auc', 'average_precision', 'pr_auc', 'ici', 'brier'
    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    if metric_str == "ici":
        metric_val, ci_lower, ci_upper = Bootstrapping(
            y_true,
            y_proba,
            metric_str=metric_str,
            n_bootstraps=n_bootstraps,
            confidence_level=0.95,
            threshold=threshold,
            random_state=random_state,
            bin_thresholds=bin_thresholds,
            show_progress=show_progress,
        )
    else:
        metric_val, ci_lower, ci_upper = Bootstrapping(
            y_true,
            y_proba,
            metric_str=metric_str,
            n_bootstraps=n_bootstraps,
            confidence_level=0.95,
            threshold=threshold,
            random_state=random_state,
            show_progress=show_progress,
        )
    final_str = f"{metric_val:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
    return final_str


def eval_outcome_model(
    outcome,
    model_name,
    model_imp_dir,
    data_imp_dir,
    results_dir,
    n_bootstraps,
    n_bins,
    show_progress,
    metrics_strs,
    seed,
):
    """
    Run the full evaluation pipeline for one (outcome, model) pair and export results.

    Loads the trained model and its data splits, then computes and writes:
    test-set predictions, risk-bin thresholds/tables/plots, ROC and PR curves,
    and bootstrapped discrimination metrics

    Parameters
    ----------
    outcome : str
        Outcome/target name
    model_name : str
        Name of the model
    model_imp_dir: pathlib.Path
        Input model dir
    data_imp_dir: pathlib.Path
        Input data dir
    results_dir : pathlib.Path
        Output results root.
    n_bootstraps : int
        Bootstrap iterations for all CI estimates.
    n_bins : int
        Number of risk bins (3 or 4).
    show_progress : bool
        Whether to show bootstrap progress bars.
    metrics_strs : list[str]
        Discrimination metrics to compute (e.g. "f1", "brier", "ici").
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    None
        All outputs are written to disk as a side effect.
    """

    log(f"Running eval for outcome: {outcome}; model: {model_name}")
    ###############################################################################
    ############### 1) SET UP BASE DATA + WHERE RESULTS ARE TRACKED ###############
    ###############################################################################
    log("Getting base data and probability outputs...")
    raw_data_dict = get_data(outcome_folder=outcome, file_dir=data_imp_dir)
    model = joblib.load(model_imp_dir / outcome / f"{model_name}.joblib")
    data_dict = {
        "train": {
            "X": raw_data_dict["X_train"],
            "y": raw_data_dict["y_train"].values.ravel(),
        },
        "val": {
            "X": raw_data_dict["X_val"],
            "y": raw_data_dict["y_val"].values.ravel(),
        },
        "test": {
            "X": raw_data_dict["X_test"],
            "y": raw_data_dict["y_test"].values.ravel(),
        },
    }
    for set_type, set_dict in data_dict.items():
        data_dict[set_type]["y_proba"] = model.predict_proba(set_dict["X"])[:, 1]
    ## export test preds (used in app later)
    all_predictions = pd.DataFrame(
        {"prob": data_dict["test"]["y_proba"], "label": data_dict["test"]["y"]}
    )
    export_data(
        data_to_export=all_predictions,
        export_path=results_dir / "app/all_preds" / outcome / f"{model_name}.parquet",
    )
    ########################################################
    ##################### 2) RISK BINS #####################
    ########################################################
    log("Building risk bins...")
    # bin thresholds
    train_val_probs = np.concatenate(
        [data_dict["train"]["y_proba"], data_dict["val"]["y_proba"]]
    )
    bin_thresholds = get_logspace_thresholds(train_val_probs, n_bins=n_bins)
    export_data(
        None,
        export_path=results_dir
        / "app"
        / "bin_thresholds"
        / outcome
        / f"{model_name}.npz",
        thresholds=bin_thresholds,
    )
    ## metrics
    bin_dict = get_bin_metrics(
        y_true=data_dict["test"]["y"],
        y_proba=data_dict["test"]["y_proba"],
        thresholds=bin_thresholds,
        n_bootstraps=n_bootstraps,
        n_bins=n_bins,
        seed=seed,
    )
    ## aggregate + export
    df_rows = []
    for bin_name in bin_dict.keys():
        table_dict = bin_dict[bin_name]["for_table"]
        df_rows.append(
            {
                "outcome": outcome,
                "model": model_name,
                "bin": bin_name,
                **table_dict,
            }
        )
    bin_df = pd.DataFrame(df_rows)
    export_data(
        data_to_export=bin_df,
        export_path=results_dir
        / "tables"
        / "bins"
        / outcome
        / f"{model_name}_bin_table.xlsx",
    )
    ## Risk bar plots
    y_max = Y_MAX_DICT[outcome]
    bin_fig, bin_ax = plot_risk_bar_dot(
        bin_dict, n_bins=n_bins, title=f"{outcome}-{model_name}", y_max=y_max
    )
    export_data(
        data_to_export=bin_fig,
        export_path=results_dir
        / "figures"
        / "bins"
        / outcome
        / f"{model_name}_bin_plot.pdf",
    )
    ############################################################
    ##################### 3) AUROC + AUPRC #####################
    ############################################################
    log("Getting AUROC and AUPRC...")
    ## get metrics per train, val, test, + plot prc
    for set_type, set_dict in data_dict.items():
        roc_dict = get_roc_stats(
            y_true=set_dict["y"],
            y_proba=set_dict["y_proba"],
            n_bootstraps=n_bootstraps,
            seed=seed,
            show_progress=show_progress,
        )
        prc_dict = get_pr_stats(
            y_true=set_dict["y"],
            y_proba=set_dict["y_proba"],
            n_bootstraps=n_bootstraps,
            seed=seed,
            show_progress=show_progress,
        )
        data_dict[set_type].update({**roc_dict, **prc_dict})
        ## PLOT PRC (each needs own plot bc diff event rates)
        prc_fig, prc_ax = plot_PRC(
            recall=prc_dict["prc_stats"]["recall"],
            precision=prc_dict["prc_stats"]["precision"],
            ap_string=prc_dict["ap_string"],
            set_type=set_type,
            event_rate=float(np.mean(set_dict["y"])),
            model_name=model_name,
        )
        export_data(
            data_to_export=prc_fig,
            export_path=results_dir
            / "figures"
            / "PRC"
            / set_type
            / outcome
            / f"{model_name}.pdf",
        )
    ## plot ROC (all in one plot)
    roc_fig, roc_ax = plot_ROC(data_dict=data_dict, model_name=model_name)
    export_data(
        data_to_export=roc_fig,
        export_path=results_dir / "figures" / "ROC" / outcome / f"{model_name}.pdf",
    )
    #####################################################################
    ##################### 4) DISCRIMINATION METRICS #####################
    #####################################################################
    log("Getting discrimination metrics...")
    metrics_rows = []
    # use threshold obtained from val ROC for all sets
    binary_threshold = data_dict["val"]["threshold_raw"]
    for set_type, set_dict in data_dict.items():
        for metric in metrics_strs:
            if metric == "ici":
                bin_thresholds_for_ici = bin_thresholds
            else:
                bin_thresholds_for_ici = None
            metric_str = get_discrimination_str(
                y_true=set_dict["y"],
                y_proba=set_dict["y_proba"],
                metric_str=metric,
                threshold=binary_threshold,
                n_bootstraps=n_bootstraps,
                random_state=seed,
                bin_thresholds=bin_thresholds_for_ici,
                show_progress=show_progress,
            )
            data_dict[set_type][metric] = metric_str
        ## one row in metric df per (train, val, test)
        metrics_rows.append(
            {
                "Cohort": set_type,
                "Model": model_name,
                "AUROC": set_dict["auc_string"],
                "AUPRC": set_dict["ap_string"],
                "AUPRC lift": set_dict["ap_lift"],
                "Brier": set_dict["brier"],
                "F1-score": set_dict["f1"],
                "Accuracy": set_dict["accuracy"],
                "Precision": set_dict["precision"],
                "Recall": set_dict["recall"],
                "ICI": set_dict["ici"],
                "Threshold": binary_threshold,
            }
        )
    ## aggregate + export
    metric_df = pd.DataFrame(metrics_rows)
    export_data(
        data_to_export=metric_df,
        export_path=results_dir
        / "tables"
        / "metrics"
        / outcome
        / f"{model_name}_metrics.xlsx",
    )
    plt.close("all")
    log("DONE!")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="Eval", description="Evaluate a single model for a given outcome"
    )
    parser.add_argument("--outcome", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_imp_dir", type=Path, required=True)
    parser.add_argument("--data_imp_dir", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--n_bootstraps", type=int, required=True)
    parser.add_argument("--n_bins", type=int, required=True, choices=[3, 4])
    parser.add_argument(
        "--show_progress", type=str, required=True, choices=["True", "False"]
    )
    parser.add_argument("--metrics_strs", nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    show_progress = True if args.show_progress == "True" else False
    eval_outcome_model(
        outcome=args.outcome,
        model_name=args.model_name,
        model_imp_dir=args.model_imp_dir,
        data_imp_dir=args.data_imp_dir,
        results_dir=args.results_dir,
        n_bootstraps=args.n_bootstraps,
        n_bins=args.n_bins,
        show_progress=show_progress,
        metrics_strs=args.metrics_strs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

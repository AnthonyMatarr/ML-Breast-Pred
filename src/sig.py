from src.config import SEED
import warnings
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from MLstatkit import Delong_test
from sklearn.metrics import roc_auc_score


def get_pval_df(X_test, y_test, model_dict):
    """
    Helper function to generate dataframe used for heatmap construction
    """
    models = list(model_dict.keys())
    pval_df = pd.DataFrame(0.0, index=models, columns=models)
    winner_df = pd.DataFrame("0", index=models, columns=models)
    for model_nameA, modelA in model_dict.items():
        for model_nameB, modelB in model_dict.items():
            if model_nameA == model_nameB:
                continue
            y_proba_A = modelA.predict_proba(X_test)[:, 1]
            y_proba_B = modelB.predict_proba(X_test)[:, 1]
            results = Delong_test(
                true=y_test,
                prob_A=y_proba_A,
                prob_B=y_proba_B,
                alpha=0.95,
                return_auc=True,
                return_ci=True,
                n_boot=1,
                random_state=SEED,
                verbose=0,
            )
            p_val = results[1]
            # p_val = format_p_val(p_val)
            auc_A = results[4]
            auc_B = results[5]
            if auc_A > auc_B:
                best_model = "A"  # row is better
            else:
                best_model = "B"  # col is better
            pval_df.loc[model_nameA, model_nameB] = p_val

            winner_df.loc[model_nameA, model_nameB] = best_model
    return pval_df, winner_df


def generate_delong_heatmap(X_test, y_test, model_dict, outcome_name, result_path=None):
    """
    For a single outcome, generate and plots heatmap comparing models w/ DeLongs test p-values

    Parameters
    ---------
    X_test: pandas dataframe
        Tabular data used to generate probabilities for AUROC values
    y_test: numpy.ndarray
        True labels of tabular data used to generate AUROC values
    model_dict: dict
        Dictionary mapping outcome names to a dictionary mapping model names to models
        Format:
        {
            <outcome_name> str: {
                <model_name> str: <model>
            }
        }
    outcome_name: str
        Name of outcome whose models are being compared
    result_path: Optional pathlib.Path; defaults to None
        Path to results directory where raw p-val table and heatmap will be saved
        If left None nothing is exported
    """
    pval_df, winner_df = get_pval_df(X_test, y_test, model_dict)
    mask = np.zeros_like(pval_df)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [(0.0, "green"), (0.15, "purple"), (1.0, "blue")]  # 0  # 0.05  # 1
    custom_cmap = LinearSegmentedColormap.from_list("custom_pval", colors)
    norm = TwoSlopeNorm(vmin=0, vcenter=0.05, vmax=1)
    sns.heatmap(
        pval_df,
        cmap=custom_cmap,
        annot=True,
        fmt=".4f",
        # center=0.05,
        vmin=0,
        vmax=1,
        # center=1,
        linewidths=1,
        # linecolor="grey",
        mask=mask,
        square=True,
        cbar_kws={"label": "p-value"},
        ax=ax,
    )
    for i in range(pval_df.shape[0]):
        for j in range(pval_df.shape[1]):
            if i <= j or pval_df.iloc[i, j] > 0.05:  # type: ignore
                continue
            winner = winner_df.iloc[i, j]
            # Place the marker: could use text or Unicode triangle/arrow
            if winner == "A":  # Row model wins
                annotation = "\u2190"  # Left triangle (row wins)
            else:
                annotation = "\u2193"  # Down triangle (column wins)
            ax.text(
                j + 0.5,
                i + 0.75,
                annotation,
                va="center",
                ha="center",
                fontsize=12,
                color="black",
            )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title(f"{outcome_name}- Model vs Model AUROC comparison")
    plt.tight_layout()

    ### Export
    if result_path:
        ## Heatmap
        heat_path = result_path / "figures" / "heatmap" / f"{outcome_name}_heatmap.pdf"
        if heat_path.exists():
            warnings.warn(f"Over-writing heatmap {heat_path}")
            heat_path.unlink()
        heat_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(heat_path, bbox_inches="tight")
        ##P-val df
        pval_path = result_path / "tables" / "p_vals" / f"{outcome_name}_pvals.xlsx"
        if pval_path.exists():
            warnings.warn(f"Over-writing heatmap {pval_path}")
            pval_path.unlink()
        pval_path.parent.mkdir(exist_ok=True, parents=True)
        pval_df.to_excel(pval_path)
    plt.show()


def get_friedman_df(*_, model_dict, outcome_dict):
    """
    Create dataframe used for friedman test to compare models across all outcomes
    NOTE: Transpose to compare outcomes across all models
    Parameters
    ---------
    model_dict: dict
        Dictionary mapping outcome names to a dictionary mapping model names to models
        Format:
        {
            <outcome_name> str: {
                <model_name> str: <model>
            }
        }
    outcome_dict: dict
        Dictionary mapping outcome names to a dictionary mapping data subsets to data
        Format:
        {
            <outcome_name> str: {
                'X_train': pandas dataframe
                'y_train': pandas dataframe
            }
        }
    Returns
    -------
    friedman_df:
        Pandas dataframe with:
            Rows: Outcome names
            Columns: Model names
            Values: AUROC of a model in predicting an outcome
    Raises
    -----
    ValueError:
        If positional arguments are given
    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    ## Generate df with rows=outcomes, cols=models
    for out_models in model_dict.values():
        models = list(out_models.keys())
        break
    outcomes = list(model_dict.keys())
    friedman_df = pd.DataFrame(0.0, columns=models, index=outcomes)
    for outcome_name, outcome_data in outcome_dict.items():
        X_test = outcome_data["X_test"]
        y_test = outcome_data["y_test"]
        cur_model_dict = model_dict[outcome_name]

        for model_name, model in cur_model_dict.items():
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(
                y_true=y_test,
                y_score=y_proba,
            )
            friedman_df.loc[outcome_name, model_name] = roc_auc
    return friedman_df

import warnings

import argparse
from datetime import datetime


from src.data_utils import get_data, export_data
from src.eval import get_logspace_thresholds, get_bin_metrics, plot_risk_bar_dot

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import StratifiedKFold

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import lightgbm as lgb
from pathlib import Path

import optuna
from optuna.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

#### GLOBALS ####
EARLY_STOP_ROUNDS = 50
LGBM_MAX_ROUNDS = 1000


def get_feat_groups(is_cancer):
    feat_groups = {
        # ── Continuous / binary singletons ───────────────────────────────────────
        "AGE": ["AGE"],
        "PRALBUM": ["PRALBUM"],
        "PRWBC": ["PRWBC"],
        "PRHCT": ["PRHCT"],
        "PRPLATE": ["PRPLATE"],
        "OPERYR": ["OPERYR"],
        "BMI": ["BMI"],
        "ASACLAS": ["ASACLAS"],
        "SEX": ["SEX"],
        "HISPANIC": ["HISPANIC"],
        "HXCHF": ["HXCHF"],
        "ASCITES": ["ASCITES"],
        "DIALYSIS": ["DIALYSIS"],
        "HYPERMED": ["HYPERMED"],
        "SMOKE": ["SMOKE"],
        "DISCANCR": ["DISCANCR"],
        "URGENCY": ["URGENCY"],
        "INOUT": ["INOUT"],
        "BLEEDDIS": ["BLEEDDIS"],
        "TRANSFUS": ["TRANSFUS"],
        "VENTILAT": ["VENTILAT"],
        "HXCOPD": ["HXCOPD"],
        # ── One-hot encoded categoricals ─────────────────────────────────────────
        "RACE": [
            "RACE_American Indian or Alaska Native",
            "RACE_Asian",
            "RACE_Black or African American",
            "RACE_Native Hawaiian or Pacific Islander",
            "RACE_White",
            "RACE_otherUnknown",
        ],
        "DIABETES": ["DIABETES_INSULIN", "DIABETES_NO", "DIABETES_ORAL"],
        # "RENAFAIL": ["RENAFAIL_No", "RENAFAIL_Unknown_21", "RENAFAIL_Yes"], #missing so left out of useable feats
        "STEROID": ["STEROID"],
        # "DYSPNEA": ["DYSPNEA_No", "DYSPNEA_Unknown_21_24", "DYSPNEA_Yes"], #missing so left out of useable feats
        # "WNDINF": ["WNDINF_No", "WNDINF_Unknown_21_24", "WNDINF_Yes"], #missing so left out of useable feats
        # "WTLOSS": ["WTLOSS_No", "WTLOSS_Unknown_21_24", "WTLOSS_Yes"], #missing so left out of useable feats
        "SURGINDICD": [
            "SURGINDICD_ABBREASTICD",
            "SURGINDICD_ABSICD",
            "SURGINDICD_BENIGNICD",
            "SURGINDICD_CARCINOMAICD",
            "SURGINDICD_CONGICD",
            "SURGINDICD_INFLOTHERICD",
            "SURGINDICD_MALIGNANTICD",
            "SURGINDICD_METASTATICICD",
            "SURGINDICD_PROPHYLACTICICD",
        ],
        "ANESTHES": ["ANESTHES_General", "ANESTHES_MAC", "ANESTHES_otherUnknown"],
        "SURGSPEC": [
            "SURGSPEC_General",
            "SURGSPEC_Plastics",
            "SURGSPEC_otherUnknown",
        ],
        # ── CPT procedure flags (individual binary, not OHE) ─────────────────────
        "SNLBCPT": ["SNLBCPT"],
        "ALNDCPT": ["ALNDCPT"],
        "PARTIALCPT": ["PARTIALCPT"],
        "SUBSIMPLECPT": ["SUBSIMPLECPT"],
        "RADICALCPT": ["RADICALCPT"],
        "MODIFIEDRADICALCPT": ["MODIFIEDRADICALCPT"],
        "IMMEDIATECPT": ["IMMEDIATECPT"],
        "DELAYEDCPT": ["DELAYEDCPT"],
        "TEINSERTIONCPT": ["TEINSERTIONCPT"],
        "TEEXPANDERCPT": ["TEEXPANDERCPT"],
        "FREECPT": ["FREECPT"],
        "LATCPT": ["LATCPT"],
        "SINTRAMCPT": ["SINTRAMCPT"],
        "SINTRAMSUPERCPT": ["SINTRAMSUPERCPT"],
        "BITRAMCPT": ["BITRAMCPT"],
        "MASTOCPT": ["MASTOCPT"],
        "BREASTREDCPT": ["BREASTREDCPT"],
        "FATGRAFTCPT": ["FATGRAFTCPT"],
        "ADJTISTRANSCPT": ["ADJTISTRANSCPT"],
        "AUGPROSIMPCPT": ["AUGPROSIMPCPT"],
        "OTHERRECONTECHCPT": ["OTHERRECONTECHCPT"],
        "REVRECBREASTCPT": ["REVRECBREASTCPT"],
        "NPWTCPT": ["NPWTCPT"],
    }
    # SURGINDICD is binary when cancer cohort
    if is_cancer:
        feat_groups["SURGINDICD"] = ["SURGINDICD"]
    return feat_groups


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


###### tune_train_model() helpers ######
def _make_cv_splits(y, n_folds, seed):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def _search_space(trial):
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", 1.0, 100.0, log=True
        ),
    }


def _build_params(trial_params, y, seed, n_jobs):
    """
    Merge Optuna-suggested params with fixed constants.
    Kept separate bc trial_params gets saved and compared across iterations;
    the fixed constants are implementation details that never change.
    """
    return {
        **trial_params,
        "objective": "binary",
        # "metric": "binary_logloss",  # instead of AUPRC bc of potenital instability
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": seed,
        "n_jobs": 0,  # 0 uses default set on OpenMP
    }


def tune_train_model(
    X_train, y_train, X_val, y_val, n_trials, n_cv_folds, seed, n_jobs, study_name
):
    """
    Tune LightGBM hyperparams via Optuna, then fit a final model on the full training set
    """
    cv_splits = _make_cv_splits(y=y_train, n_folds=n_cv_folds, seed=seed)

    ## Optuna objective
    def objective(trial):
        params = _build_params(_search_space(trial), y_train, seed, n_jobs)
        fold_aps = []
        fold_iters = []

        for fold_idx, (tr_idx, tst_idx) in enumerate(cv_splits):
            # Do CV split
            X_tr = X_train.iloc[tr_idx]
            X_tst = X_train.iloc[tst_idx]
            y_tr = y_train[tr_idx]
            y_tst = y_train[tst_idx]

            fold_model = lgb.LGBMClassifier(**params, n_estimators=LGBM_MAX_ROUNDS)
            fold_model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_tst, y_tst)],
                eval_metric="binary_logloss",  # instead of AP bc AP unstable w/ low event rate
                callbacks=[
                    lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            y_proba = fold_model.predict_proba(X_tst)[:, 1]
            fold_aps.append(float(average_precision_score(y_tst, y_proba)))
            fold_iters.append(int(fold_model.best_iteration_))
            ## Report to pruner
            trial.report(fold_aps[-1], step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("mean_best_iter", int(np.mean(fold_iters)))
        trial.set_user_attr("fold_aps", fold_aps)
        return float(np.mean(fold_aps))

    ## Run study (tuning)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    cv_score = float(best_trial.value)
    mean_best_iter = int(best_trial.user_attrs.get("mean_best_iter", LGBM_MAX_ROUNDS))
    best_params = best_trial.params  # tuned only; no fixed constants
    full_params = _build_params(best_params, y_train, seed, n_jobs)

    ## Final fit
    n_estimators_ceiling = max(int(mean_best_iter * 1.5), LGBM_MAX_ROUNDS // 2)
    model = lgb.LGBMClassifier(
        **full_params,
        n_estimators=n_estimators_ceiling,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model, best_params, cv_score


def format_bin_dict(bin_dict):
    for bin_name in bin_dict.keys():
        # event rate
        event_rate_w_cis = bin_dict[bin_name]["event_rate_w_CIs"]
        bin_dict[bin_name][
            "event_rate"
        ] = f"{event_rate_w_cis['event_rate']:.2%} ({event_rate_w_cis['lower_CI']:.2%}, {event_rate_w_cis['upper_CI']:.2%})"
        # n tot
        n_perc = bin_dict[bin_name]["n_perc"]
        bin_dict[bin_name]["total n"] = f"{n_perc['n']} ({n_perc['perc']:.2%})"
        # n pos
        n_perc_pos = bin_dict[bin_name]["perc_all_pos"]
        bin_dict[bin_name]["pos n"] = f"{n_perc_pos['n']} ({n_perc_pos['perc']:.2%})"
        # remove event_rate_w_CIs from keys
        bin_dict[bin_name] = {
            k: v
            for k, v in bin_dict[bin_name].items()
            if k not in ["event_rate_w_CIs", "n_perc", "perc_all_pos"]
        }
        ## ROUND
        bin_dict[bin_name]["lift"] = round(bin_dict[bin_name]["lift"], 2)
        bin_dict[bin_name]["mean_model_output"] = round(
            bin_dict[bin_name]["mean_model_output"], 2
        )
    return bin_dict


def eval_model(X_train, y_train, X_val, y_val, X_test, y_test, model, export_dir):
    ## Get preds
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_val = model.predict_proba(X_val)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]
    ## Get bin thresholds
    train_val_probs = np.concatenate([y_proba_train, y_proba_val])
    for n_bins in [3, 4]:
        bin_thresholds = get_logspace_thresholds(train_val_probs, n_bins=n_bins)
        bin_metric_dict = get_bin_metrics(
            y_true=y_test,
            y_proba=y_proba_test,
            thresholds=bin_thresholds,
            bin_report_dict={},  # empty dict to initialize
            n_bootstraps=10,  # not too interested in CIs here
            n_bins=n_bins,
        )
        ### Plot bins + export
        ax = plot_risk_bar_dot(bin_metric_dict, y_max=0.4, n_bins=n_bins)
        fig = ax.get_figure()
        export_data(
            data_to_export=fig, export_path=export_dir / f"bin_plot_{n_bins}.pdf"
        )
        plt.close()
        ## Format dict + export
        bin_metric_dict = format_bin_dict(bin_metric_dict)
        bin_metric_df = pd.DataFrame.from_dict(bin_metric_dict, orient="index")
        export_data(
            data_to_export=bin_metric_df,
            export_path=export_dir / f"bin_report_{n_bins}.tsv",
        )
    ## Get AUROC + AP
    ## Will append these to export later
    event_rate = np.mean(y_test)
    train_ap = average_precision_score(y_true=y_train, y_score=y_proba_train)
    val_ap = average_precision_score(y_true=y_val, y_score=y_proba_val)
    test_ap = average_precision_score(y_true=y_test, y_score=y_proba_test)
    res_dict = {
        "event_rate": f"{event_rate:.2%}",
        "num_feats": int(X_test.shape[1]),
        "train_auroc": round(roc_auc_score(y_true=y_train, y_score=y_proba_train), 4),
        "val_auroc": round(roc_auc_score(y_true=y_val, y_score=y_proba_val), 4),
        "test_auroc": round(roc_auc_score(y_true=y_test, y_score=y_proba_test), 4),
        "train_ap (lift)": f"{train_ap:.4f} ({(train_ap/event_rate):.2f})",
        "val_ap (lift)": f"{val_ap:.4f} ({(val_ap/event_rate):.2f})",
        "test_ap (lift)": f"{test_ap:.4f} ({(test_ap/event_rate):.2f})",
    }
    return res_dict


def get_perm_imp(model, X_val, y_val, feature_groups, n_repeats, seed):
    """
    Run grouper permutation importance on validation set

    Permute each variable group together (maintaining row-wise relationships)
    """
    rng = np.random.default_rng(seed)
    baseline_score = float(
        average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
    )

    X_perm = X_val.copy()
    records = []

    for group_name, group_cols in feature_groups.items():
        ## Skips cols not in df
        cols_present = [c for c in group_cols if c in X_val.columns]
        if not cols_present:
            continue

        og_vals = X_val[cols_present].values.copy()
        ap_drops = []

        for _ in range(n_repeats):
            perm_idx = rng.permutation(len(X_val))
            X_perm[cols_present] = og_vals[perm_idx]  # shuffle all together
            perm_ap = average_precision_score(y_val, model.predict_proba(X_perm)[:, 1])
            ap_drops.append(baseline_score - perm_ap)

        ## Restore permuted cols
        X_perm[cols_present] = og_vals

        records.append(
            {
                "Feature": group_name,
                "Mean": np.mean(ap_drops),
                "Stdev": np.std(ap_drops, ddof=1),
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values("Mean", ascending=False)
        .reset_index(drop=True)
    )


def get_drop_cols(perm_imp_df, perc_per_iter, feature_dict):
    """
    Removes bottom perc_per_iter% performing features
    """
    num_feats = len(perm_imp_df)
    num_to_remove = max(1, int(num_feats * perc_per_iter))
    groups_to_remove = perm_imp_df.iloc[-num_to_remove:]["Feature"].tolist()

    cols_to_remove = []
    for group in groups_to_remove:
        cols_to_remove += feature_dict[group]
    return cols_to_remove


def calibrate_model(X, y, n_splits, seed, model, n_cv_jobs):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(model), cv=skf, n_jobs=n_cv_jobs
    )
    calibrated_model.fit(X, y)
    return calibrated_model


def run_iter_feat_red(
    outcome,
    cohort,
    base_X_train,
    y_train,
    base_X_val,
    y_val,
    base_X_test,
    y_test,
    n_cv_folds,
    num_optuna_trials,
    num_perm_repeats,
    n_reduction_repeats,
    perc_per_iter,
    seed,
    n_cv_jobs,
    export_dir,
):
    feature_grpups = get_feat_groups(is_cancer=(cohort == "cancer"))
    X_df_train = base_X_train
    X_df_val = base_X_val
    X_df_test = base_X_test
    metric_list = []
    for reduction_iter in range(n_reduction_repeats):
        log(f"Reduction: ({reduction_iter+1} / {n_reduction_repeats})")
        ## Run tuner + fit model
        log("\tRunning tuner...")
        model, best_params, cv_score = tune_train_model(
            X_train=X_df_train,
            y_train=y_train,
            X_val=X_df_val,
            y_val=y_val,
            n_trials=num_optuna_trials,
            n_cv_folds=n_cv_folds,
            seed=seed,
            n_jobs=n_cv_jobs,
            study_name=f"{outcome}_{reduction_iter}",
        )
        log(f"\t\t Model best_iteration: {model.best_iteration_}")
        ## Calibrate this model
        log("\tCalibrating...")
        cal_model = calibrate_model(
            X=X_df_val,
            y=y_val,
            n_splits=n_cv_folds,
            seed=seed,
            model=model,
            n_cv_jobs=n_cv_jobs,
        )
        ## Evaluate calibrated model
        log("\tGetting metrics...")
        metric_dict = eval_model(
            X_train=X_df_train,
            X_val=X_df_val,
            X_test=X_df_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            model=cal_model,
            export_dir=export_dir / f"bins/iter_{reduction_iter}",
        )
        metric_dict["reduction"] = reduction_iter
        metric_list.append(metric_dict)
        log("\tGetting perm importances...")
        ## Get perm importance
        perm_imp_df = get_perm_imp(
            model,
            X_df_val,
            y_val,
            feature_grpups,
            n_repeats=num_perm_repeats,
            seed=seed,
        )
        export_data(
            data_to_export=perm_imp_df,
            export_path=export_dir / f"perm_df/iter_{reduction_iter}.tsv",
        )
        ## Remove feats
        drop_cols = get_drop_cols(
            perm_imp_df=perm_imp_df,
            perc_per_iter=perc_per_iter,
            feature_dict=feature_grpups,
        )
        X_df_train = X_df_train.drop(drop_cols, axis=1)
        X_df_val = X_df_val.drop(drop_cols, axis=1)
        X_df_test = X_df_test.drop(drop_cols, axis=1)

    result_df = pd.DataFrame(metric_list)
    export_data(data_to_export=result_df, export_path=export_dir / "metrics.tsv")
    return result_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track iterative feature reduction for a single (cohort, outcome)"
    )
    parser.add_argument("--outcome", type=str)
    parser.add_argument("--cohort", type=str)
    parser.add_argument("--imp_dir", type=Path)
    parser.add_argument("--export_dir", type=Path)
    parser.add_argument("--n_cv_folds", type=int)
    parser.add_argument("--num_optuna_trials", type=int)
    parser.add_argument("--num_perm_repeats", type=int)
    parser.add_argument("--n_reduction_repeats", type=int)
    parser.add_argument("--perc_per_iter", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n_cv_jobs", type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    log("Importing data...")
    ## Load data
    data_dict = get_data(args.outcome, file_dir=args.imp_dir)
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"].values.ravel()
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"].values.ravel()
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"].values.ravel()

    ### Iterative feat reduction on TRAIN set
    log(f"Starting iteration for outcome: {args.outcome}")
    perm_res_df = run_iter_feat_red(
        outcome=args.outcome,
        cohort=args.cohort,
        base_X_train=X_train,
        y_train=y_train,
        base_X_val=X_val,
        y_val=y_val,
        base_X_test=X_test,
        y_test=y_test,
        n_cv_folds=args.n_cv_folds,
        num_optuna_trials=args.num_optuna_trials,
        num_perm_repeats=args.num_perm_repeats,
        n_reduction_repeats=args.n_reduction_repeats,
        perc_per_iter=args.perc_per_iter,
        seed=args.seed,
        n_cv_jobs=args.n_cv_jobs,
        export_dir=args.export_dir,
    )
    log("DONE!")


if __name__ == "__main__":
    main()

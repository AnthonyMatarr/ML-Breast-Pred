## tune.py
import joblib
from joblib import Parallel, delayed
import argparse
from pathlib import Path
from datetime import datetime
import json


from src.data_utils import get_data, log
from src.nn_model import TorchNNClassifier
from src.eval import (
    get_logspace_thresholds,
    get_bin_metrics,
    get_discrimination_str,
)
from src.feat_eng import calibrate_model

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score

import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
import warnings

EARLY_STOP_ROUNDS = 50
BOOSTING_MAX_ROUNDS = 1000


def extract_metrics_from_log(log_path, model_name):
    text = Path(log_path).read_text()
    if len(text) < 1500:
        return {
            "outcome": "-----",
            "event_rate": "-----",
            "CV AP score": "-----",
            "bin_Very Low": "-----",
            "bin_Low": "-----",
            "bin_Moderate": "-----",
            "bin_High": "-----",
            "train_roc": "-----",
            "val_roc": "-----",
            "train_ap": "-----",
            "val_ap": "-----",
            "model_name": "-----",
        }
    # Find the start of each JSON object and decode the second one
    first_start = text.index("{")
    data_first, first_end = json.JSONDecoder().raw_decode(text, first_start)
    second_start = text.index("{", first_end)
    data_second, _ = json.JSONDecoder().raw_decode(text, second_start)

    output_dict = {
        f"bin_{bin_name}": round(float(data_second["bins"][bin_name]["lift"]), 2)
        for bin_name in ["Very Low", "Low", "Moderate", "High"]
    }
    return {
        "outcome": data_first["outcome_name"],
        "model_name": model_name,
        "event_rate": f"{data_second['event_rate']:.2%}",
        "CV AP score": round(float(data_first["best_score"]), 4),
        **output_dict,
        "train_roc": data_second["train_roc"],
        "val_roc": data_second["val_roc"],
        "train_ap": data_second["train_ap"],
        "val_ap": data_second["val_ap"],
    }


def lightgbm_model_builder(trial, seed):
    # Tree structure
    max_depth = trial.suggest_int("max_depth", 3, 12)
    num_leaves = trial.suggest_int("num_leaves", 4, 64)

    # Learning
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.5, log=True)
    min_split_gain = trial.suggest_float("min_split_gain", 0.0, 5.0)
    min_child_weight = trial.suggest_float("min_child_weight", 1e-4, 10.0, log=True)
    min_child_samples = trial.suggest_int("min_child_samples", 10, 150)

    # Sampling
    feature_fraction = trial.suggest_float("feature_fraction", 0.4, 1.0)

    bagging_freq = trial.suggest_int("bagging_freq", 1, 15)
    feature_fraction_bynode = trial.suggest_float("feature_fraction_bynode", 0.5, 1.0)

    imbalance_strategy = trial.suggest_categorical(
        "imbalance_strategy", ["weight_only", "neg_bagging_only", "both"]
    )

    pos_bagging_fraction = 1.0
    if imbalance_strategy == "weight_only":
        neg_bagging_fraction = 1.0
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 50.0, log=True)
    elif imbalance_strategy == "neg_bagging_only":
        neg_bagging_fraction = trial.suggest_float("neg_bagging_fraction", 0.1, 1.0)
        scale_pos_weight = 1.0
    else:
        neg_bagging_fraction = trial.suggest_float("neg_bagging_fraction", 0.1, 1.0)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 50.0, log=True)

    # Regularization
    lambda_l1 = trial.suggest_float("lambda_l1", 1e-8, 50.0, log=True)
    lambda_l2 = trial.suggest_float("lambda_l2", 1e-8, 50.0, log=True)
    max_bin = trial.suggest_int("max_bin", 64, 512)

    return lgb.LGBMClassifier(
        objective="binary",
        # n_estimators is a ceiling only; early stopping determines actual tree count
        n_estimators=BOOSTING_MAX_ROUNDS,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_weight=min_child_weight,
        min_split_gain=min_split_gain,
        min_child_samples=min_child_samples,
        feature_fraction=feature_fraction,
        pos_bagging_fraction=pos_bagging_fraction,
        neg_bagging_fraction=neg_bagging_fraction,
        bagging_freq=bagging_freq,
        feature_fraction_bynode=feature_fraction_bynode,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        max_bin=max_bin,
        scale_pos_weight=scale_pos_weight,
        tree_learner="serial",
        n_jobs=1,
        seed=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        deterministic=True,
        force_row_wise=True,
        metric="average_precision",
        verbosity=-1,
    )


def lr_model_builder(trial, seed):
    C = trial.suggest_float("C", 1e-2, 10.0, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    pos_weight = trial.suggest_float(
        "pos_weight", 1.0, 10.0, log=True
    )  # high pos weight drives up time to convergence
    class_weight = {0: 1.0, 1: pos_weight}

    return LogisticRegression(
        l1_ratio=l1_ratio,
        C=C,
        tol=1e-3,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=class_weight,
        random_state=seed,
        solver="saga",
        max_iter=2000,
        warm_start=False,
    )


def xgboost_model_builder(trial, seed):
    # Tree structure
    max_depth = trial.suggest_int("max_depth", 3, 12)
    max_bin = trial.suggest_categorical("max_bin", [64, 128, 256, 512])
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if grow_policy == "lossguide":
        max_leaves = trial.suggest_int("max_leaves", 16, 512, log=True)
    else:
        max_leaves = 0

    # Learning
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
    min_child_weight = trial.suggest_float("min_child_weight", 1e-2, 100.0, log=True)
    gamma = trial.suggest_float("gamma", 0.0, 5.0)
    max_delta_step = trial.suggest_float("max_delta_step", 0.0, 10.0)

    # Regularization
    reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True)

    # Sampling — colsample_bylevel freed from fixed 0.85
    subsample = trial.suggest_float("subsample", 0.4, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0)
    colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.4, 1.0)

    # Class imbalance
    scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 100.0, log=True)

    return xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=BOOSTING_MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        grow_policy=grow_policy,
        learning_rate=learning_rate,
        max_bin=max_bin,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        max_leaves=max_leaves,
        max_delta_step=max_delta_step,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def nn_model_builder(trial, seed):
    n_layers = trial.suggest_int("n_layers", 1, 2)

    hl_1 = trial.suggest_int("hl_1", 16, 256)
    dr_1 = trial.suggest_float("dr_1", 1e-3, 0.6)
    h_sizes = [hl_1]
    dropouts = [dr_1]

    if n_layers >= 2:
        hl_2 = trial.suggest_int("hl_2", 16, 256)
        dr_2 = trial.suggest_float("dr_2", 1e-3, 0.6)
        h_sizes.append(hl_2)
        dropouts.append(dr_2)

    act_name = trial.suggest_categorical("act_func_str", ["relu", "leaky_relu"])
    neg_slope = (
        trial.suggest_float("neg_slope", 1e-3, 0.4, log=True)
        if act_name == "leaky_relu"
        else 0.01
    )
    optimizer_str = "adamw"
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    bs = trial.suggest_categorical("batch_size", [128, 256, 512])
    es_patience = trial.suggest_int("es_patience", 15, 40)
    pos_weight = trial.suggest_float("pos_weight", 1.0, 50.0, log=True)

    return TorchNNClassifier(
        hidden_size_list=h_sizes,
        dropouts=dropouts,
        activation_name=act_name,
        neg_slope=neg_slope,
        lr=lr,
        weight_decay=wd,
        epochs=500,  # ceiling; early stopping determines actual count
        batch_size=bs,
        optimizer_str=optimizer_str,
        pos_weight=pos_weight,
        device="cpu",
        seed=seed,
        ## Early stopping
        early_stopping=True,
        es_patience=es_patience,
        es_min_delta=0.005,  # must improve by 0.5%
        # val_split=0.15, #dont need this if passing in eval_set
        monitor="auprc",  # not used for backprob, only for es
    )


MODEL_CONFIG = {
    "lgbm": {
        "builder": lightgbm_model_builder,
        "early_stopping": True,
        "fit_kwargs": {
            "eval_metric": "average_precision",
            "callbacks": [
                lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        },
        "best_iter_attr": "best_iteration_",  # trailing underscore — sklearn convention
        "n_pruning_folds": 0,  # all in parallel
    },
    "xgb": {
        "builder": xgboost_model_builder,
        "early_stopping": True,
        # early_stopping_rounds already in constructor
        "fit_kwargs": {"verbose": False},
        "best_iter_attr": "best_iteration",  # no trailing underscore
        "n_pruning_folds": 0,  # all in parallel
    },
    "lr": {
        "builder": lr_model_builder,
        "early_stopping": False,
    },
    "nn": {
        "builder": nn_model_builder,
        "early_stopping": True,
        "fit_kwargs": {},  # no additional kwargs
        "best_iter_attr": "best_iteration_",  # custom convention
        "n_pruning_folds": 0,  # all in parallel
    },
}


def _fold_worker(
    model_builder,
    params,
    seed,
    tr_idx,
    val_idx,
    X_train,
    y_train,
    fit_kwargs,
    best_iter_attr,
):
    """
    Executes one CV fold in a worker process.
    Rebuilds the model from params via FixedTrial so each worker
    has an independent copy with no shared state.
    """
    from optuna.trial import FixedTrial

    y_val = y_train[val_idx]
    if y_val.sum() == 0:
        return None, None

    model = model_builder(FixedTrial(params), seed)
    X_tr = X_train.iloc[tr_idx]
    y_tr = y_train[tr_idx]
    X_val = X_train.iloc[val_idx]

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], **fit_kwargs)

    score = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
    best_iter = getattr(model, best_iter_attr, None)
    return float(score), best_iter


def make_objective(
    X_train,
    y_train,
    model_builder,
    scoring,
    early_stopping,
    cv,
    seed,
    n_parallel_cv,
    fit_kwargs=None,
    best_iter_attr=None,
    n_pruning_folds=None,
):
    splits = list(cv.split(np.zeros(len(y_train)), y_train))

    def objective(trial):
        model = model_builder(trial, seed)

        if not early_stopping:
            ## For LR
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                try:
                    scores = cross_val_score(
                        model,
                        X_train,
                        y_train,
                        scoring=scoring,
                        cv=cv,
                        n_jobs=n_parallel_cv,
                        error_score=np.nan,
                    )
                except ValueError:  # when all folds fail
                    trial.set_user_attr("converged", False)
                    return float(y_train.mean())
                if np.isnan(scores).any():  # when
                    trial.set_user_attr("converged", False)
                    return float(y_train.mean())
            trial.set_user_attr("cv_std", float(np.std(scores)))
            trial.set_user_attr("converged", True)
            return float(np.mean(scores))

        # Early-stopping: manual fold loop so eval_set can be threaded through
        if fit_kwargs is None or best_iter_attr is None:
            raise ValueError(
                "fit_kwargs and best_iter_attr must be set in MODEL_CONFIG for early-stopping models"
            )

        params = dict(trial.params)  # used by parallel workers
        if n_pruning_folds is None:  # run all sequentially
            seq_splits = splits
            par_splits = []
        else:
            seq_splits = splits[:n_pruning_folds]
            par_splits = splits[n_pruning_folds:]

        fold_scores, fold_iters = [], []

        ### SEQUENTIAL ###
        for fold_idx, (tr_idx, val_idx) in enumerate(seq_splits):
            score, best_iter = _fold_worker(
                model_builder,
                params,
                seed,
                tr_idx,
                val_idx,
                X_train,
                y_train,
                fit_kwargs,
                best_iter_attr,
            )
            if score is None:
                continue
            fold_scores.append(score)
            fold_iters.append(best_iter)
            trial.report(score, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        ### PARALLEL ###
        if par_splits:
            results = Parallel(n_jobs=n_parallel_cv)(
                delayed(_fold_worker)(
                    model_builder,
                    params,
                    seed,
                    tr_idx,
                    val_idx,
                    X_train,
                    y_train,
                    fit_kwargs,
                    best_iter_attr,
                )
                for tr_idx, val_idx in par_splits
            )
            for score, best_iter in results:
                if score is not None:
                    fold_scores.append(score)
                    fold_iters.append(best_iter)

        if not fold_scores:  # just for safety if all trials pruned
            raise optuna.TrialPruned()
        trial.set_user_attr("mean_best_iter", int(np.mean(fold_iters)))
        trial.set_user_attr("cv_std", float(np.std(fold_scores)))
        return float(np.mean(fold_scores))

    return objective


def tune_single_model_outcome(
    *_,
    model_builder,
    model_abrv,
    outcome_name,
    X_train,
    y_train,
    scoring,
    n_trials,
    early_stopping,
    n_cv_splits,
    n_repeats,
    seed,
    n_parallel_cv,
    fit_kwargs,
    best_iter_attr,
    n_pruning_folds,
):
    if _ != tuple():
        raise ValueError("This function does not accept positional arguments")

    ## n_cv_splits-fold CV repeated n_repeats times
    cv = RepeatedStratifiedKFold(
        n_splits=n_cv_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )
    # NN and LR don't have early stopping- no intermediate metric reports for pruner
    pruner = (
        optuna.pruners.MedianPruner(n_warmup_steps=n_cv_splits)
        if early_stopping
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        study_name=f"{model_abrv}_{outcome_name}_study",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True),
        pruner=pruner,
    )
    if model_abrv == "lr":
        # force into 1 thread, to maintain convergence warning catch
        n_parallel_cv = 1
    study.optimize(
        make_objective(
            X_train,
            y_train,
            model_builder,
            scoring,
            early_stopping,
            cv,
            seed,
            n_parallel_cv,
            fit_kwargs=fit_kwargs,
            best_iter_attr=best_iter_attr,
            n_pruning_folds=n_pruning_folds,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_trial = study.best_trial
    result_dict = {
        "model_abrv": model_abrv,
        "outcome_name": outcome_name,
        "scoring": scoring,
        "best_score": best_trial.value,
        # use avg stopping point for global param set
        "mean_best_iter": best_trial.user_attrs.get("mean_best_iter"),
        "best_params": best_trial.params,
    }
    return result_dict


def train_final_model(
    *_,
    model_abrv,
    model_builder,
    best_params,
    X_train,
    y_train,
    mean_best_iter,
    num_cv_splits,
    seed,
):
    if _ != tuple():
        raise ValueError("This function does not take positional arguments!")
    fixed_trial = optuna.trial.FixedTrial(best_params)
    model = model_builder(fixed_trial, seed)

    if model_abrv in ["lgbm", "xgb", "nn"]:
        if mean_best_iter is None:
            raise ValueError(f"mean_best_iter required for {model_abrv}")

        # Scale up by (K+1)/K to account for the larger training set in the
        # final fit vs. each CV fold during tuning
        n_iter = int(mean_best_iter * ((num_cv_splits + 1) / num_cv_splits))

        if model_abrv == "lgbm":
            model.set_params(n_estimators=n_iter)
        elif model_abrv == "xgb":
            model.set_params(
                n_estimators=n_iter,
                early_stopping_rounds=None,  # remove early stopping bc requires val set
            )
        else:  # nn
            # run full n_iter epochs w/o stopping
            log(f"Num iter: {n_iter}")
            model.set_params(epochs=n_iter, early_stopping=False)
            log(f"Num epochs: {model.epochs}")
        model.fit(X_train, y_train)
    elif model_abrv == "lr":
        for multiplier in [1, 2]:
            model.set_params(max_iter=model.max_iter * multiplier)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                try:
                    model.fit(X_train, y_train)
                    break  # converged if we reach here
                except ConvergenceWarning:
                    if multiplier == 1:
                        log(
                            f"LR did NOT converge (max_iter={model.max_iter}), retrying w/ max_iter={model.max_iter * 2}"
                        )
                    else:
                        log(
                            f"WARNING: LR did NOT converge after retry (max_iter={model.max_iter}). Returning non-converged model"
                        )
    else:
        raise ValueError(f"Unrecognized model_abrv: {model_abrv}")
    return model


def get_prelim_results(data_dict, trained_model, eval_bootstraps):

    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"].values.ravel()
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"].values.ravel()

    y_proba_train = trained_model.predict_proba(X_train)[:, 1]
    y_proba_val = trained_model.predict_proba(X_val)[:, 1]

    bin_thresholds = get_logspace_thresholds(y_proba_train)
    bin_metrics = get_bin_metrics(
        y_true=y_val,
        y_proba=y_proba_val,
        thresholds=bin_thresholds,
        bin_report_dict={},
        n_bootstraps=eval_bootstraps,
    )
    train_roc_str = get_discrimination_str(
        y_true=y_train,
        y_proba=y_proba_train,
        metric_str="roc_auc",
        threshold=None,
        n_bootstraps=eval_bootstraps,
        bin_thresholds=None,
    )
    val_roc_str = get_discrimination_str(
        y_true=y_val,
        y_proba=y_proba_val,
        metric_str="roc_auc",
        threshold=None,
        n_bootstraps=eval_bootstraps,
        bin_thresholds=None,
    )
    train_ap_str = get_discrimination_str(
        y_true=y_train,
        y_proba=y_proba_train,
        metric_str="average_precision",
        threshold=None,
        n_bootstraps=eval_bootstraps,
        bin_thresholds=None,
    )
    val_ap_str = get_discrimination_str(
        y_true=y_val,
        y_proba=y_proba_val,
        metric_str="average_precision",
        threshold=None,
        n_bootstraps=eval_bootstraps,
        bin_thresholds=None,
    )
    result_dict = {
        "event_rate": float(np.mean(y_val)),
        "bins": bin_metrics,
        "train_roc": train_roc_str,
        "val_roc": val_roc_str,
        "train_ap": train_ap_str,
        "val_ap": val_ap_str,
    }
    return result_dict


def save_model(model, model_abrv, save_dir, best_params=None, calibrated=False):
    save_dir.mkdir(exist_ok=True, parents=True)
    if model_abrv == "nn" and not calibrated:
        import torch

        path = save_dir / "nn.pt"
        checkpoint = {
            "h_params": best_params,
            "state_dict": model.model_.state_dict(),
            "feature_names_in_": model.feature_names_in_,
            "epochs": model.epochs,
        }
        torch.save(checkpoint, path)
    else:
        path = save_dir / f"{model_abrv}.joblib"
        joblib.dump(model, path)
    assert path.exists()
    log(f"Model saved to {path}")
    return path


def build_parser():
    parser = argparse.ArgumentParser(prog="tuner")
    parser.add_argument(
        "--model_abrv", required=True, type=str, choices=["nn", "lr", "xgb", "lgbm"]
    )
    parser.add_argument("--outcome_name", required=True, type=str)
    parser.add_argument("--scoring", required=True, type=str)
    parser.add_argument("--import_dir", required=True, type=Path)
    parser.add_argument("--model_save_dir", required=False, default=None, type=Path)
    parser.add_argument("--n_trials", required=False, default=250, type=int)
    parser.add_argument("--n_cv_splits", required=False, default=3, type=int)
    parser.add_argument("--n_repeats", required=False, default=5, type=int)
    parser.add_argument("--n_parallel_cv", required=False, default=-1, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--eval_bootstraps", required=True, type=int)
    return parser


def main(argv=None):
    log("Parsing args..")
    args = build_parser().parse_args(argv)
    config = MODEL_CONFIG[args.model_abrv]

    log("Loading data...")
    data_dict = get_data(args.outcome_name, file_dir=args.import_dir)
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"].values.ravel()

    log(
        f"Tuning {args.model_abrv} for '{args.outcome_name}' ({args.n_trials} trials)..."
    )
    result_dict = tune_single_model_outcome(
        model_builder=config["builder"],
        model_abrv=args.model_abrv,
        outcome_name=args.outcome_name,
        X_train=X_train,
        y_train=y_train,
        scoring=args.scoring,
        n_trials=args.n_trials,
        early_stopping=config["early_stopping"],
        n_cv_splits=args.n_cv_splits,
        n_repeats=args.n_repeats,
        seed=args.seed,
        n_parallel_cv=args.n_parallel_cv,
        fit_kwargs=config.get("fit_kwargs"),
        best_iter_attr=config.get("best_iter_attr"),
        n_pruning_folds=config.get("n_pruning_folds"),
    )

    log(f"Tuning results: \n\t{json.dumps(result_dict, indent=4)}")

    log("Training model with selected params...")
    trained_model = train_final_model(
        model_abrv=args.model_abrv,
        model_builder=config["builder"],
        best_params=result_dict["best_params"],
        X_train=X_train,
        y_train=y_train,
        mean_best_iter=result_dict["mean_best_iter"],
        num_cv_splits=args.n_cv_splits,
        seed=args.seed,
    )
    log("Model trained, getting prelim results...")
    eval_dict = get_prelim_results(
        data_dict=data_dict,
        trained_model=trained_model,
        eval_bootstraps=args.eval_bootstraps,
    )
    log(f"Prelim results: \n {json.dumps(eval_dict, indent=4)}")
    if args.model_save_dir is not None:
        log("Calibrating model...")
        X_val = data_dict["X_val"]
        y_val = data_dict["y_val"].values.ravel()
        cal_model = calibrate_model(
            X=X_val,
            y=y_val,
            n_splits=args.n_cv_splits,
            seed=args.seed,
            model=trained_model,
            n_cv_jobs=args.n_cv_splits,
        )

        log("Saving trained and calibrated models...")
        save_model(
            model=trained_model,
            model_abrv=args.model_abrv,
            best_params=result_dict["best_params"],
            save_dir=args.model_save_dir / "trained" / args.outcome_name,
        )
        save_model(
            model=cal_model,
            model_abrv=args.model_abrv,
            best_params=result_dict["best_params"],
            save_dir=args.model_save_dir / "calibrated" / args.outcome_name,
            calibrated=True,
        )
    else:
        log("model_save_dir is None, not saving or calibrating model...")
    log("DONE!")


if __name__ == "__main__":
    main()

from src.config import SEED

## General purpose
import joblib
import threading
import json
import warnings
import logging
import numpy as np
import optuna
from func_timeout import func_timeout, FunctionTimedOut
from sklearn.exceptions import ConvergenceWarning
import time
from sklearn.model_selection import train_test_split

## Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.base import ClassifierMixin, BaseEstimator

## Sampling
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from src.data_utils import get_feature_lists

## Focal Loss
import warnings

warnings.filterwarnings("ignore", message="overflow encountered in exp")
warnings.filterwarnings("ignore", message="Cannot compute class probabilities")
## GLOBALS

MODEL_N_JOBS = {
    "lr": 1,
    "lgbm": 1,
    "xgb": 1,
}
LOG_LOCK = threading.Lock()
STAGE = 2
if STAGE == 1:
    N_SPLITS = 3
elif STAGE == 2:
    N_SPLITS = 5
else:
    raise ValueError("Stage must be one of [1,2]")

N_PARALLEL_CV_JOBS = N_SPLITS


#############################################################################################
######################################## Focal Loss  ########################################
#############################################################################################
#### Models performed well in stage 1 w/o this--> leaving out
## keep for potential future use
class FocalObjective:
    """Picklable focal loss objective"""

    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        # Clip to prevent overflow
        y_pred = np.clip(y_pred, -50, 50)

        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        y = y_true.astype(float)

        pt = np.where(y == 1, p, 1 - p)
        focal_weight = np.power(1 - pt, self.gamma)
        alpha_t = np.where(y == 1, self.alpha, 1 - self.alpha)

        grad = alpha_t * focal_weight * (self.gamma * pt * np.log(pt + 1e-8) + (p - y))
        hess = alpha_t * focal_weight * p * (1 - p) * (self.gamma * (2 * pt - 1) + 1)

        return grad, hess


def custom_auc_scorer(y_true, y_pred, **kwargs):
    """
    Custom AUROC scorer that handles both:
    - 2D arrays (standard predict_proba output)
    - 1D arrays (custom objective raw scores from LightGBM/XGBoost)

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predictions (can be 1D raw scores or 2D probabilities)

    Returns
    -------
    float
        AUROC score

    Notes
    -----
    - Not used for either stage 1 or 2
    - Kept just for legacy and potential future use
    """
    # If 2D array (standard predict_proba), take positive class column
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, 1]
    # If 1D array (raw scores from custom objective), use directly

    return roc_auc_score(y_true, y_pred)


# CUSTOM_AUC_SCORER = make_scorer(custom_auc_scorer, needs_proba=True)


#############################################################################################
###################################### MODEL BUILDERS #######################################
#############################################################################################
# ========================> LOGISTIC REGRESSION
def lr_model_builder_stage1(trial):
    # Regularization strength (allow weaker and stronger)
    C = trial.suggest_float("C", 1e-4, 10.0, log=True)

    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])

    if penalty == "elasticnet":
        solver = "saga"
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    elif penalty == "l1":
        solver = trial.suggest_categorical("solver_l1", ["liblinear", "saga"])
        l1_ratio = None
    else:  # l2
        solver = trial.suggest_categorical(
            "solver_l2", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
        )
        l1_ratio = None

    # Moderately constrained pos_weight
    if solver in ["saga", "sag"]:
        pos_weight = trial.suggest_float("pos_weight_saga", 1.0, 8.0, log=True)
    elif solver == "liblinear":
        pos_weight = trial.suggest_float("pos_weight_linear", 1.0, 12.0, log=True)
    else:
        pos_weight = trial.suggest_float("pos_weight_general", 1.0, 8.0, log=True)

    class_weight = {0: 1.0, 1: pos_weight}

    if solver == "liblinear":
        intercept_scaling = trial.suggest_float(
            "intercept_scaling", 1e-2, 1e2, log=True
        )
        n_jobs = 1
    else:
        intercept_scaling = 1.0
        n_jobs = MODEL_N_JOBS["lr"]

    max_iter = 5000 if solver in ["saga", "sag"] else 4000

    return LogisticRegression(
        penalty=penalty,
        C=C,
        tol=1e-4,
        fit_intercept=True,
        intercept_scaling=intercept_scaling,
        class_weight=class_weight,
        random_state=SEED,
        solver=solver,
        max_iter=max_iter,
        l1_ratio=l1_ratio,
        warm_start=False,
        n_jobs=n_jobs,
    )


def lr_model_builder(trial, outcome_name):
    if outcome_name == "med":
        C = trial.suggest_float("C", 0.01, 0.15, log=True)
        penalty = "l1"
        solver = "liblinear"
        pos_weight = trial.suggest_float("pos_weight_linear", 8.0, 12.0, log=True)
        l1_ratio = None
    elif outcome_name == "mort":
        C = trial.suggest_float("C", 0.06, 0.25, log=True)
        penalty = "l1"
        solver = "saga"
        pos_weight = trial.suggest_float("pos_weight_saga", 4.5, 7.5, log=True)
        l1_ratio = None
    elif outcome_name == "reop":
        C = trial.suggest_float("C", 0.015, 0.2, log=True)
        penalty = "l1"
        solver = "liblinear"
        pos_weight = trial.suggest_float("pos_weight_linear", 6.0, 12.0, log=True)
        l1_ratio = None
    elif outcome_name in ["surg", "vte"]:  # no clear pattern here
        # Regularization strength (allow weaker and stronger)
        C = trial.suggest_float("C", 1e-4, 10.0, log=True)

        penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])

        if penalty == "elasticnet":
            solver = "saga"
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        elif penalty == "l1":
            solver = trial.suggest_categorical("solver_l1", ["liblinear", "saga"])
            l1_ratio = None
        else:  # l2
            solver = trial.suggest_categorical(
                "solver_l2", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
            )
            l1_ratio = None

        # Moderately constrained pos_weight
        if solver in ["saga", "sag"]:
            pos_weight = trial.suggest_float("pos_weight_saga", 1.0, 8.0, log=True)
        elif solver == "liblinear":
            pos_weight = trial.suggest_float("pos_weight_linear", 1.0, 12.0, log=True)
        else:
            pos_weight = trial.suggest_float("pos_weight_general", 1.0, 8.0, log=True)
    else:
        raise ValueError(
            f"model_name must be one of ['med', 'mort', 'reop', 'surg', 'vte'], got {outcome_name} instead!"
        )
    class_weight = {0: 1.0, 1: pos_weight}

    if solver == "liblinear":
        intercept_scaling = trial.suggest_float(
            "intercept_scaling", 1e-2, 1e2, log=True
        )
        n_jobs = 1
    else:
        intercept_scaling = 1.0
        n_jobs = MODEL_N_JOBS["lr"]

    max_iter = 5000 if solver in ["saga", "sag"] else 4000

    return LogisticRegression(
        penalty=penalty,
        C=C,
        tol=1e-4,
        fit_intercept=True,
        intercept_scaling=intercept_scaling,
        class_weight=class_weight,
        random_state=SEED,
        solver=solver,
        max_iter=max_iter,
        l1_ratio=l1_ratio,
        warm_start=False,
        n_jobs=n_jobs,
    )


# ========================> LIGHTGBM
def lightgbm_model_builder_stage1(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.005, 0.1, log=True)
    n_estimators = trial.suggest_int("n_estimators", 200, 1200)

    max_depth = trial.suggest_int("max_depth", 3, 12)  # allow shallower trees
    max_leaves = min(512, 2**max_depth)
    min_leaves = min(16, max_leaves)
    num_leaves = trial.suggest_int("num_leaves", min_leaves, max_leaves)

    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 20, 200)
    min_gain_to_split = trial.suggest_float("min_gain_to_split", 0.0, 3.0)

    feature_fraction = trial.suggest_float("feature_fraction", 0.6, 1.0)
    bagging_freq = trial.suggest_int("bagging_freq", 1, 10)
    # let scale_pos_weight be moderate; ADASYN will also act if enabled
    scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 15.0, log=True)

    pos_bagging_fraction = trial.suggest_float("pos_bagging_fraction", 0.7, 1.0)
    neg_bagging_fraction = trial.suggest_float("neg_bagging_fraction", 0.2, 1.0)

    lambda_l1 = trial.suggest_float("lambda_l1", 0.0, 20.0)
    lambda_l2 = trial.suggest_float("lambda_l2", 0.0, 30.0)

    max_bin = trial.suggest_int("max_bin", 64, 300)

    return LGBMClassifier(
        objective="binary",  # no focal loss for stage 1
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        min_split_gain=min_gain_to_split,
        feature_fraction=feature_fraction,
        pos_bagging_fraction=pos_bagging_fraction,
        neg_bagging_fraction=neg_bagging_fraction,
        bagging_freq=bagging_freq,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        scale_pos_weight=scale_pos_weight,
        max_bin=max_bin,
        tree_learner="feature_parallel",
        n_jobs=MODEL_N_JOBS["lgbm"],
        seed=SEED,
        bagging_seed=SEED,
        feature_fraction_seed=SEED,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )


def lightgbm_model_builder(trial, outcome_name):
    if outcome_name == "med":
        learning_rate = trial.suggest_float("learning_rate", 0.006, 0.012, log=True)
        n_estimators = trial.suggest_int("n_estimators", 500, 850)
        max_depth = trial.suggest_int("max_depth", 3, 5)  # shallower trees
        lambda_l1 = trial.suggest_float("lambda_l1", 4.0, 16.0)  # high regularization
        lambda_l2 = trial.suggest_float("lambda_l2", 10.0, 24.0)  # high regularization
    elif outcome_name == "surg":
        learning_rate = trial.suggest_float("learning_rate", 0.008, 0.03, log=True)
        n_estimators = trial.suggest_int("n_estimators", 300, 700)
        max_depth = trial.suggest_int("max_depth", 3, 5)  # shallower trees
        lambda_l1 = trial.suggest_float("lambda_l1", 10.0, 20.0)  # high regularization
        lambda_l2 = trial.suggest_float("lambda_l2", 0.0, 30.0)  # same as rest
    elif outcome_name in ["mort", "reop", "vte"]:  # keep same as stage 1 space
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.1, log=True)
        n_estimators = trial.suggest_int("n_estimators", 200, 1200)
        max_depth = trial.suggest_int("max_depth", 3, 10)  # a bit shallower than before
        lambda_l1 = trial.suggest_float("lambda_l1", 0.0, 20.0)
        lambda_l2 = trial.suggest_float("lambda_l2", 0.0, 30.0)
    else:
        raise ValueError(
            f"model_name must be one of ['med', 'mort', 'reop', 'surg', 'vte'], got {outcome_name} instead!"
        )
    #### Shared accross all outcomes
    max_leaves = min(512, 2**max_depth)
    min_leaves = min(16, max_leaves)
    num_leaves = trial.suggest_int("num_leaves", min_leaves, max_leaves)

    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 20, 200)
    min_gain_to_split = trial.suggest_float("min_gain_to_split", 0.0, 3.0)

    feature_fraction = trial.suggest_float("feature_fraction", 0.6, 1.0)
    bagging_freq = trial.suggest_int("bagging_freq", 1, 10)
    # let scale_pos_weight be moderate; ADASYN will also act if enabled
    scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 15.0, log=True)

    pos_bagging_fraction = trial.suggest_float("pos_bagging_fraction", 0.7, 1.0)
    neg_bagging_fraction = trial.suggest_float("neg_bagging_fraction", 0.2, 1.0)

    max_bin = trial.suggest_int("max_bin", 64, 300)

    return LGBMClassifier(
        objective="binary",  # no focal loss for stage 1
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        min_split_gain=min_gain_to_split,
        feature_fraction=feature_fraction,
        pos_bagging_fraction=pos_bagging_fraction,
        neg_bagging_fraction=neg_bagging_fraction,
        bagging_freq=bagging_freq,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        scale_pos_weight=scale_pos_weight,
        max_bin=max_bin,
        tree_learner="feature_parallel",
        n_jobs=MODEL_N_JOBS["lgbm"],
        seed=SEED,
        bagging_seed=SEED,
        feature_fraction_seed=SEED,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )


# ========================> XGBOOST
def xgb_model_builder_stage1(trial):
    sampling_strategy = trial.params.get("sampling_strategy", "none")

    if sampling_strategy == "tomek":
        n_estimators = trial.suggest_int("n_estimators", 200, 700)
    else:
        n_estimators = trial.suggest_int("n_estimators", 300, 1200)

    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 8)

    gamma = trial.suggest_float("gamma", 0.0, 5.0)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 20.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 25.0)

    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    colsample_bylevel = 0.85  # keep fixed in stage 1

    scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 20.0, log=True)

    min_child_weight = trial.suggest_int("min_child_weight", 1, 20)

    return XGBClassifier(
        objective="binary:logistic",  # no focal loss for stage 1
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=min_child_weight,
        tree_method="hist",
        n_jobs=MODEL_N_JOBS["xgb"],
        random_state=SEED,
        eval_metric="auc",
    )


def xgb_model_builder(trial, outcome_name):
    if outcome_name == "surg":
        ## med depth
        max_depth = trial.suggest_int("max_depth", 5, 8)
        ## lower lr
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.025, log=True)
        # more estimators
        n_estimators = trial.suggest_int("n_estimators", 800, 1200)
        # higher pos weight
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 10.0, 15.0, log=True)
        # same
        min_child_weight = trial.suggest_int("min_child_weight", 1, 20)
    elif outcome_name == "med":
        ## med depth
        max_depth = trial.suggest_int("max_depth", 5, 8)
        ## lower lr
        learning_rate = trial.suggest_float("learning_rate", 0.013, 0.04, log=True)
        # lowerish estimators
        n_estimators = trial.suggest_int("n_estimators", 400, 700)
        # lower pos weight
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 5.0, log=True)
        # higher
        min_child_weight = trial.suggest_int("min_child_weight", 10, 20)
    elif outcome_name in ["mort", "reop", "vte"]:
        max_depth = trial.suggest_int("max_depth", 3, 8)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        n_estimators = trial.suggest_int("n_estimators", 300, 1200)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 20.0, log=True)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 20)
    else:
        raise ValueError(
            f"model_name must be one of ['med', 'mort', 'reop', 'surg', 'vte'], got {outcome_name} instead!"
        )

    gamma = trial.suggest_float("gamma", 0.0, 5.0)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 20.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 25.0)

    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    colsample_bylevel = 0.85  # keep fixed in stage 1

    return XGBClassifier(
        objective="binary:logistic",  # no focal loss for stage 1
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=min_child_weight,
        tree_method="hist",
        n_jobs=MODEL_N_JOBS["xgb"],
        random_state=SEED,
        eval_metric="auc",
    )


#############################################################################################
########################################## TUNING ###########################################
#############################################################################################
# Not using sampler for stage2 --> keep for legacy (for now)
def get_sampler(strategy, ratio, n_neighbors=None):
    """
    Helper function that returns sampler
    Parameters
    ----------
    strategy: string; One of ['smotenc', 'adasyn', 'tomek')
        Specify sampling strategy to use
    ratio: float
        Sampling ratio used within appropriate sampler
    """
    if strategy == "adasyn":
        if n_neighbors == None:
            return ValueError("Must pass n_neighbors for ADASYN")
        sampler = ADASYN(
            sampling_strategy=ratio,
            random_state=SEED,
            n_neighbors=n_neighbors,
        )
    elif strategy == "tomek":
        sampler = TomekLinks()
    else:
        raise ValueError(f"Got unknown sampling strategy: {strategy}")
    return sampler


def get_default_params(model_abrv):
    """
    Returns sensible default hyperparameters for each model type.
    Used as fallback if all Optuna trials are pruned.

    Parameters
    ----------
    model_abrv: str
        Model abbreviation: 'lr', 'lgbm', 'xgb','cat' or 'svc'

    Returns
    -------
    dict
        Dictionary of default hyperparameters for the model
    """
    if model_abrv == "lr":
        fallback_model = LogisticRegression(
            C=1.0,
            penalty="l2",
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
            max_iter=5000,
            n_jobs=MODEL_N_JOBS["lr"],
        )
        fallback_params = {
            "C": 1.0,
            "penalty": "l2",
            "class_weight": "balanced",
            "solver": "lbfgs",
        }
    elif model_abrv == "lgbm":
        fallback_model = LGBMClassifier(
            objective="binary",
            learning_rate=0.1,
            n_estimators=100,
            max_depth=-1,
            num_leaves=31,
            min_data_in_leaf=20,
            class_weight="balanced",
            n_jobs=MODEL_N_JOBS["lgbm"],
            seed=SEED,
            deterministic=True,
            force_row_wise=True,
            verbosity=-1,
        )
        fallback_params = {
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": -1,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
        }
    elif model_abrv == "xgb":
        fallback_model = XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.1,
            n_estimators=100,
            max_depth=6,
            scale_pos_weight=10.0,  # ~inverse of your 4% positive rate
            tree_method="hist",
            n_jobs=MODEL_N_JOBS["xgb"],
            random_state=SEED,
        )
        fallback_params = {
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 6,
            "scale_pos_weight": 10.0,
        }
    else:
        raise ValueError(f"Unknown model: {model_abrv}")
    return fallback_model, fallback_params


def make_objective(X_train, y_train, model_builder, outcome_name, scoring="roc_auc"):
    """
    Creates objective for model tuning
    Includes sampling param and calls appropriate model builder
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    def objective(trial):
        # sampling_strategy = trial.suggest_categorical(
        #     "sampling_strategy", ["adasyn", "none"]
        # )
        sampling_strategy = "none"  # fix strategy at none for stage 2
        if sampling_strategy == "adasyn":
            sampling_ratio = trial.suggest_float("sampling_ratio", 0.2, 0.5)
            n_neighbors = 5
        else:
            n_neighbors = None
            sampling_ratio = None
        # build model
        model = model_builder(trial, outcome_name)
        # create pipeline
        if sampling_strategy == "none":
            pipeline = model
        else:
            ## Create pipeline w/ sampler
            sampler = get_sampler(
                strategy=sampling_strategy,
                ratio=sampling_ratio,
                n_neighbors=n_neighbors,
            )
            pipeline = Pipeline([("sampler", sampler), ("classifier", model)])
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring=scoring,
            cv=skf,
            n_jobs=N_PARALLEL_CV_JOBS,
        )
        return np.round(np.mean(scores), 4)

    return objective


def tune_model_mult_outcomes(
    *_,
    model_builder,
    model_abrv,
    outcome_dict,
    scoring,
    log_file_path,
    save_path,
    n_parallel_trials,
    n_trials,
    timeout_per_trial=6000,
    clear_progress=False,
):
    """
    Tunes a given model for each given outcome using Optuna and writes params/tuning results to memory.
    Used for LR, LGBM, XGB

    Parameters
    ----------
    model_builder: callable
        Function that takes an optuna.trial object and returns a built estimator
    model_abrv: string
        Abbreviation of model to be tuned, one of [lr, svc, lgbm]
    outcome_dict: dict
        Dictionary mapping outcome names to relevant data
        Format:
            {
                outcome_type (str): {
                    'X_train': pandas dataframe,
                    'y_train':pandas dataframe
                    'X_val': pandas dataframe
                    'y_val': pandas dataframe
                    'X_test': pandas dataframe
                    'y_test': pandas dataframe
                }
            }
    scoring: str
        String specifying which scoring metric to use for tuning.
        Ultimately passed into sklearn.model_selection.cross_val_score().
    log_file_path: pathlib.Path
        Absolute path to file where tuning logs will be written to.
    save_path: pathlib.Path
        Absolute path to directory where best CV score/params are written to in json format.
    n_trials: Optional int; defaults to 500
    timeout_per_trial: Optional int; defaults to 6000
        Specify max number of seconds a trial can take before abandoning/pruning
        Set to 100 minutes, so essentially timeout pruning is off by default
    clear_progress: Boolean
        If True, check if there's any study progress in db file and delete
        If False, don't check (and therefore maintain any progress if exists)
    Returns
    -------
    Nothing. This function only writes to memory
    Raises
    ------
    ValueError:
        If positional arguments are given
    """
    if _ != tuple():
        raise ValueError("This func does not take positional args")

    ############ Set up paths ############
    ## Clear log files
    if log_file_path.exists():
        warnings.warn(f"Over-writing log at path: {log_file_path}")
        log_file_path.unlink()
    log_file_path.parent.mkdir(exist_ok=True, parents=True)
    ## Clear JSON path
    if save_path.exists():
        save_path.unlink()
    save_path.parent.mkdir(exist_ok=True, parents=True)

    ## Clear existing studies
    if clear_progress:
        for db_file in save_path.parent.glob(f"{model_abrv}_*.db"):
            db_file.unlink()
            with open(log_file_path, "a") as f:
                f.write(f"Deleted existing study database: {db_file.name}\n")

    ############ Set up logger ############
    root_logger = logging.getLogger()  # Root logger
    root_logger.setLevel(logging.INFO)

    # Remove all existing FileHandlers to prevent cross-contamination
    for handler in root_logger.handlers[:]:  # Iterate over copy
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()  # close file handle

    # Now add the handler for current model
    file_handler = logging.FileHandler(log_file_path, mode="a")
    root_logger.addHandler(file_handler)

    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    ############ Run for each outcome ############
    result_dict = {}
    for outcome_name, outcome_data in outcome_dict.items():
        with open(log_file_path, "a") as f:
            f.write(f"Working on outcome: {outcome_name}...\n")
        X_train = outcome_data["X_train"]
        y_train = outcome_data["y_train"].values.ravel()
        #### SUBSAMPLE

        ## Stage 1- 50%
        if STAGE == 1:
            X_dummy, X_sub, y_dummy, y_sub = train_test_split(
                X_train, y_train, stratify=y_train, test_size=0.5
            )
        ## Stage 2- 100%
        else:
            X_sub, y_sub = X_train, y_train

        ## Create objective
        base_objective = make_objective(
            X_sub, y_sub, model_builder, outcome_name, scoring
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            storage = (
                f"sqlite:///{save_path.parent / f'{model_abrv}_{outcome_name}.db'}"
            )
            study = optuna.create_study(
                storage=storage,
                study_name=f"{model_abrv}_{outcome_name}_study",
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
                pruner=optuna.pruners.HyperbandPruner(),
                load_if_exists=True,
            )

            ###### with timeout ######
            # Helper function for parallel trials with timeout wrapper
            def run_trials(n_trials_per_worker):
                study_local = optuna.load_study(
                    study_name=f"{model_abrv}_{outcome_name}_study",
                    storage=storage,
                )

                # Timeout wrapper
                def timeout_objective(trial):
                    try:
                        start = time.time()
                        result = func_timeout(
                            timeout_per_trial,
                            base_objective,
                            args=(trial,),
                        )
                        duration = time.time() - start
                        with LOG_LOCK:
                            with open(log_file_path, "a") as f:
                                f.write(
                                    f"Trial {trial.number} finished in: {duration:.1f}s \n"
                                )
                        return result
                    except FunctionTimedOut:
                        with LOG_LOCK:
                            with open(log_file_path, "a") as f:
                                f.write(
                                    f"Trial {trial.number} timed out after {timeout_per_trial}s --> Parameters: {trial.params}\n"
                                )
                        raise optuna.TrialPruned()

                # Timeout_objective instead of base_objective
                study_local.optimize(timeout_objective, n_trials=n_trials_per_worker)  # type: ignore

            # Parallelize trials using threading
            threads = []
            trials_per_worker = int(n_trials / n_parallel_trials)
            for _ in range(n_parallel_trials):
                thread = threading.Thread(target=run_trials, args=(trials_per_worker,))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()
            # Load final results
            study = optuna.load_study(
                study_name=f"{model_abrv}_{outcome_name}_study",
                storage=storage,
            )
            ####### Optuna timeout safety net #######
            try:
                best_score = study.best_value
                best_params = study.best_params
                ## Deal with sampling
                # sampling_strategy = best_params.get("sampling_strategy")
                # sampling_ratio = best_params.get("sampling_ratio")
                # if sampling_strategy != "none":
                #     ## resample
                #     # numerical_cols = get_feature_lists(X_sub)["numerical_cols"]
                #     # cat_cols = [
                #     #     i
                #     #     for i, col in enumerate(X_sub.columns)
                #     #     if col not in numerical_cols
                #     # ]
                #     sampler = get_sampler(
                #         strategy=sampling_strategy,
                #         ratio=sampling_ratio,
                #         n_neighbors=5,
                #     )
                #     _, y_train_resampled = sampler.fit_resample(X_sub, y_sub)  # type: ignore
                #     # ## log resample
                #     # with open(log_file_path, "a") as f:
                #     #     f.write(
                #     #         f"Best sampling: {sampling_strategy} "
                #     #         f"(ratio={sampling_ratio:.3f})\n"
                #     #         f"Original size: {len(y_train)}, "
                #     #         f"Resampled size: {len(y_train_resampled)}\n"
                #     #     )
                # else:
                #     y_train_resampled = y_train.copy()
                ## get results
                result_dict[outcome_name] = {
                    "best_score": best_score,
                    "best_params": best_params,
                    # "sampling_strategy": sampling_strategy,
                    # "sampling_ratio": sampling_ratio,
                    # "n_train_original": len(y_train),
                    # "n_train_resampled": len(y_train_resampled),
                    # "study": study, # NO need to save this, but including in case want to inspect later
                }

                with open(log_file_path, "a") as f:
                    f.write(
                        f"{outcome_name}: best {scoring}: {best_score:.4f}\nparams={best_params}\n"
                    )
                    f.write(f"{'*' * 100}\n")
            except ValueError:
                # All trials were pruned - use default parameters
                with open(log_file_path, "a") as f:
                    f.write(
                        f"WARNING: All trials for {outcome_name} were pruned. "
                        f"Using default hyperparameters.\n"
                    )
                    # get default model + params
                    fallback_model, fallback_params = get_default_params(model_abrv)
                    # Run 5-fold CV with default params
                    skf = StratifiedKFold(
                        n_splits=N_SPLITS, shuffle=True, random_state=SEED
                    )
                    fallback_scores = cross_val_score(
                        fallback_model,  # type: ignore
                        X_sub,
                        y_sub,
                        scoring=scoring,
                        cv=skf,
                        n_jobs=N_PARALLEL_CV_JOBS,
                    )
                    fallback_score = np.mean(fallback_scores)
                    result_dict[outcome_name] = {
                        "best_score": float(fallback_score),
                        "best_params": fallback_params,
                        "fallback_used": True,  # Used later for building model
                    }
                    with open(log_file_path, "a") as f:
                        f.write(
                            f"{outcome_name} (FALLBACK): {scoring}: {fallback_score:.4f}\n"
                            f"params={fallback_params}\n"
                        )
    # Remove handler when this model is done
    root_logger.removeHandler(file_handler)
    file_handler.close()

    with open(save_path, "w") as f:
        json.dump(result_dict, f, indent=4)

    return result_dict


#############################################################################################
###################################### PRELIM RESULTS #######################################
#############################################################################################
def get_prelim_results(
    *_,
    results_path,
    model_builder,
    model_abrv,
    outcome_dict,
    model_save_dir=None,
):
    """
    Calculates train + validation AUROC scores and prints them out.
    Also used to export models when model_save_dir is not None.

    Parameters
    ----------
    results_path: pathlib.Path
        Absolute path to json file containing dictionary mapping of outcome types to results
        Format:
        {
            outcome_type (str): {
                'best_score': float,
                'best_params': dict
            }
        }
    model_builder: callable
        Function that takes an optuna.trial object and returns a built estimator

    model_abrv: str
        Abbreviation of model used
    outcome_dict: dict
        Dictionary mapping outcome names to relevant data
        Format:
        {
            outcome_type (str): {
                'X_train': pandas dataframe,
                'y_train': pandas dataframe
                'X_val': pandas dataframe,
                'y_val': pandas dataframe
                'X_test': pandas dataframe,
                'y_test': pandas dataframe
            }
        }
    model_save_dir: pathlib.Path; defaults to None
        Directory to save models. If None, will not save models
    Returns
    --------
    Nothing, just prints out results and (optionally) saves models

    Raises
    --------
    ValueError:
        If positional arguments are given
    """
    if _ != tuple():
        raise ValueError("This function does not take position arguments!")

    print(f"{'-'*30} {model_abrv} {'-'*30}")
    with open(results_path, "r") as f:
        results_dict = json.load(f)
    for outcome, results in results_dict.items():
        ##Get tuning results
        best_score = results["best_score"]
        best_params = results["best_params"]
        # Get sampling info
        sampling_strategy = results.get("sampling_strategy", "none")
        sampling_ratio = results.get("sampling_ratio", None)
        # Check if fallback was used
        if results.get("fallback_used", False):
            model, _ = get_default_params(model_abrv)
        else:
            # Remove sampling params before passing to model builder
            model_params = {
                k: v
                for k, v in best_params.items()
                if k not in ["sampling_strategy", "sampling_ratio"]
            }
            trial = optuna.trial.FixedTrial(model_params)
            model = model_builder(trial, outcome)

        ##### Train model #####
        X_train = outcome_dict[outcome]["X_train"]
        y_train = outcome_dict[outcome]["y_train"].values.ravel()
        ## Resample if needed
        if sampling_strategy != "none":
            # numerical_cols = get_feature_lists(X_train)["numerical_cols"]
            # cat_cols = [
            #     i for i, col in enumerate(X_train.columns) if col not in numerical_cols
            # ]
            sampler = get_sampler(
                strategy=sampling_strategy, ratio=sampling_ratio, n_neighbors=5
            )
            X_train, y_train = sampler.fit_resample(X_train, y_train)  # type: ignore

        model.fit(X_train, y_train)  # type: ignore
        ## Pass into custom wrapper to get predict_proba functionality --> NOT used for stage1/2
        # if model_abrv == "lgbm":
        #     wrapped_model = LGBMWithCustomProbaWrapper(**model.get_params())
        #     wrapped_model.__dict__.update(model.__dict__)
        #     model = wrapped_model
        # elif model_abrv == "xgb":
        #     wrapped_model = XGBCompatibleClassifier(model)
        #     model = wrapped_model
        ##### Export model #####
        if model_save_dir:
            save_path = model_save_dir / outcome / f"{model_abrv}.joblib"
            if save_path.exists():
                save_path.unlink()
            save_path.parent.mkdir(exist_ok=True, parents=True)
            joblib.dump(model, save_path)

        ##### Get prelim results #####
        X_val = outcome_dict[outcome]["X_val"]
        y_val = outcome_dict[outcome]["y_val"]

        ## SVC
        if model_abrv == "svc":
            # not probabilities, but appropriately ranked
            train_pred_proba = model.decision_function(X_train)  # type: ignore
            val_pred_proba = model.decision_function(X_val)  # type: ignore
        ## All other models (lr, xgb, lightgbm)
        else:
            train_pred_proba = model.predict_proba(X_train)[:, 1]  # type: ignore
            val_pred_proba = model.predict_proba(X_val)[:, 1]  # type: ignore
        train_auc = roc_auc_score(y_train, train_pred_proba)  # type: ignore
        val_auc = roc_auc_score(y_val, val_pred_proba)  # type: ignore

        ##### Output results #####
        # Updated output
        print(f"Outcome: \t{outcome}")
        print(f"Best CV AUROC: \t{best_score:.3f}")
        if sampling_ratio is not None:
            print(f"Sampling: \t{sampling_strategy} (ratio={sampling_ratio:.3f})")
            print(
                f"Train size: \t{len(y_train)} (original: {results.get('n_train_original')})"
            )
        else:
            print(f"Sampling: \t{sampling_strategy} (ratio=N/A)")

        print(f"Train AUROC: \t{train_auc:.3f}")
        print(f"Val ROC AUC: \t{val_auc:.3f}")
        print(f"PARAMS: \t{best_params}")
        print("*" * 10)


#############################################################################################
############################## CUSTOM LGBM w/ 2D predict_proba ##############################
#############################################################################################
## Only need for focal loss, not using at this moment
class LGBMWithCustomProbaWrapper(LGBMClassifier):
    """
    Notes
    -----
    - Not used for stage 1/2
    - Kept for legacy and potential future use
    """

    _estimator_type = "classifier"

    # Try best to match OG LGBMClassifier init
    def __init__(
        self,
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        subsample_for_bin=200000,
        objective=None,
        class_weight=None,
        min_split_gain=0.0,
        min_child_weight=0.001,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=None,
        n_jobs=None,
        importance_type="split",
        **kwargs,  # kwargs can be trailing only
    ):
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            importance_type=importance_type,
            **kwargs,
        )
        self._estimator_type = "classifier"

    def predict_proba(self, X):
        proba = super().predict_proba(X)
        # If 1D (it is when using custom focal loss func), fix to 2D:
        if proba.ndim == 1:  # type: ignore
            return np.vstack([1 - proba, proba]).T  # type: ignore
        return proba


class XGBCompatibleClassifier(ClassifierMixin, BaseEstimator):
    """
    Must wrap trained xgb model bc focal loss:
    1) Produces model w/o classifier attribute
    2) Fails sklearn's check_is_fitted() check during calibration (when passed into FrozenEstimator())

    Notes
    -----
    - Not used for stage 1/2
    - Kept for legacy and potential future use
    """

    def __init__(self, model=None, **kwargs):
        if model is None:
            self.model = XGBClassifier(**kwargs)
        else:
            self.model = model
        self._estimator_type = "classifier"  # type: ignore
        # Propagate key fitted attributes
        for attr in ["classes_", "feature_names_in_", "n_features_in_"]:
            if hasattr(model, attr):
                setattr(self, attr, getattr(model, attr))
        # Ensure booster_ is present
        if hasattr(model, "booster"):
            self.booster_ = self.model.booster
        # For public learned attributes ending in _
        for attr in dir(model):
            if (
                attr.endswith("_")
                and not attr.startswith("__")
                and not hasattr(self, attr)
            ):
                try:
                    value = getattr(model, attr)
                    setattr(self, attr, value)
                except Exception:
                    continue

    def fit(self, X, y=None, **fit_params):
        self.model.fit(X, y, **fit_params)
        # propagate needed fitted attrs
        for attr in ["classes_", "feature_names_in_", "n_features_in_"]:
            if hasattr(self.model, attr):
                setattr(self, attr, getattr(self.model, attr))
        if hasattr(self.model, "booster"):
            self.booster_ = self.model.booster
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        out = self.model.get_params(deep=deep)
        # do not include the trained model itself; allow clone to rebuild it
        out["model"] = None
        return out

    def set_params(self, **params):
        # Remove model param if present
        params = {k: v for k, v in params.items() if k != "model"}
        self.model.set_params(**params)
        return self

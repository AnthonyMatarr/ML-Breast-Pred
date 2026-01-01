from pathlib import Path
import numpy as np
import pandas as pd

BASE_PATH = Path(__file__).parent.parent
from app.config import CHOSEN_MODEL_DICT


def load_population_probs(outcome_name):
    df = pd.read_parquet(
        BASE_PATH
        / "app"
        / "all_preds"
        / f"{outcome_name}_{CHOSEN_MODEL_DICT[outcome_name]}.parquet"
    )
    return df["prob"].values, df["label"].values


def load_bin_thresholds(outcome_name):
    """
    Loads bin thresholds for a given outcome/model from .npz files.
    Returns an array of bin edges.
    """
    thresholds_path = (
        BASE_PATH
        / "app"
        / "bin_thresholds"
        / f"{outcome_name}_{CHOSEN_MODEL_DICT[outcome_name]}.npz"
    )
    npz_data = np.load(thresholds_path)
    return npz_data["thresholds"]


def bin_occur_rates(outcome, thresholds):
    probs, true = load_population_probs(outcome)
    # thresholds = load_bin_thresholds(outcome)
    n_bins = len(thresholds) + 1
    bin_indices = np.digitize(probs, thresholds, right=False)  # type: ignore
    event_rates = []
    counts = []
    for b in range(n_bins):
        mask = bin_indices == b
        n = mask.sum()
        counts.append(n)
        if n == 0:
            event_rates.append(np.nan)
        else:
            event_rates.append(true[mask].mean())
    return event_rates


def get_risk_category(prob, outcome):
    """Assign outcome-specific risk category with emoji and color code."""
    thresholds = load_bin_thresholds(outcome)

    if prob < thresholds[0]:
        return "Very Low", "🟢", "#0ebd0d"  # Green
    elif prob < thresholds[1]:
        return "Low", "🟡", "#ffd401"  # Yellow
    elif prob < thresholds[2]:
        return "Moderate", "🟠", "#ee9410"  # Orange
    else:
        return "High", "🔴", "#c21615"  # Red


def transform_yes_no(input_val):
    if input_val == "Yes":
        return 1
    elif input_val == "No":
        return 0
    else:
        raise ValueError(f"Invalid input: {input_val}. Expected 'Yes' or 'No'")


def transform_sex(input_val):
    if input_val == "Male":
        return 0
    elif input_val == "Female":
        return 1
    else:
        raise ValueError(f"Invalid input: {input_val}. Expected 'Male' or 'Female'")


def transform_race(input_val):
    if input_val in ["White", "Black or African American", "Asian"]:
        return input_val
    elif input_val == "American Indian/Alaska Native":
        return "American Indian or Alaska Native"
    elif input_val == "Native Hawaiian/Pacific Islander":
        return "Native Hawaiian or Pacific Islander"
    # another func for this but putting here for consistency
    elif input_val == "Unknown/Other":
        return "otherUnknown"
    else:
        raise ValueError(f"Invalid input: {input_val}")


def transform_diab(input_val):
    if input_val.upper() in ["INSULIN", "ORAL"]:
        return input_val.upper()
    elif input_val == "No Diabetes":
        return "NO"
    else:
        return ValueError(f"Invalid input: {input_val}")


def transform_asa(input_val):
    """
    Note on encoding process for this variable.

    GUI Input
        - I, II, III, IV/V
    Pipeline Input
        - 1,2,3,4
    Model input (post pipeline)
        - 0,1,2,3
    Display:
        - I, II, III, IV/V

    Here we transform from GUI Input --> Pipeline input
    """
    match input_val:
        case "I":
            return 1
        case "II":
            return 2
        case "III":
            return 3
        case "IV/V":
            return 4
        case _:
            raise ValueError(f"Invalid input: {input_val}")


def transform_icd(input_val):
    match input_val:
        case "Breast Tumor (Malignant)":
            return "MALIGNANTICD"
        case "Inflammatory Breast Condition":
            return "INFLOTHERICD"
        case "Breast Tumor (Carcinoma in situ)":
            return "CARCINOMAICD"
        case "Breast Lesion (Benign)":
            return "BENIGNICD"
        case "Prophylaxis":
            return "PROPHYLACTICICD"
        case "Previous Mastectomy":
            return "ABSICD"
        case "Abnormal Breast Imaging":
            return "ABBREASTICD"
        case "Congenital Breast Disorder":
            return "CONGICD"
        case "Breast Tumor (Metastatic)":
            return "METASTATICICD"
        case _:
            raise ValueError(f"Invalid input: {input_val}")


def transform_spec(input_val):
    if input_val in ["General Surgery", "Plastic Surgery"]:
        return input_val
    elif input_val == "Other/unknown":
        return "otherUnknown"
    else:
        raise ValueError(f"Invalid input for surgical specialty: {input_val}")


def transform_anes(input_val):
    if input_val == "General":
        return input_val
    elif input_val == "Other/unknown":
        return "otherUnknown"
    elif input_val == "Monitored Anesthesia Care":
        return "MAC"
    else:
        raise ValueError(f"Invalid input for anesthesia: {input_val}")


def transform_inout(input_val):
    if input_val == "Inpatient":
        return 1
    elif input_val == "Outpatient":
        return 0
    else:
        raise ValueError(
            f"Invalid input: {input_val}. Expected 'Inpatient' or 'Outpatient'"
        )


def transform_casetype(input_val):
    if input_val == "Elective":
        return 0
    elif input_val == "Urgent/Emergent":
        return 1
    else:
        raise ValueError(
            f"Invalid input: {input_val}. Expected one of ['Elective', 'Urgent/Emergent']."
        )


def transform_ord_cpt(input_val):
    if input_val == "None":
        return "0"
    elif input_val == "Unilateral":
        return "1"
    elif input_val == "Bilateral":
        return "2+"
    else:
        raise ValueError(
            f"Invalid input: {input_val}. Expected one of ['None', 'Unilateral', 'Bilateral']."
        )


def transform_hispanic(input_val):
    if input_val == "Hispanic":
        return 1
    elif input_val == "Not Hispanic/Unknown":
        return 0
    else:
        raise ValueError(f"Unrecognized entry for hispanic {input_val}")


def transform_yes_no_unknown(input_val, col_name):
    if input_val in ["Yes", "No"]:
        return input_val
    elif input_val == "Unknown":
        if col_name == "RENAFAIL":
            yr = "21"
        elif col_name in ["DYSPNEA", "WNDINF", "WTLOSS"]:
            yr = "21-24"
        else:
            raise ValueError(
                f"Unrecognized column name for transforming yes/no/unknown: {col_name}"
            )
        return f"Unknown({yr})"
    else:
        raise ValueError(
            f"Unrecognized entry for transforming yes/no/unknown: {input_val}"
        )

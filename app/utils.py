from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

BASE_PATH = Path(__file__).parent.parent
from app.app_config import CHOSEN_MODEL_DICT


@st.cache_data
def load_population_probs(outcome_name):
    df = pd.read_parquet(
        BASE_PATH
        / "app"
        / "all_preds"
        / f"{outcome_name}_{CHOSEN_MODEL_DICT[outcome_name]}.parquet"
    )
    return df["prob"].values, df["label"].values


@st.cache_data
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


def transform_spec(input_val):
    if input_val in ["General Surgery", "Plastic Surgery"]:
        return input_val.replace(" Surgery", "")
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

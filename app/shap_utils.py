from src.feat_imp import get_ohe_cols, combine_encoded
import copy
import numpy as np
import pandas as pd
import streamlit as st


def split_catgeorical(feature_name, model_value):
    try:
        code_str_split = str(model_value).split("_")
        colname = code_str_split[0]
        code_val = code_str_split[1]
        try:
            assert colname == feature_name
        except AssertionError:
            raise ValueError(
                f"Split column name: {colname} does not match passed in column name: {feature_name}"
            )
        return code_val
    except IndexError:
        raise IndexError(f"Feat name: {feature_name}, value: {model_value}")


# --- inverse of transform_race ---
def inv_race(feature_name, code_str):
    code_str = split_catgeorical(feature_name, code_str)
    m = {
        "White": "White",
        "Black or African American": "Black or African American",
        "Asian": "Asian",
        "American Indian or Alaska Native": "American Indian/Alaska Native",
        "Native Hawaiian or Pacific Islander": "Native Hawaiian/Pacific Islander",
        "otherUnknown": "Unknown/Other",
    }
    try:
        return m[code_str]
    except KeyError:
        raise ValueError(f"Invalid race code: {code_str}")


# --- inverse of transform_diab ---
def inv_diab(feature_name, code_str):
    code_str = split_catgeorical(feature_name, code_str)
    m = {
        "INSULIN": "Insulin",
        "ORAL": "Oral",
        "NO": "No Diabetes",
    }
    try:
        return m[code_str]
    except KeyError:
        raise ValueError(f"Invalid diabetes code: {code_str}")


# --- inverse of transform_anes ---
def inv_anes(feature_name, code_str):
    code_str = split_catgeorical(feature_name, code_str)
    m = {
        "General": "General",
        "otherUnknown": "Other/unknown",
        "MAC": "Monitored Anesthesia Care",
    }
    try:
        return m[code_str]
    except KeyError:
        raise ValueError(f"Invalid ANESTHES code: {code_str}")


# --- inverse of transform_spec ---
def inv_spec(feature_name, code_str):
    code_str = split_catgeorical(feature_name, code_str)
    m = {
        "General": "General Surgery",
        "Plastic": "Plastic Surgery",
        "otherUnknown": "Other/unknown",
    }
    try:
        return m[code_str]
    except KeyError:
        raise ValueError(f"Invalid SURGSPEC code: {code_str}")


# --- inverse of transform_asa ---
def inv_asa(code):
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

    Here we transform from Model input --> Display
    """
    code = int(code)
    m = {
        0: "I",
        1: "II",
        2: "III",
        3: "IV/V",
    }
    try:
        return m[code]
    except KeyError:
        raise ValueError(f"Invalid ASA code: {code}")


# --- inverse of transform_ord_cpt ---
def inv_ord_cpt(code_str):
    # model stores "0","1","2+" (string) after transform
    code_str = int(float(code_str))
    m = {
        0: "None",
        1: "Unilateral",
        2: "Bilateral",
    }
    try:
        return m[code_str]
    except KeyError:
        raise ValueError(f"Invalid ordinal CPT code: {code_str}")


# --- inverse of transform_casetype ---
def inv_casetype(code):
    code = int(code)
    if code == 0:
        return "Elective"
    if code == 1:
        return "Urgent/Emergent"
    raise ValueError(f"Invalid URGENCY code: {code}")


# --- inverse of transform_inout ---
def inv_inout(code):
    code = int(code)
    if code == 1:
        return "Inpatient"
    if code == 0:
        return "Outpatient"
    raise ValueError(f"Invalid INOUT code: {code}")


# --- inverse of transform_yes_no ---
def inv_yes_no(code):
    code = int(code)
    if code == 1:
        return "Yes"
    if code == 0:
        return "No"
    raise ValueError(f"Invalid yes/no code: {code}")


############################################################################################################
############################################################################################################
############################################################################################################
def get_explainer_name(model_name):
    if model_name in ["xgb", "lgbm"]:
        return "TreeExplainer"
    elif model_name in ["svc", "lr"]:
        return "LinearExplainer"
    else:
        return "KernelExplainer"


def decode_cat_for_display(feature_name, model_value):
    """
    Map model-level codes back to the same labels used in the Streamlit UI.
    feature_name: column name in the model input / SHAP features
    model_value:  value in shap_combined.display_data (or original data) for that feature
    """
    # normalize once
    code_val = str(model_value)
    # demographics
    if feature_name == "RACE":
        return inv_race(feature_name, code_val)
    if feature_name == "DIABETES":
        return inv_diab(feature_name, code_val)

    # simple yes/no binaries
    if feature_name in [
        "SMOKE",
        "HXCOPD",
        "HXCHF",
        "HYPERMED",
        "BLEEDDIS",
        "DISCANCR",
    ]:
        return inv_yes_no(float(code_val))

    # ASA class
    if feature_name == "ASACLAS":
        return inv_asa(float(code_val))

    # intra-op categorical
    if feature_name == "SURGSPEC":
        print(feature_name)
        print(code_val)
        return inv_spec(feature_name, code_val)
    if feature_name == "ANESTHES":
        return inv_anes(feature_name, code_val)
    if feature_name == "INOUT":
        return inv_inout(float(code_val))
    if feature_name == "URGENCY":
        return inv_casetype(float(code_val))

    # ordinal CPT-style codes
    if feature_name in [
        # Resection
        "PARTIALCPT",
        "SUBSIMPLECPT",
        "MODIFIEDRADICALCPT",
        # Axillary
        "SNLBCPT",
        "ALNDCPT",
        # Implant
        "IMMEDIATECPT",
        "TEINSERTIONCPT",
        # Autologous
        "FREECPT",
        "SINTRAMCPT",
        # Adjunct
        "BREASTREDCPT",
        "ADJTISTRANSCPT",
    ]:
        return inv_ord_cpt(code_val)

    # default: just cast to string
    raise ValueError(
        f"Unrecognized feature name: {feature_name} with value: {code_val}"
    )


def feature_value_label(name, disp_value, num_series):
    # numeric features: as before using num_original_series
    if name in num_series.index:
        val = num_series[name]
        if name == "AGE":
            return f"{val:.0f}"
        elif name == "BMI":
            return f"{val:.1f}"
        else:
            return f"{val:.2f}"

    # categorical / binary / ordinal: decode to Streamlit label
    return decode_cat_for_display(name, disp_value)


def pretty_feature_name(name):
    feature_label_map = {
        # Demographics / baseline
        "AGE": "Age (years)",
        "RACE": "Race",
        "BMI": "BMI (kg/m²)",
        # Comorbidities / status
        "SMOKE": "Current smoker",
        "DIABETES": "Diabetes status",
        "HXCOPD": "COPD",
        "HXCHF": "CHF",
        "HYPERMED": "Hypertension",
        "BLEEDDIS": "Bleeding disorder",
        "DISCANCR": "Disseminated cancer",
        "ASACLAS": "ASA class",
        # Preoperative labs
        "PRALBUM": "Albumin (g/dL)",
        "PRWBC": "WBC (×10³/µL)",
        "PRHCT": "Hematocrit (%)",
        "PRPLATE": "Platelet (×10³/µL)",
        # Intraoperative / setting
        "URGENCY": "Case type",
        "ANESTHES": "Anesthesia",
        "SURGSPEC": "Specialty",
        "INOUT": "Setting",
        # Resection
        "PARTIALCPT": "Partial mastectomy",
        "SUBSIMPLECPT": "Simple mastectomy",
        "MODIFIEDRADICALCPT": "Modified radical mastectomy",
        # Axillary
        "SNLBCPT": "Sentinel lymph node biopsy",
        "ALNDCPT": "Axillary lymph node dissection",
        # Implant
        "IMMEDIATECPT": "Immediate Implant",
        "TEINSERTIONCPT": "Tissue expander insertion",
        # Autologous
        "FREECPT": "Free flap",
        "SINTRAMCPT": "Single-pedicle TRAM flap",
        # Adjunct
        "BREASTREDCPT": "Breast reduction",
        "ADJTISTRANSCPT": "Adjacent tissue transfer",
    }

    return feature_label_map.get(name, name)


def combine_encoded_for_app(input_data, shap_raw):
    ohe_dict = get_ohe_cols(input_data)
    ohe_cols = ohe_dict.keys()
    raw_feat_order = []
    for col in input_data.columns.to_list():
        if col in ohe_cols:
            for sub_col in ohe_dict[col]:
                raw_feat_order.append(f"{col}_{sub_col}")
        else:
            raw_feat_order.append(col)

    shap_old = copy.deepcopy(shap_raw)
    for col_name in ohe_cols:
        shap_combined, _ = combine_encoded(
            shap_old, col_name, [col_name in n for n in shap_old.feature_names]  # type: ignore
        )
        shap_old = shap_combined
    return shap_combined


@st.cache_data
def compute_shap_data(
    _explainer, _input_data, _pipeline, processed_data_hash, outcome_name
):
    """
    Compute SHAP values once and cache them.
    processed_data_hash and outcome_name are used soley to invalidate cache when input changes.

    All parameters prefixed with _ to tell Streamlit not to hash them directly.
    """
    expected_features = _explainer.feature_names
    input_data = _input_data[expected_features].copy()
    shap_raw = _explainer(input_data)

    # Combine one-hot encoded values
    shap_combined = combine_encoded_for_app(input_data, shap_raw)

    # Get scaler for inverse transform
    num_name, num_pipe, num_cols = _pipeline.transformers_[0]
    assert num_name == "num"
    scaler = num_pipe.named_steps["scaler"]

    # numeric outputs after BMI step (hard-coded order)
    feature_names = _pipeline.get_feature_names_out()
    num_out_cols = [
        n.replace("num__", "") for n in feature_names if n.startswith("num__")
    ]
    # Inverse transform numeric features
    feat_names = list(shap_combined.feature_names)
    num_indices = [feat_names.index(col) for col in num_out_cols]
    x_trans_row = shap_combined.data[0]
    x_num_scaled = np.array([x_trans_row[i] for i in num_indices]).reshape(1, -1)
    x_num_original = scaler.inverse_transform(x_num_scaled)
    num_original_series = pd.Series(x_num_original.ravel(), index=num_out_cols)
    # Return all data needed for plotting
    return {
        "phi": shap_combined.values[0],
        "feat_names": np.array(shap_combined.feature_names),
        "disp_row": (
            shap_combined.display_data[0]
            if shap_combined.display_data is not None
            else shap_combined.data[0]
        ),
        "num_original_series": num_original_series,
    }

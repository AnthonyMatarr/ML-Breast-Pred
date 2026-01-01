## Append path to root
import sys
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_PATH))
from src.preprocess import remove_prefix
import app.utils as util
from app.config import CHOSEN_MODEL_DICT
from app.shap_utils import (
    pretty_feature_name,
    feature_value_label,
    compute_shap_data,
    get_explainer_name,
)
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import hashlib


#################### MAIN CLINICAL RESULTS ####################
# Cache model loading for performance
@st.cache_resource
def load_model_pipeline_explainer(outcome_name):
    """Load model and preprocessor for a specific outcome."""
    ## MODEL
    model_path = (
        BASE_PATH
        / "app"
        / "models"
        / "calibrated"
        / f"{outcome_name}_{CHOSEN_MODEL_DICT[outcome_name]}.joblib"
    )
    model = joblib.load(model_path)
    ## PIPELINE
    preprocessor = joblib.load(
        BASE_PATH / "app" / "preprocessors" / f"{outcome_name}_pipeline.joblib"
    )
    ## EXPLAINER
    explainer = joblib.load(
        BASE_PATH / "app" / "shap_explainers" / f"{outcome_name}.joblib"
    )
    return model, preprocessor, explainer


def plot_risk_bins(bin_occur_rates, bin_idx, folder_name, color):
    """
    Plot occurrence rates per bin with patient's bin highlighted.

    Args:
        bin_occur_rates: List of occurrence rates [0.001, 0.004, 0.009, 0.018]
        bin_idx: Index of patient's assigned bin (0-3)
        display_name: Name of outcome for labeling
    """
    labels = ["Very Low", "Low", "Moderate", "High"]

    # Create color list: highlight patient's bin
    # colors = ["#1f77b4"] * len(labels)  # default blue
    colors = ["#BDBDBD"] * len(labels)  # default grey
    # colors[bin_idx] = "#ff7f0e"  # highlight in orange
    colors[bin_idx] = color

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, [rate * 100 for rate in bin_occur_rates], color=colors)

    # Add percentage labels on top of bars
    for i, (bar, rate) in enumerate(zip(bars, bin_occur_rates)):
        height = bar.get_height()
        label = f"{rate:.2%}"

        # Make highlighted bar's label bold
        if i == bin_idx:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                label,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
                color="black",  # Black for selected
                zorder=10,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                label,
                ha="center",
                va="bottom",
                fontweight="normal",
                fontsize=10,
                color="#BDBDBD",  # Grey for non-selected
                zorder=10,
            )
    # Grey out x-axis tick labels for non-selected bins
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    # Modify tick label colors
    for i, tick_label in enumerate(ax.get_xticklabels()):
        if i == bin_idx:
            tick_label.set_color("black")
            tick_label.set_fontweight("bold")
        else:
            tick_label.set_color("#BDBDBD")  # Grey for non-selected
    # horizontal line for overall population rate
    _, true = util.load_population_probs(folder_name)
    tot_occur_rate = true.mean()  # type: ignore
    ax.axhline(
        y=tot_occur_rate * 100,
        color="#467fb8",  # Blue to contrast with grey/colored bars
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"Average risk across all breast surgery patients: {tot_occur_rate:.1%}",
        zorder=5,  # Ensure line appears above grid
    )
    # legend for the reference line
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    ax.set_ylabel(
        "Complication Rate (%)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel(
        "Risk Category",
        fontsize=11,
        fontweight="bold",
    )
    # ax.set_title(f"{display_name} Occurrence by Risk Category", fontsize=12)
    ax.set_title(
        "Observed Complication Rate by Risk Category",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0, max([r * 100 for r in bin_occur_rates]) * 1.15)  # Add headroom

    # Add grid for readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    return fig


def show_imputed(display_name, folder_name, input_data, num_dict, imp_cols):
    """
    Display imputed values if applicable
    """
    _, preprocessor, _ = load_model_pipeline_explainer(folder_name)
    with st.expander(f"📋 Imputed Values for {display_name}"):
        ## Get pipeline steps
        num_pipe = preprocessor.named_transformers_["num"]
        imputer = num_pipe.named_steps["imputer"]
        ## Get intermediate values
        X_num_raw = input_data[list(num_dict.keys())]
        X_imputed = imputer.transform(X_num_raw)
        # Build a small df for display
        num_cols_list = list(num_dict.keys())
        imp_display = {}
        for col in imp_cols:
            ## Find index amongst num cols
            col_idx = num_cols_list.index(col)
            raw_val = X_imputed[0, col_idx]
            round_rule = num_dict[col]["round rule"]
            imp_display[num_dict[col]["Display Name"]] = round_rule(raw_val)

        display_df = pd.DataFrame.from_dict(
            imp_display, orient="index", columns=["Value"]
        )
        st.dataframe(display_df, width="content")


def get_dynamic_BMI():
    # ================== Dynamic Weight ===================
    # Initialize session state
    if "prev_weight_unit" not in st.session_state:
        st.session_state.prev_weight_unit = "lbs"
    if "weight_kg" not in st.session_state:
        st.session_state.weight_kg = 73.94
    if "weight_lbs" not in st.session_state:
        st.session_state.weight_lbs = 164.0

    # Nested columns: input on left, checkbox on right
    weight_input_col, weight_check_col = st.columns([3, 2])

    with weight_check_col:
        st.write("")  # Spacer to align with input
        weight_unknown = st.checkbox("N/A", key="weight_unknown")

    # Unit selector below both
    if not weight_unknown:
        weight_unit = st.radio(
            "Weight unit",
            ["lbs", "kg"],
            index=0 if st.session_state.prev_weight_unit == "lbs" else 1,
            key="weight_unit",
            horizontal=True,
            help="Unit used for weight entry (lbs or kg).",
        )
        # Detect if the unit changed and convert before rendering inputs
        if weight_unit != st.session_state.prev_weight_unit:
            if weight_unit == "lbs":
                # KG -> LBS
                st.session_state.weight_lbs = st.session_state.weight_kg * 2.20462
            else:
                # LBS -> KG
                st.session_state.weight_kg = st.session_state.weight_lbs / 2.20462
            st.session_state.prev_weight_unit = weight_unit

    # Render input box
    with weight_input_col:
        if weight_unknown:
            # Show current stored value (grayed out) instead of 0
            display_val = (
                st.session_state.weight_lbs
                if st.session_state.prev_weight_unit == "lbs"
                else st.session_state.weight_kg
            )
            st.number_input(
                f"Weight ({st.session_state.prev_weight_unit})",
                value=float(display_val),
                disabled=True,
                help="""
                Patient's body weight, recorded preoperatively. Used to calculate BMI. 
                
                *Currently marked as unknown - see ***Imputed Values*** after model prediction at bottom of page for more information about how this NA value is dealt with*
                """,
            )
            weight = None
        else:
            weight_unit = st.session_state.get("weight_unit", "lbs")
            if weight_unit == "lbs":
                weight = st.number_input(
                    "Weight (lbs)",
                    min_value=2.20462,
                    value=st.session_state.weight_lbs,
                    key="weight_lbs",
                    help="Patient's body weight, recorded preoperatively. Used to calculate BMI.",
                )
            else:
                weight_kg = st.number_input(
                    "Weight (kg)",
                    min_value=1.0,
                    value=st.session_state.weight_kg,
                    key="weight_kg",
                    help="Patient's body weight, recorded preoperatively. Used to calculate BMI.",
                )
                weight = weight_kg * 2.20462
    st.markdown("---")
    # ================== Dynamic Height ===================
    # Initialize session state
    if "prev_height_unit" not in st.session_state:
        st.session_state.prev_height_unit = "in"
    if "height_m" not in st.session_state:
        st.session_state.height_m = 1.63
    if "height_in" not in st.session_state:
        st.session_state.height_in = 64.0

    # Nested columns: input on left, checkbox on right
    height_input_col, height_check_col = st.columns([3, 2])

    with height_check_col:
        st.write("")  # Spacer to align with input
        height_unknown = st.checkbox("N/A", key="height_unknown")

    # Unit selector below both
    if not height_unknown:
        height_unit = st.radio(
            "Height unit",
            ["in", "m"],
            index=0 if st.session_state.prev_height_unit == "in" else 1,
            key="height_unit",
            horizontal=True,
            help="Unit used for height entry (in or m).",
        )
        # Detect if the unit changed and convert before rendering inputs
        if height_unit != st.session_state.prev_height_unit:
            if height_unit == "in":
                # Meters -> Inches
                st.session_state.height_in = st.session_state.height_m * 39.3701
            else:
                # Inches -> Meters
                st.session_state.height_m = st.session_state.height_in / 39.3701
            st.session_state.prev_height_unit = height_unit

    # Render input box
    with height_input_col:
        if height_unknown:
            # Show current stored value (grayed out) instead of 0
            display_val = (
                st.session_state.height_in
                if st.session_state.prev_height_unit == "in"
                else st.session_state.height_m
            )
            st.number_input(
                f"Height ({st.session_state.prev_height_unit})",
                value=float(display_val),
                disabled=True,
                help="""
                    Patient's height as recorded preoperatively. Used to calculate BMI.
                    
                    *Currently marked as unknown - see ***Imputed Values*** after model prediction at bottom of page for more information about how this NA value is dealt with*
                    """,
            )
            height = None
        else:
            height_unit = st.session_state.get("height_unit", "in")
            if height_unit == "in":
                height = st.number_input(
                    "Height (in)",
                    min_value=39.3701,
                    value=st.session_state.height_in,
                    key="height_in",
                    help="Patient's height as recorded preoperatively. Used to calculate BMI.",
                )
            else:
                height_m = st.number_input(
                    "Height (m)",
                    min_value=1.0,
                    value=st.session_state.height_m,
                    key="height_m",
                    help="Patient's height as recorded preoperatively.",
                )
                height = height_m * 39.3701
    st.markdown("---")
    # ================== BMI Display ===================
    if weight is None or height is None:
        st.info(
            "**Current BMI:** N/A (height or weight missing; will be imputed later)"
        )
    else:
        bmi = (weight * 703) / (height**2)
        st.success(f"**Current BMI:** {bmi:.1f} kg/m²")
    return height, weight


def get_input_data():
    st.header("Patient Information")
    col1, col2, col3, col4 = st.columns(4)
    # ================== Demographics, BMI, Comorbidities ===================
    with col1:
        with st.expander("**Demographics**", expanded=False):
            # ================== Age ===================
            # Nested columns: input on left, checkbox on right
            age_input_col, age_check_col = st.columns([3, 2])
            with age_check_col:
                st.write("")  # Spacer to align with input
                age_unknown = st.checkbox("N/A", key="age_unknown")
            with age_input_col:
                if age_unknown:
                    st.number_input(
                        "Age",
                        value=0.0,
                        disabled=True,
                        help="""
                            Age in years on the date of the principal operative procedure. 
                            
                            *Currently marked as unknown - see ***Imputed Values*** after model prediction at 
                            bottom of page for more information about how this NA value is dealt with*
                            """,
                    )
                    age = None
                else:
                    age = st.number_input(
                        "Age",
                        min_value=18,
                        max_value=90,
                        value=56,
                        help="Age in years on the date of the principal operative procedure.",
                    )
            # --> male(0)/female(1)
            sex = st.selectbox(
                "Sex",
                ["Male", "Female"],
                index=1,
                help="Patient-reported biological sex at the time of surgery.",
            )
            hispanic = st.selectbox(
                "Ethnicity",
                ["Hispanic", "Not Hispanic/Unknown"],
                index=1,
                help="Self-reported ethnicity, categorized as Hispanic or Not Hispanic/Unknown.",
            )
            race = st.selectbox(
                "Race",
                [
                    "White",
                    "Black or African American",
                    "Asian",
                    "American Indian/Alaska Native",  # American Indian or Alaska Native
                    "Native Hawaiian/Pacific Islander",  # Native Hawaiian or Pacific Islander
                    "Unknown/Other",  # --> otherUnknown
                ],
                index=0,
                help="Self-reported race category documented in the medical record.",
            )

        with st.expander("**BMI**", expanded=False):
            height, weight = get_dynamic_BMI()

        with st.expander("**Comorbidities**", expanded=False):
            smoke = st.selectbox(
                "Current Smoker (within 1 year)",
                ["Yes", "No"],
                index=1,
                help="The patient has smoked cigarettes within 1 year before surgery. Use of cigars, pipes, or smokeless tobacco is not included.",
            )
            diabetes = st.selectbox(
                "Diabetes",
                [
                    "Oral",
                    "Insulin",
                    "No Diabetes",  # --> No
                ],
                index=2,
                help="The patient requires daily exogenous insulin or oral hypoglycemic agents. Diet-controlled diabetes does not qualify.",
            )
            hxcopd = st.selectbox(
                "COPD",
                ["Yes", "No"],
                index=1,
                help="""
                    A history of emphysema and/or chronic bronchitis meeting at least one of the following:

                    &nbsp;&nbsp;&nbsp;&nbsp;•Functional limitation requiring oxygen or limiting ADLs
                    
                    &nbsp;&nbsp;&nbsp;&nbsp;•Prior hospitalization for COPD
                    
                    &nbsp;&nbsp;&nbsp;&nbsp;•Chronic bronchodilator therapy
                    
                    &nbsp;&nbsp;&nbsp;&nbsp;•FEV1 <75% predicted
                    
                    *NOTE:* Asthma, interstitial fibrosis, or sarcoidosis are excluded.
                    """,
            )
            dyspnea = st.selectbox(
                "Dyspnea",
                ["Yes", "No", "Unknown"],
                index=1,
                help="Shortness of breath noted at rest or on exertion within 30 days pre-op, based on clinical documentation.",
            )
            hxchf = st.selectbox(
                "Congestive Heart Failure",
                ["Yes", "No"],
                index=1,
                help="Newly diagnosed CHF within 30 days or chronic CHF with active symptoms/signs in the 30 days prior to surgery.",
            )
            hypermed = st.selectbox(
                "Hypertension Requiring Medication",
                ["Yes", "No"],
                index=1,
                help="Documented diagnosis of hypertension requiring antihypertensive medication within 30 days before surgery.",
            )
            dialysis = st.selectbox(
                "Dialysis",
                ["Yes", "No"],
                index=1,
                help="Patient has required peritoneal dialysis, hemodialysis, hemofiltration, hemodiafiltration, or ultrafiltration within 2 weeks prior to surgery. Patients who refuse indicated dialysis are coded 'Yes'.",
            )
            renafail = st.selectbox(
                "Acute Renal Failure",
                ["Yes", "No", "Unknown"],
                index=1,
                help="""
                    Preoperative renal dysfunction defined as Stage 2/3 AKI:

                    &nbsp;&nbsp;&nbsp;&nbsp;• **Stage 2:** Serum creatinine 2.0–<3.0× baseline within 7 days
                    
                    &nbsp;&nbsp;&nbsp;&nbsp;• **Stage 3:** Any of the following:
                    
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• ≥3.0× baseline within 7 days
                    
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Rise of ≥0.3 mg/dL to ≥4.0 mg/dL within 48 hours
                    
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• ≥1.5× baseline to ≥4.0 mg/dL within 7 days
                    """,
            )
            bleed = st.selectbox(
                "Bleeding Disorder",
                ["Yes", "No"],
                index=1,
                help="History of congenital or acquired bleeding diathesis, anticoagulation therapy, or clinical coagulopathy documented preoperatively.",
            )
            wtloss = st.selectbox(
                "Weight Loss",
                ["Yes", "No", "Unknown"],
                index=1,
                help="Unintentional weight loss >10% of body weight in the 6 months prior to surgery.",
            )
            ascites = st.selectbox(
                "Ascites",
                ["Yes", "No"],
                index=1,
                help="Clinically detectable or radiographically confirmed peritoneal fluid within 30 days pre-op. Must be associated with liver disease or malignancy unless otherwise documented.",
            )
            steroid = st.selectbox(
                "Corticosteroid",
                ["Yes", "No"],
                index=1,
                help="Use of systemic corticosteroids, anti-rejection agents, DMARDs, or other immunosuppressants for ≥10 days within 30 days pre-op, or on an active long-interval regimen extending into the surgical period.",
            )
            discancr = st.selectbox(
                "Disseminated Cancer",
                ["Yes", "No"],
                index=1,
                help="""
                Metastatic cancer to at least one major organ and ***ONE*** of the following:

                &nbsp;&nbsp;&nbsp;&nbsp;•Receiving active treatment within the past year

                &nbsp;&nbsp;&nbsp;&nbsp;•Declined treatment
                
                &nbsp;&nbsp;&nbsp;&nbsp;•Deemed untreatable
                
                **Includes:** ALL, AML, Stage IV lymphomas.

                **Excludes:** CLL, CML, lymphomas stages I–III, multiple myeloma.
                """,
            )
            vent = st.selectbox(
                "Preoperative Ventilator Dependence",
                ["Yes", "No"],
                index=1,
                help="The patient requires mechanical ventilation for any duration during the 48 hours immediately preceding surgery. CPAP for sleep apnea is excluded.",
            )
            wndinf = st.selectbox(
                "Wound Infection",
                ["Yes", "No", "Unknown"],
                index=1,
                help="Presence of superficial, deep, or organ/space infection documented preoperatively.",
            )
            asa_class = st.selectbox(
                "ASA Physical Status Classification",
                [
                    "I",  # --> 1-No Disturb
                    "II",  # --> 2-Mild Disturb
                    "III",  # --> 3-Severe Disturb
                    "IV/V",  # --> 4/5
                ],
                index=2,
                help="""
                    &nbsp;&nbsp;&nbsp;&nbsp;• **ASA I:** Normal healthy patient

                    &nbsp;&nbsp;&nbsp;&nbsp;• **ASA II:** Mild systemic disease
                    
                    &nbsp;&nbsp;&nbsp;&nbsp;• **ASA III:** Severe systemic disease
                    
                    &nbsp;&nbsp;&nbsp;&nbsp;• **ASA IV:** Severe systemic disease, constant threat to life
                    
                    &nbsp;&nbsp;&nbsp;&nbsp;• **ASA V:** Moribund, not expected to survive without operation
                    """,
            )
            transfus = st.selectbox(
                "Blood Transfusion (Preoperative)",
                ["Yes", "No"],
                index=1,
                help="Receipt of ≥1 unit of packed RBCs within the 72 hours prior to surgery.",
            )
        # ================== Preoperative Laboratory Valuess ===================
        with st.expander("**Preoperative Laboratory Values**", expanded=False):
            # ====================> Albumin
            alb_input_col, alb_check_col = st.columns([3, 2])

            with alb_check_col:
                st.write("")
                alb_unknown = st.checkbox("N/A", key="alb_unknown")

            with alb_input_col:
                if alb_unknown:
                    st.number_input(
                        "Albumin (g/dL)",
                        value=0.0,
                        disabled=True,
                        help="""
                            Serum albumin drawn within 90 days pre-op. 
                            
                            *Currently marked as unknown - see ***Imputed Values*** after model prediction 
                            at bottom of page for more information about how this NA value is dealt with*
                            """,
                    )
                    pralbumin = None
                else:
                    pralbumin = st.number_input(
                        "Albumin (g/dL)",
                        min_value=0.0,
                        max_value=None,
                        value=4.2,
                        help="Serum albumin drawn within 90 days pre-op.",
                    )

            # ====================> WBC
            wbc_input_col, wbc_check_col = st.columns([3, 2])

            with wbc_check_col:
                st.write("")
                wbc_unknown = st.checkbox("N/A", key="wbc_unknown")

            with wbc_input_col:
                if wbc_unknown:
                    st.number_input(
                        "White Blood Cell Count",
                        value=0.0,
                        disabled=True,
                        help="""
                            Most recent WBC value (*10^9/L) within 90 days prior to surgery. 
                            
                            *Currently marked as unknown - see ***Imputed Values*** after model prediction at bottom of 
                            page for more information about how this NA value is dealt with*
                            """,
                    )
                    prwbc = None
                else:
                    prwbc = st.number_input(
                        "White Blood Cell Count",
                        min_value=0.0,
                        max_value=None,
                        value=6.5,
                        help="Most recent WBC value (*10^9/L) within 90 days prior to surgery.",
                    )

            # ====================> HCT
            hct_input_col, hct_check_col = st.columns([3, 2])

            with hct_check_col:
                st.write("")
                hct_unknown = st.checkbox("N/A", key="hct_unknown")

            with hct_input_col:
                if hct_unknown:
                    st.number_input(
                        "Hematocrit (%)",
                        value=0.0,
                        disabled=True,
                        help="""
                            Most recent Hct (%) within 90 days prior to surgery. 
                            
                            *Currently marked as unknown - see ***Imputed Values*** after model prediction at bottom of 
                            page for more information about how this NA value is dealt with*
                            """,
                    )
                    prhct = None
                else:
                    prhct = st.number_input(
                        "Hematocrit (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=39.6,
                        help="Most recent Hct (%) within 90 days prior to surgery.",
                    )

            # ====================> PLATE
            plate_input_col, plate_check_col = st.columns([3, 2])

            with plate_check_col:
                st.write("")
                plate_unknown = st.checkbox("N/A", key="plate_unknown")

            with plate_input_col:
                if plate_unknown:
                    st.number_input(
                        "Platelet Count",
                        value=0.0,
                        disabled=True,
                        help="""
                            Most recent platelet level (*10^9/L) within 90 days prior to surgery. 
                            
                            *Currently marked as unknown - see ***Imputed Values*** after model prediction at 
                            bottom of page for more information about how this NA value is dealt with*
                            """,
                    )
                    prplate = None
                else:
                    prplate = st.number_input(
                        "Platelet Count",
                        min_value=0.0,
                        max_value=None,
                        value=254.0,
                        help="Most recent platelet level (*10^9/L) within 90 days prior to surgery.",
                    )
    # ================== Operative Characteristics ===================
    with col2:
        with st.expander("**Operative Characteristics**", expanded=False):
            surg_icd = st.selectbox(
                "Diagnosis",
                [
                    "Breast Tumor (Malignant)",  # --> MALIGNANTICD
                    "Inflammatory Breast Condition",  # --> INFLOTHERICD
                    "Breast Tumor (Carcinoma in situ)",  # --> CARCINOMAICD
                    "Breast Lesion (Benign)",  # --> BENIGNICD
                    "Prophylaxis",  # --> PROPHYLACTICICD
                    "Previous Mastectomy",  # --> ABSICD
                    "Abnormal Breast Imaging",  # --> ABBREASTICD
                    "Congenital Breast Disorder",  # --> CONGICD
                    "Breast Tumor (Metastatic)",  # --> METASTATICICD
                ],
                index=0,
                help="""
                    Primary indication for the breast operation, categorized using mutually exclusive NSQIP-based ICD-9/ICD-10 codes, including:

                    &nbsp;&nbsp;&nbsp;&nbsp;• **carcinoma in situ** (ICD-9 233; ICD-10 D05)

                    &nbsp;&nbsp;&nbsp;&nbsp;• **malignant neoplasm** (174, 175, C50)

                    &nbsp;&nbsp;&nbsp;&nbsp;• **prophylactic indications** (V50.4, V84, Z15, Z40)

                    &nbsp;&nbsp;&nbsp;&nbsp;• **abnormal imaging findings** (793.8, R92)

                    &nbsp;&nbsp;&nbsp;&nbsp;• **metastatic disease** (198.8, C79, C78)

                    &nbsp;&nbsp;&nbsp;&nbsp;• **benign breast disorders** (217, D24, 610, N60)

                    &nbsp;&nbsp;&nbsp;&nbsp;• **inflammatory conditions** (611, N61–N64)

                    &nbsp;&nbsp;&nbsp;&nbsp;• **congenital breast disorders** (757.6, Q83)

                    &nbsp;&nbsp;&nbsp;&nbsp;• **acquired absence of breast** (V45.71, Z90.1)
                    """,
            )
            # ========================> Operation Time
            optime_input_col, optime_check_col = st.columns([3, 2])

            with optime_check_col:
                st.write("")  # Spacer to align with input
                optime_unknown = st.checkbox("N/A", key="optime_unknown")

            with optime_input_col:
                if optime_unknown:
                    st.number_input(
                        "Operation Time (minutes)",
                        value=0.0,
                        disabled=True,
                        help="""
                            Total operative duration in minutes for the principal procedure and related components. 
                            
                            *Currently marked as unknown - see ***Imputed Values*** after model prediction at 
                            bottom of page for more information about how this NA value is dealt with*
                            """,
                    )
                    optime = None
                else:
                    optime = st.number_input(
                        "Operation Time (minutes)",
                        min_value=0.0,
                        max_value=None,
                        value=83.0,
                        help="Total operative duration in minutes for the principal procedure and related components.",
                    )

            operyr = st.number_input(
                "Operation Year",
                min_value=2008,
                max_value=2026,
                value=2026,
                step=1,
                help="""
                    Calendar year during which the operation occurred. 

                    **NOTE:** Only patients from 2008-2024 were used to train/evaluate these models
                    """,
            )
            elect_surg = st.selectbox(
                "Case Type",
                [
                    "Elective",  # --> 0
                    "Urgent/Emergent",  # --> 1
                ],
                index=0,
                help="""
                    Determination of operative priority based on surgeon/anesthesiologist classification.
                    
                    &nbsp;&nbsp;&nbsp;&nbsp;• **Urgent/Emergent:** Must occur during same admission or within 48 hours and documented as urgent/emergent. 
                    
                    &nbsp;&nbsp;&nbsp;&nbsp;• **Elective:** Planned procedure, scheduled in advance for non-life-threatening issues or quality of life.
                    """,
            )
            anesth = st.selectbox(
                "Anesthesia",
                [
                    "General",
                    "Other/unknown",  # --> otherUnknown
                    "Monitored Anesthesia Care",  # --> MAC
                ],
                index=0,
                help="""
                    The primary type of anesthesia administered for the index surgical procedure, as recorded in NSQIP. Categories generally include:

                    &nbsp;&nbsp;&nbsp;&nbsp;• **General anesthesia:** Endotracheal or laryngeal mask ventilation with loss of consciousness.

                    &nbsp;&nbsp;&nbsp;&nbsp;• **Regional anesthesia:** Spinal, epidural, or peripheral nerve block used as the primary anesthetic.

                    &nbsp;&nbsp;&nbsp;&nbsp;• **Monitored anesthesia care (MAC):** Intravenous sedation with local anesthesia, without securing the airway.

                    &nbsp;&nbsp;&nbsp;&nbsp;• **Local anesthesia only:** Local infiltration without sedation or regional technique.

                    &nbsp;&nbsp;&nbsp;&nbsp;• **Other/Unknown:** Any anesthesia type not captured by the standard NSQIP categories.
                    """,
            )
            surg_spec = st.selectbox(
                "Surgical Specialty",
                ["General Surgery", "Plastic Surgery", "Other/unknown"],
                index=0,
                help="The surgical specialty of the **attending surgeon** responsible for the index operation, as documented in NSQIP.",
            )
            inout = st.selectbox(
                "Setting",
                ["Inpatient", "Outpatient"],
                index=1,
                help="Hospital admission status at time of the surgical procedure.",
            )

    # ==================Breast Resection Procedures/Axillary Surgery ===================
    with col3:
        with st.expander("**Breast Resection Procedures**", expanded=False):
            part_cpt = st.selectbox(
                "Partial mastectomy",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Breast-conserving surgery involving excision of tumor with margin of normal tissue, including lumpectomy, segmental mastectomy, and quadrantectomy. 
                    
                    **CPT codes: 19301, 19302, 19125, 19120, 19160, 19162**
                    """,
            )
            sub_simp = st.selectbox(
                "Simple (Total) Mastectomy",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Removal of the entire breast tissue without formal axillary dissection; includes subcutaneous and skin-sparing variations. 

                    **CPT codes: 19303, 19304, 19180, 19182**
                    """,
            )
            modrad = st.selectbox(
                "Modified Radical Mastectomy",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Removal of the entire breast with preservation of the pectoralis major muscle and concurrent axillary lymph node dissection. 

                    **CPT codes: 19307, 19240** 
                    """,
            )
            rad = st.selectbox(
                "Radical Mastectomy",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    En bloc removal of the entire breast, pectoralis muscles, and axillary contents (Halsted mastectomy). 

                    **CPT codes: 19305, 19306, 19200, 19220**
                    """,
            )
        with st.expander("**Axillary Surgery**", expanded=False):
            slnb = st.selectbox(
                "Sentinel Lymph Node Biopsy",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Performance of sentinel lymph node biopsy for staging or treatment of breast disease using injection/identification techniques with limited nodal sampling.
                
                    **CPT codes: 38500, 38525**
                    """,
            )
            alnd = st.selectbox(
                "Axillary Lymph Node Dissection",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Completion axillary lymph node dissection or removal of axillary lymphatic tissue to levels I–III for staging or oncologic control. 
                    
                    **CPT codes: 38740, 38745, 19302, 19305, 19306, 19200, 19220, 19162, 19240, 19307**
                    """,
            )
    # ================== Implant-Based Reconstruction/Autologous Reconstruction/Adjunct & Revision Procedures ===================
    with col4:
        with st.expander("**Implant-Based Reconstruction**", expanded=False):
            imm = st.selectbox(
                "Immediate Implant Reconstruction",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Placement of a permanent breast implant at the time of mastectomy during the same operative setting. 

                    **CPT code: 19340**
                    """,
            )
            delayed_cpt = st.selectbox(
                "Delayed Implant Reconstruction",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Placement of a permanent implant in a previously reconstructed or healed mastectomy site after the index mastectomy procedure.
                    
                    **CPT code: 19342**
                    """,
            )
            teinsertion = st.selectbox(
                "Tissue Expander Insertion",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Placement of a breast tissue expander during mastectomy or revision surgery as the first stage of staged implant reconstruction. (
                    
                    **CPT code: 19357**
                    """,
            )
            teexchange = st.selectbox(
                "Tissue Expander Exchange",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Exchange of a breast tissue expander for a permanent implant or alternative reconstructive modality as part of staged breast reconstruction. 

                    **CPT code: 11970**
                    """,
            )

        with st.expander("**Autologous Reconstruction**", expanded=False):
            free_cpt = st.selectbox(
                "Free Flap Breast Reconstruction",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Autologous reconstruction using free tissue transfer (e.g., DIEP, free TRAM), including microvascular anastomosis. 

                    **CPT code: 19364**
                    """,
            )
            lat = st.selectbox(
                "Latissimus Dorsi Flap Reconstruction",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Pedicled latissimus dorsi myocutaneous flap transfer to the breast, with or without implant placement. 

                    **CPT code: 19361**
                    """,
            )
            sintramp = st.selectbox(
                "Pedicled TRAM Flap Reconstruction",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Unilateral pedicled transverse rectus abdominis myocutaneous (TRAM) flap for breast reconstruction. 

                    **CPT code: 19367**
                    """,
            )
            bitram = st.selectbox(
                "Bipedicled TRAM Flap Reconstruction",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Bipedicled TRAM flap breast reconstruction performed in the same operative session. 

                    **CPT code: 19369**
                    """,
            )
            sinsuper = st.selectbox(
                "Supercharged or Enhanced Unilateral TRAM",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Pedicled TRAM flap reconstruction with vascular augmentation (“supercharging”) to improve perfusion. 

                    **CPT code: 19368**
                    """,
            )

        with st.expander("**Adjunct & Revision Procedures**", expanded=False):
            augprosimp = st.selectbox(
                "Prosthetic Breast Augmentation",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Placement of breast implants for cosmetic or reconstructive augmentation. 

                    **CPT code: 19325**
                    """,
            )
            masto = st.selectbox(
                "Mastopexy",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Breast lift procedure involving skin and parenchymal reshaping to correct ptosis.

                    **CPT code: 19316**
                    """,
            )
            brered = st.selectbox(
                "Breast Reduction",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Reduction mammaplasty to remove excess breast tissue and skin for symptomatic macromastia or asymmetry. 

                    **CPT code: 19318**
                    """,
            )
            fatgraft = st.selectbox(
                "Fat Grafting to the Breast",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Autologous fat harvest, processing, and injection for volume restoration or contour refinement. 

                    **CPT codes: 15771, 15772**
                    """,
            )
            revbreast = st.selectbox(
                "Revision of Reconstructed Breast",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Surgical revision of prior breast reconstruction to improve contour, symmetry, or address complications (e.g., capsulorrhaphy, pocket revision). 

                    **CPT code: 19380**
                    """,
            )
            adjtistrans = st.selectbox(
                "Adjacent Tissue Transfer",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Local tissue rearrangement or flap advancement/rotation for breast reconstruction or contour revision. 

                    **CPT codes: 14000–14999**
                    """,
            )
            npwt = st.selectbox(
                "Negative Pressure Wound Therapy",
                ["Yes", "No"],
                index=1,
                help="""
                    Application of negative pressure wound therapy systems to a breast surgical site for wound management or closure assistance. 

                    **CPT codes: 97605–97608**
                    """,
            )
            other_cpt = st.selectbox(
                "Other Reconstructive Techniques",
                [
                    "None",  # --> 0
                    "Unilateral",  # --> 1
                    "Bilateral",  # --> 2+
                ],
                index=0,
                help="""
                    Any breast reconstructive approach not captured in the above categories, including uncommon flaps or hybrid methods. 

                    **Various CPT codes not specified**
                    """,
            )

    # ================== Create input DF ===================
    input_data = pd.DataFrame(
        {
            ## Demographics ##
            "AGE": [age],
            "SEX": [util.transform_sex(sex)],
            "ETHNICITY_HISPANIC": [util.transform_hispanic(hispanic)],
            "RACE": [util.transform_race(race)],
            ## BMI ##
            "HEIGHT": [height],
            "WEIGHT": [weight],
            ## Comorbidities ##
            "SMOKE": [util.transform_yes_no(smoke)],
            "DIABETES": [util.transform_diab(diabetes)],
            "HXCOPD": [util.transform_yes_no(hxcopd)],
            "DYSPNEA": [util.transform_yes_no_unknown(dyspnea, "DYSPNEA")],
            "HXCHF": [util.transform_yes_no(hxchf)],
            "HYPERMED": [util.transform_yes_no(hypermed)],
            "DIALYSIS": [util.transform_yes_no(dialysis)],
            "RENAFAIL": [util.transform_yes_no_unknown(renafail, "RENAFAIL")],
            "BLEEDDIS": [util.transform_yes_no(bleed)],
            "WTLOSS": [util.transform_yes_no_unknown(wtloss, "WTLOSS")],
            "ASCITES": [util.transform_yes_no(ascites)],
            "STEROID": [util.transform_yes_no(steroid)],
            "DISCANCR": [util.transform_yes_no(discancr)],
            "VENTILAT": [util.transform_yes_no(vent)],
            "WNDINF": [util.transform_yes_no_unknown(wndinf, "WNDINF")],
            "ASACLAS": [util.transform_asa(asa_class)],
            "TRANSFUS": [util.transform_yes_no(transfus)],
            ## Pre-Op Lab (Blood)
            "PRALBUM": [pralbumin],
            "PRWBC": [prwbc],
            "PRHCT": [prhct],
            "PRPLATE": [prplate],
            # Op Chars
            "SURGINDICD": [util.transform_icd(surg_icd)],
            "OPTIME": [optime],
            "OPERYR": [operyr],
            "URGENCY": [util.transform_casetype(elect_surg)],
            "ANESTHES": [util.transform_anes(anesth)],
            "SURGSPEC": [util.transform_spec(surg_spec)],
            "INOUT": [util.transform_inout(inout)],
            ## Breast Resection Procedures
            "PARTIALCPT": [util.transform_ord_cpt(part_cpt)],
            "SUBSIMPLECPT": [util.transform_ord_cpt(sub_simp)],
            "RADICALCPT": [util.transform_ord_cpt(rad)],
            "MODIFIEDRADICALCPT": [util.transform_ord_cpt(modrad)],
            # Axillary Surgery
            "SNLBCPT": [util.transform_ord_cpt(slnb)],
            "ALNDCPT": [util.transform_ord_cpt(alnd)],
            ## Implant-Based Reconstruction
            "IMMEDIATECPT": [util.transform_ord_cpt(imm)],
            "DELAYEDCPT": [util.transform_ord_cpt(delayed_cpt)],
            "TEINSERTIONCPT": [util.transform_ord_cpt(teinsertion)],
            "TEEXPANDERCPT": [util.transform_ord_cpt(teexchange)],
            ## Autologous Recon
            "FREECPT": [util.transform_ord_cpt(free_cpt)],
            "LATCPT": [util.transform_ord_cpt(lat)],
            "SINTRAMCPT": [util.transform_ord_cpt(sintramp)],
            "BITRAMCPT": [util.transform_ord_cpt(bitram)],
            "SINTRAMSUPERCPT": [util.transform_ord_cpt(sinsuper)],
            ## Adjunct & Revision
            "AUGPROSIMPCPT": [util.transform_ord_cpt(augprosimp)],
            "MASTOCPT": [util.transform_ord_cpt(masto)],
            "BREASTREDCPT": [util.transform_ord_cpt(brered)],
            "FATGRAFTCPT": [util.transform_ord_cpt(fatgraft)],
            "REVRECBREASTCPT": [util.transform_ord_cpt(revbreast)],
            "ADJTISTRANSCPT": [util.transform_ord_cpt(adjtistrans)],
            "NPWTCPT": [util.transform_yes_no(npwt)],
            "OTHERRECONTECHCPT": [util.transform_ord_cpt(other_cpt)],
        }
    )
    ## Maps column name to unknown
    ### This order is hard-coded and matches order passed into imputer
    num_dict = {
        "AGE": {
            "Value": age,
            "Display Name": "Age",
            "round rule": lambda x: int(round(x)),
        },
        "HEIGHT": {
            "Value": height,
            "Display Name": "Height (in)",
            "round rule": lambda x: round(x, 2),
        },
        "WEIGHT": {
            "Value": weight,
            "Display Name": "Weight (lbs)",
            "round rule": lambda x: round(x, 2),
        },
        "PRALBUM": {
            "Value": pralbumin,
            "Display Name": "Albumin (g/dL)",
            "round rule": lambda x: round(x, 2),
        },
        "PRWBC": {
            "Value": prwbc,
            "Display Name": "White Blood Cell Count (*10^9/L)",
            "round rule": lambda x: round(x, 2),
        },
        "PRHCT": {
            "Value": prhct,
            "Display Name": "Hematocrit (%)",
            "round rule": lambda x: round(x, 2),
        },
        "PRPLATE": {
            "Value": prplate,
            "Display Name": "Platelet Count (*10^9/L)",
            "round rule": lambda x: round(x, 2),
        },
        "OPTIME": {
            "Value": optime,
            "Display Name": "Operation Time (min)",
            "round rule": lambda x: round(x, 2),
        },
        # this shouldnt run, dont give option
        ## just including for completeness
        "OPERYR": {
            "Value": operyr,
            "Display Name": "Operation Year",
            "round rule": lambda x: int(round(x)),
        },
    }
    # Get list of imputed vals
    imp_cols = [
        col_name for col_name, sub_dict in num_dict.items() if sub_dict["Value"] is None
    ]
    return input_data, num_dict, imp_cols


def create_shap_plot(shap_data, num_feats):
    """
    Create SHAP plot from pre-computed SHAP data.
    This function is fast and can be called whenever num_feats changes.
    """
    phi = shap_data["phi"]
    feat_names = shap_data["feat_names"]
    disp_row = shap_data["disp_row"]
    num_original_series = shap_data["num_original_series"]

    # sort by absolute impact
    order = np.argsort(-np.abs(phi))
    phi = phi[order]
    feat_names = feat_names[order]
    disp_row = disp_row[order]

    # choose k "explicit" features; aggregate the rest
    k = min(num_feats, len(phi))

    phi_main = phi[:k]
    feat_main = feat_names[:k]
    disp_main = disp_row[:k]

    phi_tail = phi[k:]

    if len(phi_tail) > 0:
        tail_sum = phi_tail.sum()
        tail_count = len(phi_tail)

        phi_all = np.concatenate([phi_main, np.array([tail_sum])])
        feat_all = np.concatenate(
            [feat_main, np.array([f"{tail_count} other features"])]
        )
        disp_all = np.concatenate([disp_main, np.array(["(aggregated)"])])
    else:
        phi_all = phi_main
        feat_all = feat_main
        disp_all = disp_main

    # NORMALIZE: Convert log-odds to percentage contributions
    total_abs_contribution = np.sum(np.abs(phi_all))

    if total_abs_contribution > 0:
        phi_top = (phi_all / total_abs_contribution) * 100.0
    else:
        phi_top = np.zeros_like(phi_all)

    feat_top = feat_all
    disp_top = disp_all

    # Calculate dynamic font sizes
    num_bars = len(phi_top)
    ytick_fontsize = max(9, min(14, 14 - (num_bars - 5) * 0.15))
    bar_label_fontsize = max(8, min(12, 12 - (num_bars - 5) * 0.15))

    fig, ax = plt.subplots(figsize=(12, 14))

    colors = np.where(phi_top >= 0, "#ff006e", "#118ab2")
    bars = ax.barh(range(len(phi_top)), phi_top, color=colors)

    ax.set_yticks(range(len(phi_top)))
    ax.set_yticklabels(
        [
            (
                f"{pretty_feature_name(name)}: {feature_value_label(name, val, num_original_series)}"
                if "(other features)" not in name
                and name != f"{len(phi_tail)} other features"
                else f"{name}"
            )
            for name, val in zip(feat_top, disp_top)
        ],
        fontsize=ytick_fontsize,
    )

    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    ax.set_xlabel(
        "Relative contribution to risk prediction (%)",
        fontweight="bold",
        fontsize=15,
    )
    ax.set_ylabel(
        f"Top {k} factors ranked by impact on risk",
        fontweight="bold",
        fontsize=15,
    )
    ax.set_title(
        "What’s driving this patient’s predicted risk?",
        fontweight="bold",
        pad=15,
        fontsize=20,
    )

    values = phi_top
    bar_lengths = np.array(values)

    max_right = np.max(bar_lengths[bar_lengths > 0], initial=0.0)
    max_left = np.min(bar_lengths[bar_lengths < 0], initial=0.0)

    value_range = max_right - max_left

    if value_range > 0:
        padding_pct = 0.15
        padding = value_range * padding_pct
    else:
        max_abs = max(abs(max_right), abs(max_left))
        padding = max(0.5, max_abs * 0.2)

    x_min = max_left - padding
    x_max = max_right + padding

    if x_max <= x_min:
        center = (max_left + max_right) / 2
        half_width = 0.5
        x_min, x_max = center - half_width, center + half_width

    ax.set_xlim(x_min, x_max)

    for bar, value in zip(bars, values):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ha = "left" if x >= 0 else "right"

        ax.text(
            x,
            y,
            f"{value:+.1f}%",
            va="center",
            ha=ha,
            fontsize=bar_label_fontsize,
            color="black",
        )

    red_patch = mpatches.Patch(color="#ff006e", label="Increases predicted risk")
    blue_patch = mpatches.Patch(color="#118ab2", label="Decreases predicted risk")

    ax.legend(
        handles=[red_patch, blue_patch],
        loc="best",
        frameon=True,
        title="Direction of effect",
        fontsize=13,
        title_fontsize=15,
    )
    plt.gca().invert_yaxis()
    plt.tight_layout(pad=2.0)
    return fig


def check_filter_cols(input_data):
    vars_to_check = [
        "partialCPT",
        "subsimpleCPT",
        "radicalCPT",
        "modifiedRadicalCPT",
        "immediateCPT",
        "delayedCPT",
        "teinsertionCPT",
        "teexpanderCPT",
        "freeCPT",
        "latCPT",
        "SinTramCPT",
        "SinTramSuperCPT",
        "BiTramCPT",
        "MastoCPT",
        "BreastRedCPT",
        "FatGraftCPT",
        "AdjTisTransCPT",
        "AugProsImpCPT",
        "OtherReconTechCPT",
        "RevRecBreastCPT",
    ]
    for var in vars_to_check:
        if input_data[var.upper()].iloc[0] != "0":
            return True
    return False


def show_clinical_results(display_name, folder_name, input_data):
    """
    Render clinical output.

    Params
    ------
    display_name: str
        Full name of outcome displayed on interface
    folder_name: str
        Outcome abbreviation
    input_data: pd.DataFrame
        User input single patient data

    """
    with st.expander(f"📊 {display_name}", expanded=False):
        try:
            # ================== Get model output ===================
            # Load model and preprocessor
            model, preprocessor, explainer = load_model_pipeline_explainer(folder_name)
            ## Preprocess
            feature_names = preprocessor.get_feature_names_out()
            data_transformed = np.array(preprocessor.transform(input_data))
            processed_data = pd.DataFrame(data_transformed, columns=feature_names)
            processed_data = remove_prefix(processed_data)
            for col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col])

            # Compute hash from processed_data (after transformation)
            hash_string = f"{folder_name}_{processed_data.to_csv()}"
            processed_data_hash = hashlib.md5(hash_string.encode()).hexdigest()
            ## predict
            probabilities = model.predict_proba(processed_data)[:, 1]

            # Extract scalar probability
            if probabilities.ndim == 1:
                prob_positive = float(probabilities[0])
            else:
                # output is 2D
                prob_positive = float(probabilities[0, 1])
            ## Get all test output
            all_probs, all_labels = util.load_population_probs(folder_name)
            tot_patients = len(all_labels)
            # ================== Display Results ===================
            # ========== ROW 1: Risk Assessment ==========
            st.markdown("## Risk Assessment")

            col_a1, col_b1 = st.columns(2)

            with col_a1:
                # Display appropriate risk category
                bin_thresholds = util.load_bin_thresholds(folder_name)
                risk, color, color_code = util.get_risk_category(
                    prob_positive, folder_name
                )
                st.metric(label="Risk Category", value=f"{color} {risk}")

                # Display bin context
                labels = ["Very Low", "Low", "Moderate", "High"]
                cutoffs = [0.0] + list(bin_thresholds) + [1.0]
                bin_idx = np.digitize([prob_positive], cutoffs[1:])[0]
                bin_occur_rate_list = util.bin_occur_rates(folder_name, bin_thresholds)

                st.markdown(
                    f"In our independent test cohort of {tot_patients:,} breast surgery patients, <b>~{bin_occur_rate_list[bin_idx]:.2%}</b> of those with similar characteristics developed this complication",
                    unsafe_allow_html=True,
                )

            with col_b1:
                # RISK PLOT
                fig = plot_risk_bins(
                    bin_occur_rate_list,
                    bin_idx,
                    folder_name,
                    color_code,
                )
                st.pyplot(fig, bbox_inches="tight")
                plt.close()

            # ========== ROW 2: Feature Contributions ==========
            st.markdown("### What Drives This Patient's Risk")

            col_a2, col_b2 = st.columns(2)

            with col_a2:
                model_used = CHOSEN_MODEL_DICT[folder_name]
                explainer_used = get_explainer_name(model_used)
                st.info(
                    """
                    ***How to read this chart***

                    - These bars explain why ***this specific patient*** received their particular risk estimate, not which factors matter most across all patients.
                    - Each bar represents one patient factor (diagnosis, operative details, comorbidities, etc.).
                    - Bars to the **right** (pink) increased this patient’s predicted risk; bars to the **left** (blue) decreased it.
                    - Longer bars mean that factor had **more influence** on this patient’s prediction.
                    - Factors are ranked from most influential (top) to least influential (bottom).

                    **About the percentages**

                    The percentages show each factor’s **relative share** of the total explanation for why this patient 
                    differs from the average patient (absolute contributions sum to 100%). For example, if "Setting: 
                    Inpatient" shows "-12%", this factor accounts for 12% of the total model influence and decreases risk.

                    """
                )
                with st.expander("**Important Considerations**", expanded=False):
                    st.markdown(
                        """
                        **Why some factors may appear high on this chart**

                        - This chart explains how the model estimated risk *for this individual patient*. Contributions are SHAP-derived 
                        values that reflect learned model relationships from the training data (including interactions), not cause-and-effect.

                        - A value of *“None”* for a procedure indicates that the procedure was marked as **not performed** 
                        (as opposed to unilateral or bilateral). The model compares patients without the procedure to similar 
                        patients who did receive it, which can increase or decrease the predicted risk of the patient depending 
                        on observed outcome patterns.

                        - A value of *“Unknown”* means the information was **not documented or unavailable**, not that the 
                        condition was absent. Missing documentation can still influence predictions because it occurred systematically 
                        across certain time periods rather than at random.

                        - **Year of operation** may appear important because it captures broader temporal trends—such as evolving 
                        surgical techniques, perioperative care pathways, patient selection, or documentation practices.

                        - Unexpected results should be interpreted as a prompt to consider clinical context (e.g., documentation patterns, 
                        case complexity, or interacting factors) and to apply clinical judgment, rather than as evidence that a factor 
                        directly causes the outcome.
                        """
                    )
                with st.expander(
                    "**Technical notes for interested users**", expanded=False
                ):
                    st.markdown(
                        f"""
                        - Feature contributions are computed using SHAP (SHapley Additive exPlanations), which attributes 
                        the model’s prediction to individual feature values for this specific patient.

                        - This particular plot was generated using SHAP's *{explainer_used}*

                        - The explanation is *contrastive*: it explains how this patient differs from the baseline 
                        (the average prediction over our test cohort).

                        - Explanation values are computed in the log-odds space from the raw, uncalibrated model, then 
                        are normalized such that absolute values sum to 100%, maintaining the sign, ranking, and relative 
                        magnitude of each feature's impact 

                        - Calibrated risk predictions used to allocate patients into risk categories apply a monotonic transformation 
                        (Platt scaling) on raw model output that preserves the feature importance ranking and direction displayed in this plot.
                        """
                    )

            with col_b2:
                # SHAP feature count input
                n_feats = st.number_input(
                    "Number of top features to display",
                    min_value=5,
                    max_value=input_data.shape[1],
                    value=10,
                    step=1,
                    key=f"n_feats_{folder_name}",
                )
                st.markdown("")
                # n_feats = 10
                # SHAP plot
                # Compute SHAP data once (cached)
                shap_data = compute_shap_data(
                    explainer,
                    processed_data,
                    preprocessor,
                    processed_data_hash,
                    folder_name,
                )
                # Create plot (fast, regenerates only when n_feats changes)
                shap_fig = create_shap_plot(shap_data, n_feats)
                st.pyplot(shap_fig, bbox_inches="tight")
                plt.close(shap_fig)

            # ================== Dropdown to Show Model Details ===================
            st.markdown("---")
            with st.expander(f"**Extra information on model output**", expanded=False):
                show_model_details(
                    display_name,
                    folder_name,
                    prob_positive,
                )

        except Exception as e:
            st.error(f"Error predicting {display_name}: {str(e)}")


#################### EXTRA INFO ON MODEL ####################
def get_full_model_name(model_abrv):
    if model_abrv == "xgb":
        return "XGBoost"
    elif model_abrv == "lgbm":
        return "LightGBM"
    elif model_abrv == "stack":
        return "Stacked Generalization"
    elif model_abrv == "lr":
        return "Logistic Regression"
    elif model_abrv == "nn":
        return "Neural Network"
    else:
        raise ValueError(f"Unrecognized model name: {model_abrv}")


def show_model_details(display_name, outcome_abrv, prob_positive):
    """
    Display advanced model information: raw output, percentiles, thresholds, and SHAP.

    Params
    ------
    display_name: str
        Full name of outcome
    outcome_abrv: str
        Outcome abbreviation
    prob_positive: float
        Model prediction probability output for input data
    """
    model_abrv = CHOSEN_MODEL_DICT[outcome_abrv]
    chosen_model = get_full_model_name(model_abrv)
    st.subheader(f"{display_name} Model Details")
    # Load population data
    all_probs, all_labels = util.load_population_probs(outcome_abrv)
    bin_thresholds = util.load_bin_thresholds(outcome_abrv)
    # === Raw Model Output & Percentiles ===
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Calibrated Model Output")
        if outcome_abrv == "mort":
            value = f"{prob_positive:.3%}"
        else:
            value = f"{prob_positive:.2%}"
        st.metric(
            label="Model-assigned risk score",
            value=value,
            delta=None,
        )
        with st.expander("ℹ️ How to interpret this score"):
            st.write(
                "Given the limits of probability calibration for this model, the calibrated output is used primarily to place patients into risk categories. It should not be taken as an exact individual probability, but interpreted alongside the percentiles and the observed complication rates in each risk category."
            )
        st.markdown("##### Model Used:")
        st.markdown(f"##### {chosen_model}")

    with col_b:
        st.markdown("#### Risk Score Percentiles")

        # Overall percentile
        n_overall = len(all_probs)
        overall_pctile = (all_probs < prob_positive).mean() * 100

        # Percentile among patients WITHOUT the outcome
        neg_patients = all_probs[all_labels == 0]
        n_neg = len(neg_patients)
        neg_pctile = (neg_patients < prob_positive).mean() * 100

        # Percentile among patients WITH the outcome
        pos_patients = all_probs[all_labels == 1]
        n_pos = len(pos_patients)
        pos_pctile = (pos_patients < prob_positive).mean() * 100
        st.markdown(
            f"**{overall_pctile:.2f}%** of all patients (n={n_overall:,}) received a lower risk score",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**{neg_pctile:.2f}%** of patients who did not develop this complication (n={n_neg:,}) received a lower risk score",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**{pos_pctile:.2f}%** of patients who developed this complication (n={n_pos:,}) received a lower risk score",
            unsafe_allow_html=True,
        )
        with st.expander("ℹ️ How to interpret these percentiles"):
            st.write(
                f"Model output is compared with {chosen_model}'s predicted scores "
                f"in the independent test cohort (n={n_overall:,}). These percentiles show where this patient’s score falls relative to all patients, to those who did not develop the outcome, and to those who did."
            )
    # === Risk Bin Thresholds ===
    st.markdown("---")
    st.markdown("#### Risk Category Thresholds")
    col_a, col_b = st.columns(2)
    with col_a:
        labels = ["Very Low", "Low", "Moderate", "High"]
        cutoffs = [0.0] + list(bin_thresholds) + [1.0]
        bin_idx = np.digitize([prob_positive], cutoffs[1:])[0]
        threshold_data = []
        for i, lab in enumerate(labels):
            if outcome_abrv == "mort":
                threshold_range = f"{cutoffs[i]:.3%} – {cutoffs[i+1]:.3%}"
            else:
                threshold_range = f"{cutoffs[i]:.2%} – {cutoffs[i+1]:.2%}"
            threshold_data.append(
                {"Risk Category": lab, "Model Output Range": threshold_range}
            )
        threshold_df = pd.DataFrame(threshold_data)

        # Styler to bold active bin
        def highlight_active(row):
            if row.name == bin_idx:
                return ["font-weight: bold;" for _ in row]
            else:
                return ["" for _ in row]

        styled = threshold_df.style.apply(highlight_active, axis=1)

        # Generate HTML to bold selected bin
        html_table = styled.to_html(index=False)
        # render
        st.markdown(html_table, unsafe_allow_html=True)
    with col_b:
        st.markdown(
            f"Risk categories are defined using cutoffs taken from the {chosen_model} model’s predicted scores in the training and validation cohorts (n=627,070) for the {display_name} outcome. These cutoffs follow a logarithmic scale so that higher‑risk ranges are more finely separated than very low‑risk ranges."
        )

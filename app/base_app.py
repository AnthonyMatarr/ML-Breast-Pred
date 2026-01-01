## Append path to root
import sys
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_PATH))
## Other imports
import streamlit as st
import display_functions as display
from app.config import OUTCOMES


def init_session_state():
    default_keys = {
        "predictions_made": False,
        "last_input_hash": None,
        "selected_outcomes": [],
        "input_data": None,
        "num_dict": None,
        "imp_cols": [],
    }
    for k, v in default_keys.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    st.set_page_config(
        page_title="PRO-BREAST",
        page_icon="🏥",
        layout="wide",
    )
    init_session_state()
    st.title(
        "PRO-BREAST: Post-resection and Reconstruction Outcome prediction for Breast surgery"
    )
    st.markdown(
        "Predict 30-day complications after mastectomy alone, mastectomy with immediate reconstruction, and delayed breast reconstruction/revision."
    )
    st.info(
        "Adjust all fields to match your patient. Default values are set arbitrarily. To reset to default values, refresh the page. "
    )
    #################################################################################################################
    ################################################### Side Bar ####################################################
    #################################################################################################################
    outcome_info_dict = {
        "surg": """
            A postoperative “surgical complication” is present if a patient has any of the following within 30 days of surgery, including:

            &nbsp;&nbsp;&nbsp;&nbsp;•**Superficial incisional SSI** (skin or subcutaneous tissue only). 

            &nbsp;&nbsp;&nbsp;&nbsp;•**Deep incisional SSI** (involving fascia or muscle of the incision). 

            &nbsp;&nbsp;&nbsp;&nbsp;•**Organ/space SSI** (infection involving any organ or space opened or manipulated during the operation, excluding the incision itself). 

            &nbsp;&nbsp;&nbsp;&nbsp;•**Wound disruption/dehiscence** requiring clinical intervention

            &nbsp;&nbsp;&nbsp;&nbsp;•**Postoperative bleeding event** requiring transfusion of packed red blood cells or whole blood within 72 hours of the end of surgery, recorded when transfusion is given to treat or in response to postoperative hemorrhage.
            """,
        "med": """
            Composite variable capturing major postoperative medical complications occurring within 30 days of surgery, including:

            &nbsp;&nbsp;&nbsp;&nbsp;•**Postoperative pneumonia**

            &nbsp;&nbsp;&nbsp;&nbsp;•**Unplanned reintubation**

            &nbsp;&nbsp;&nbsp;&nbsp;•**Failure to wean from mechanical ventilation** >48 hours after surgery

            &nbsp;&nbsp;&nbsp;&nbsp;•**Urinary tract infection**

            &nbsp;&nbsp;&nbsp;&nbsp;•**Cerebrovascular accident / stroke** with neurologic deficit

            &nbsp;&nbsp;&nbsp;&nbsp;•**Cardiac arrest** requiring CPR

            &nbsp;&nbsp;&nbsp;&nbsp;•**Myocardial infarction**

            &nbsp;&nbsp;&nbsp;&nbsp;•**Sepsis**

            &nbsp;&nbsp;&nbsp;&nbsp;•**Septic shock**

            &nbsp;&nbsp;&nbsp;&nbsp;•**Acute renal insufficiency**
            
            &nbsp;&nbsp;&nbsp;&nbsp;•**Acute renal failure** requiring dialysis
            """,
        "mort": """
            Death from any cause within 30 days of the principal operation, whether occurring in or out of the index hospital, including after discharge or transfer. 
            """,
        "reop": """
            Any unplanned return to the operating room for a surgical procedure related to the index or concurrent procedure within 30 days, at any facility; planned/staged procedures are excluded. 
            """,
        "vte": """
            A composite variable representing the occurrence of **deep venous thrombosis** or **pulmonary embolism** within 30 days postoperatively.
            """,
    }
    st.sidebar.header("Select Outcomes to Predict")
    selected_outcomes = []
    for display_name, folder_name in OUTCOMES.items():
        if st.sidebar.checkbox(
            display_name, value=True, help=outcome_info_dict[folder_name]
        ):
            selected_outcomes.append((display_name, folder_name))

    if not selected_outcomes:
        st.warning("Please select at least one outcome to predict")
        return

    ############# Input Section #############
    input_data, num_dict, imp_cols = display.get_input_data()
    current_input_hash = hash(input_data.to_json())
    # If inputs changed compared to the last prediction, reset predictions
    if (
        st.session_state.last_input_hash is not None
        and current_input_hash != st.session_state.last_input_hash
    ):
        st.session_state.predictions_made = False

    ############# Output Section #############
    # Button triggers prediction and stores results in session state
    if st.button("Predict Outcomes", type="primary", key="pred_btn"):
        if display.check_filter_cols(input_data):
            st.session_state.predictions_made = True
            st.session_state.selected_outcomes = selected_outcomes
            st.session_state.input_data = input_data
            st.session_state.num_dict = num_dict
            st.session_state.imp_cols = imp_cols
            st.session_state.last_input_hash = current_input_hash
        else:
            st.error(
                """
                At least one field from any of the following categories must be selected before predicting outcomes:
                - ***Breast Resection Procedures*** 
                - ***Implant-Based Reconstruction***
                - ***Autologous Reconstruction***
                - ***Adjunct & Revision Procedures***
                    - Does not include the *Negative Pressure Wound Therapy*
                """
            )
            st.session_state.predictions_made = False

    # Display results if predictions have been made
    if st.session_state.predictions_made:
        st.header("Prediction Results")

        # Process each selected outcome
        for display_name, folder_name in selected_outcomes:
            display.show_clinical_results(display_name, folder_name, input_data)

        # Display imputed values
        if len(st.session_state.imp_cols) > 0:
            st.header("Imputed Values")
            st.info(
                """
            When continuous patient data is missing, the modeling pipeline uses an iterative regression-based imputation method to estimate those values.
            It models each incomplete variable using other available patient characteristics, ***defined in each outcome's respective train cohort,*** and refines these estimates over several rounds.
            Imputed values are statistical estimates, not actual measurements.
            """
            )
            for display_name, folder_name in selected_outcomes:
                display.show_imputed(
                    display_name, folder_name, input_data, num_dict, imp_cols
                )


if __name__ == "__main__":
    main()

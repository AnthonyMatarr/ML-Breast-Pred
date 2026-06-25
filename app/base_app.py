## Append path to root
import sys, os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"  # for scikit-learn/pytorch MacOS issue
BASE_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_PATH))
## Other imports
import streamlit as st
import display_functions as display
from app.app_config import OUTCOMES


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
    with st.expander("**INTENDED PATIENT PROFILE**", expanded=False):
        st.warning(
            "The models underlying this risk calculator were developed on a cohort meeting the following requirement. Thus, the use"
            "of this application for risk prediction is only recommended for patients who meet ***BOTH*** of the following criteria:"
        )
        col1, col2 = st.columns(2)
    with col1:
        with st.expander("**Cancer Diagnosis**", expanded=False):
            st.markdown("""
                Primary indication for the breast operation, categorized using mutually exclusive NSQIP-based ICD-9/ICD-10 codes, including:

                    • Carcinoma in Situ (ICD-9 233; ICD-10 D05)

                    • Malignant Neoplasm (174, 175, C50)
            """)
    with col2:
        with st.expander("**Reconstruction Procedures**", expanded=False):
            st.markdown("""
                At least one of the following procedure designations, grouped for convenience:
                
                **Resection Procedures**
                    
                    • Partial Mastectomy
                        • CPT codes: 19301, 19302, 19125, 19120, 19160, 19162
                    • Simple (Total) Mastectomy
                        • CPT codes: 19303, 19304, 19180, 19182
                    • Radical Mastectomy
                        • CPT codes: 19305, 19306, 19200, 19220
                    • Modified Radical Mastectomy
                        • CPT codes: 19307, 19240
                        
                **Implant-based Procedures**
                        
                    • Immediate Implant Reconstruction
                        • CPT code: 19340
                    • Delayed Implant Reconstruction
                        • CPT code: 19342
                    • Tissue Expander Insertion
                        • CPT code: 19357
                    • Tissue Expander Exchange
                        • CPT code: 11970

                    
                **Autologous Procedures**    
                        
                    • Free Flap Breast Reconstruction
                        • CPT code: 19364
                    • Latissimus Dorsi Flap Reconstruction
                        • CPT code: 19361
                    • Pedicled TRAM Flap Reconstruction
                        • CPT code: 19367
                    • Supercharged or Enhanced Unilateral TRAM
                        • CPT code: 19369
                    • Bipedicled TRAM Flap Reconstruction
                        • CPT code: 19368
                    
                **Adjunct and Revision Procedures**    
                        
                    • Prosthetic Breast Augmentation
                        • CPT code: 19325
                    • Mastopexy
                        • CPT code: 19316
                    • Breast Reduction
                        • CPT code: 19318
                    • Fat Grafting to the Breast
                        • CPT codes: 15771, 15772
                    • Revision of Reconstructed Breast
                        • CPT code: 19380
                    • Adjacent Tissue Transfer
                        • CPT codes: 14000–14999
                    • Other Reconstructive Techniques
                        • Various CPT codes not specified
            """)
    #################################################################################################################
    ################################################### Side Bar ####################################################
    #################################################################################################################
    outcome_info_dict = {
        "SERIOUS": """ 
            Patients classified as having a serious complication experienced at least one of the following within 30 days postoperatively: 
            
            any occurrence of Cardiac arrest, Myocardial infarction, Pneumonia, Progressive renal insufficiency, Acute renal failure, 
            Pulmonary embolism, Venous thrombosis, Unplanned reoperation, Deep incisional surgical site infection (SSI), 
            Organ space SSI, Sepsis, Unplanned intubation, Urinary tract infection, or Dehiscence
        """,
        "ANY": """ 
            Patients classified as having any complication experienced at least one of the following within 30 days postoperatively: 
            
            Cardiac arrest, Myocardial infarction, Pneumonia, *Progressive renal insufficiency, Acute renal failure, Pulmonary embolism, 
            Venous thrombosis, Unplanned reoperation, Deep incisional SSI, Organ space SSI, Sepsis, Unplanned intubation, 
            Urinary tract infection, Dehiscence, Superficial SSI, Ventilator > 48H, or Stroke
        """,
        "SSI": """
            A postoperative "surgical site infection" is present if a patient has any of the following within 30 days of surgery, including:
            
            &nbsp;&nbsp;&nbsp;&nbsp;•**Superficial incisional SSI** (skin or subcutaneous tissue only). 
            
            &nbsp;&nbsp;&nbsp;&nbsp;•**Deep incisional SSI** (involving fascia or muscle of the incision). 
            
            &nbsp;&nbsp;&nbsp;&nbsp;•**Organ/space SSI** (infection involving any organ or space opened or manipulated during the operation, excluding the incision itself).
        """,
        "UNPLNREOP": """
            Any unplanned return to the operating room for a surgical procedure related to the index or concurrent
            procedure within 30 days, at any facility; planned/staged procedures are excluded.
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
        st.session_state.predictions_made = True
        st.session_state.selected_outcomes = selected_outcomes
        st.session_state.input_data = input_data
        st.session_state.num_dict = num_dict
        st.session_state.imp_cols = imp_cols
        st.session_state.last_input_hash = current_input_hash

    # Display results if predictions have been made
    if st.session_state.predictions_made:
        st.header("Prediction Results")

        # Process each selected outcome
        for display_name, folder_name in selected_outcomes:
            display.show_clinical_results(display_name, folder_name, input_data)

        # Display imputed values
        if len(st.session_state.imp_cols) > 0:
            st.header("Imputed Values")
            st.info("""
            When patient data is missing, the modeling pipeline uses an iterative regression-based imputation method to estimate those values.
            It models each incomplete variable using other available patient characteristics, ***defined in each outcome's respective train set,*** and refines these estimates over several rounds.
            Imputed values are statistical estimates, not actual measurements.
            """)
            for display_name, folder_name in selected_outcomes:
                display.show_imputed(
                    display_name, folder_name, input_data, num_dict, imp_cols
                )


if __name__ == "__main__":
    main()

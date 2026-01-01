from shutil import rmtree
import numpy as np
import pandas as pd
import warnings
from src.data_utils import get_feature_lists


##############################################################################
################################### CLEAN ####################################
##############################################################################
def combine_columns(row):
    """Combine 5 columns with hierarchy: Yes > No > NaN"""
    # Check if any value is "Yes"
    if row.isin(["Yes", "Ye"]).any():
        return "Yes"
    # Check if any value is "No"
    elif (row == "No").any():
        return "No"
    # All values are "NULL" or NaN
    else:
        return np.nan


def merge_dfs(data_dict, verbose=False):
    """
    Merges NSQIP dataframes from 2008-2023, normalizing values to append vertically

    Parameters
    ----------
    data_dict: dict
        Dictionary mapping NSQIP file name to pandas df
    """
    data_dict_clean = {}
    no_codes_dict = {}
    ######################################
    ################ 2008 ################
    ######################################
    # {'READ', 'DISCHDEST', 'UNPLREAD'}
    temp_df = data_dict["NSQIP_08_cpt"].copy()
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    temp_df.rename(columns={"RETURNOR": "UnplReOp"}, inplace=True)
    ## Readmission
    temp_df["ReAd"] = None
    ## Unplanned Readmission
    temp_df["UnplReAd"] = None
    ## Discharge Dest
    temp_df["DISCHDEST"] = None
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["08"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["08"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["08"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2009 ################
    ######################################
    # {'READ', 'DISCHDEST', 'UNPLREAD'}
    temp_df = data_dict["NSQIP_09_cpt"].copy()
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    temp_df.rename(columns={"RETURNOR": "UnplReOp"}, inplace=True)
    ## Readmission
    temp_df["ReAd"] = None
    ## Unplanned Readmission
    temp_df["UnplReAd"] = None
    ## Discharge Dest
    temp_df["DISCHDEST"] = None
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["09"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["09"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["09"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2010 ################
    ######################################
    # {'READ', 'DISCHDEST', 'UNPLREAD'}
    temp_df = data_dict["NSQIP_10_cpt"].copy()
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    temp_df.rename(columns={"RETURNOR": "UnplReOp"}, inplace=True)
    ## ReAd
    temp_df["ReAd"] = None
    ## Unplanned ReAd
    temp_df["UnplReAd"] = None
    ## Discharge Dest
    temp_df["DISCHDEST"] = None
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["10"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["10"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["10"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2011 ################
    ######################################
    # {'UNPLREAD'}
    temp_df = data_dict["NSQIP_11_cpt"].copy()
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unpl_read_cols = ["RETURNOR", "REOPERATION"]
    temp_df["UnplReOp"] = temp_df[unpl_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unpl_read_cols, axis=1, inplace=True)
    ## Readmission
    temp_df.rename(columns={"READMISSION": "ReAd"}, inplace=True)
    ## Unplanned Readmission
    temp_df["UnplReAd"] = None
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["11"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["11"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["11"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2012 ################
    ######################################
    temp_df = data_dict["NSQIP_12_cpt"].copy()
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = [
        "RETURNOR",
        "REOPERATION",
        "REOPERATION1",
        "REOPERATION2",
        "REOPERATION3",
    ]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION",
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["12"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["12"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["12"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2013 ################
    ######################################
    temp_df = data_dict["NSQIP_13_cpt"].copy()
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = [
        "RETURNOR",
        "REOPERATION",
        "REOPERATION1",
        "REOPERATION2",
        "REOPERATION3",
    ]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION",
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["13"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["13"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["13"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2014 ################
    ######################################
    temp_df = data_dict["NSQIP_14_cpt"].copy()
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = [
        "RETURNOR",
        "REOPERATION",
        "REOPERATION1",
        "REOPERATION2",
        "REOPERATION3",
    ]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION",
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)

    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["14"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["14"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["14"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2015 ################
    ######################################
    temp_df = data_dict["NSQIP_15_cpt"].copy()
    temp_df.rename(columns={"BLEEDIS": "BLEEDDIS"}, inplace=True)
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)

    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["15"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["15"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["15"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2016 ################
    ######################################
    temp_df = data_dict["NSQIP_16_cpt"].copy()
    temp_df.rename(columns={"BLEEDIS": "BLEEDDIS"}, inplace=True)
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)

    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["16"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["16"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["16"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2017 ################
    ######################################
    temp_df = data_dict["NSQIP_17_cpt"].copy()
    temp_df.rename(columns={"BLEEDIS": "BLEEDDIS"}, inplace=True)
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)

    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["17"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["17"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["17"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2018 ################
    ######################################
    temp_df = data_dict["NSQIP_18_cpt"].copy()
    temp_df.rename(columns={"BLEEDIS": "BLEEDDIS"}, inplace=True)
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)

    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["18"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["18"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["18"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2019 ################
    ######################################
    temp_df = data_dict["NSQIP_19_cpt"].copy()
    temp_df.rename(columns={"BLEEDIS": "BLEEDDIS"}, inplace=True)
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)

    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["19"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["19"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["19"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2020 ################
    ######################################
    temp_df = data_dict["NSQIP_20_cpt"].copy()
    temp_df.rename(columns={"BLEEDIS": "BLEEDDIS"}, inplace=True)
    temp_df.rename(columns={"EMERGNCY": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)

    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["20"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["20"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["20"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2021 ################
    ######################################
    # {'RENAINSF', 'WTLOSS', 'WNDINF', 'RENAFAIL', 'DYSPNEA'}
    temp_df = data_dict["NSQIP_21_cpt"].copy()
    temp_df.rename(columns={"BLEEDIS": "BLEEDDIS"}, inplace=True)
    temp_df.rename(columns={"CASETYPE": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)
    ## Not included
    temp_df["RENAINSF"] = None
    temp_df["RENAFAIL"] = None
    temp_df["WTLOSS"] = None
    temp_df["WNDINF"] = None
    temp_df["DYSPNEA"] = None
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["21"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["21"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["21"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2022 ################
    ######################################
    # {'WNDINF', 'WTLOSS', 'DYSPNEA'}
    temp_df = data_dict["NSQIP_22_cpt"].copy()
    temp_df.rename(columns={"CASETYPE": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)
    # Not included
    temp_df["WTLOSS"] = None
    temp_df["WNDINF"] = None
    temp_df["DYSPNEA"] = None
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["22"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["22"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["22"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2023 ################
    ######################################
    # {'WNDINF', 'WTLOSS', 'DYSPNEA'}
    temp_df = data_dict["NSQIP_23_cpt"].copy()
    temp_df.rename(columns={"CASETYPE": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)
    # Not included
    temp_df["WTLOSS"] = None
    temp_df["WNDINF"] = None
    temp_df["DYSPNEA"] = None
    temp_df.reset_index(drop=True, inplace=True)
    data_dict_clean["23"] = temp_df
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["23"] = temp_df.drop(temp_drop_cols, axis=1)
    if verbose:
        print(no_codes_dict["23"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ######################################
    ################ 2024 ################
    ######################################
    # {'WNDINF', 'WTLOSS', 'DYSPNEA'}
    temp_df = data_dict["NSQIP_24_cpt"].copy()
    temp_df.rename(columns={"CASETYPE": "Urgency"}, inplace=True)
    ## Unplanned ReOp
    unplanned_reop_cols = ["RETURNOR", "REOPERATION1", "REOPERATION2", "REOPERATION3"]
    temp_df["UnplReOp"] = temp_df[unplanned_reop_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    temp_df["ReAd"] = temp_df[read_cols].apply(combine_columns, axis=1)
    temp_df.drop(read_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]
    temp_df["UnplReAd"] = temp_df[unplanned_read_cols].apply(combine_columns, axis=1)
    temp_df.drop(unplanned_read_cols, axis=1, inplace=True)
    # Not included
    temp_df["WTLOSS"] = None
    temp_df["WNDINF"] = None
    temp_df["DYSPNEA"] = None
    temp_df.reset_index(drop=True, inplace=True)
    ## DROP codes
    temp_drop_cols = [col for col in temp_df if ("PODIAG" in col or "CPT" in col)]
    no_codes_dict["24"] = temp_df.drop(temp_drop_cols, axis=1)
    data_dict_clean["24"] = temp_df
    if verbose:
        print(no_codes_dict["24"].shape)
    else:
        print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ###### ENSURE we did things right #####
    try:
        assert len(data_dict_clean) == len(data_dict)
        assert len(data_dict_clean) == len(no_codes_dict)
    except AssertionError:
        print("Dicts do not match in size...")
        print(f" New length w/ codes: {len(data_dict_clean)}")
        print(f" New length w/o codes: {len(no_codes_dict)}")
        print(f" OG length: {len(data_dict)}")
        raise AssertionError
    for year1, df1 in no_codes_dict.items():
        try:
            assert df1.shape[1] == 68
        except AssertionError:
            raise AssertionError(f"Expected 68 rows, got {df1.shape[1]} instead")
        for year2, df2 in no_codes_dict.items():
            cols_1 = set(df1.columns)
            cols_2 = set(df2.columns)
            assert cols_2 - cols_1 == set()
            assert cols_1 - cols_2 == set()
    ##### Combine
    combined_df_no_codes = pd.concat(no_codes_dict.values(), ignore_index=True)
    combined_df_w_codes = pd.concat(data_dict_clean.values(), ignore_index=True)
    print(f"Combined Shape No Codes: {combined_df_no_codes.shape}")
    print(f"Combined Shape With Codes: {combined_df_w_codes.shape}")
    return combined_df_no_codes, combined_df_w_codes


def clean_dfs(
    *_,
    combined_df,
    replace_dict,
    na_drop_cols,
    recon_cols,
    mast_cols,
    surg_ind_cols,
    num_cols,
    drop_cols,
) -> pd.DataFrame:
    """
    Implements various cleaning steps to an NSQIP combined df

    Raises
    ------
    ValueError:
        if a positional argument is passed
    Parameters
    ----------
    combined_df: pandas dataframe
        Tabular dataframe containing NSQIP data combined accross years (result of merge_dfs)
    replace_dict: dict{dict}
        Dictionary mapping column names to a sub-dictionary mapping old instance names to new ones
        Example: {"SEX": {"non-bi": "non-binary", "fem": "female"}, ...}
    na_drop_cols: list[str]
        Specify names of columns for which entries with NA values should be dropped
    recon_cols: list[str]
        Specify names of derived reconstruction variables (CPT)
        Used to create timing of surgery column
    mast_cols: list[str]
        Specify names of derived mastectomny variables (CPT)
        Used to create timing of surgery column
    surg_ind_cols: list[str]
        Specify names of derived surgical indicator variables (ICD)
        Combined into a single feature bc of mututal exclusivity

    Returns
    -------
    cleaned pandas dataframe
    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments!")
    df_combined = combined_df.copy()
    df_combined.columns = df_combined.columns.str.upper()
    ########################################################################
    ############################## Numericals ##############################
    ########################################################################
    df_combined["AGE"] = df_combined["AGE"].replace({"90+": "90"})
    df_combined[num_cols] = df_combined[num_cols].astype(float)
    df_combined[num_cols] = df_combined[num_cols].mask(
        df_combined[num_cols] < 0, np.nan
    )
    ########################################################################
    ############################## Other stuff ##############################
    ########################################################################
    na_drop_cols = [col.upper() for col in na_drop_cols]
    recon_cols = [col.upper() for col in recon_cols]
    mast_cols = [col.upper() for col in mast_cols]
    surg_ind_cols = [col.upper() for col in surg_ind_cols]

    # Remove leading/trailing 0s
    string_cols = df_combined.select_dtypes(include=["object"]).columns
    for col in string_cols:
        df_combined[col] = df_combined[col].astype(str).str.strip()
        df_combined[col] = df_combined[col].replace({"None": None})
    ## Remove CPT/ICD cols
    df_combined = df_combined.drop(drop_cols, axis=1)
    ########################################################################
    ###################### Normalize instances names #######################
    ########################################################################
    print("Normalizing instances...")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=".*Downcasting behavior.*"
        )
        df_replaced = df_combined.replace(replace_dict).infer_objects(copy=False).copy()
    ########################################################################
    ############################ Deal with NAs #############################
    ########################## + further cleaning ##########################
    ########################################################################
    print("Dealing with NAs...")
    #### MORTALITY -->
    df_replaced.rename(columns={"YRDEATH": "MORTALITY"}, inplace=True)
    # Make all non-"No"s into yes
    df_replaced["MORTALITY"] = np.where(df_replaced["MORTALITY"] == "No", "No", "Yes")
    # add in DISCHDEST "expired" instances
    df_replaced["MORTALITY"] = np.where(
        (df_replaced["MORTALITY"] == "Yes") | (df_replaced["DISCHDEST"] == "Expired"),
        "Yes",
        df_replaced["MORTALITY"],
    )
    #### UNPLREOP --> IF died, make NAs No...otherwise drop (in code below)
    unplreop_mask = (df_replaced["MORTALITY"] == "Yes") & (
        df_replaced["UNPLREOP"].isna()
    )
    df_replaced.loc[unplreop_mask, "UNPLREOP"] = "No"
    ## Remove remaining NAs + other col NAs
    df_replaced = df_replaced.dropna(subset=na_drop_cols, ignore_index=True)
    ## Rename RACE_NEW to RACE
    df_replaced = df_replaced.rename(columns={"RACE_NEW": "RACE"})
    ########################################################################
    ######## change "No"s to NULL if not recorded for a given year ########
    ########################################################################
    # READ --> not recorded 08-10
    df_replaced.loc[
        (df_replaced["READ"] == "No") & df_replaced["OPERYR"].between(2008, 2010),
        "READ",
    ] = "Unknown(08-10)"
    # UNPLREAD --> 08-11
    df_replaced.loc[
        (df_replaced["UNPLREAD"] == "No") & df_replaced["OPERYR"].between(2008, 2011),
        "UNPLREAD",
    ] = "Unknown(08-11)"
    # RENAINSF --> 21
    df_replaced.loc[
        (df_replaced["RENAINSF"] == "No") & (df_replaced["OPERYR"].astype(int) == 2021),
        "RENAINSF",
    ] = "Unknown(21)"
    # RENAFAIL --> 21
    df_replaced.loc[
        (df_replaced["RENAFAIL"] == "No") & (df_replaced["OPERYR"].astype(int) == 2021),
        "RENAFAIL",
    ] = "Unknown(21)"
    # WTLOSS --> 21-24
    df_replaced.loc[
        (df_replaced["WTLOSS"] == "No") & df_replaced["OPERYR"].between(2021, 2024),
        "WTLOSS",
    ] = "Unknown(21-24)"
    # WNDINF --> 21-24
    df_replaced.loc[
        (df_replaced["WNDINF"] == "No") & df_replaced["OPERYR"].between(2021, 2024),
        "WNDINF",
    ] = "Unknown(21-24)"
    # DYSPNEA --> 21-24
    df_replaced.loc[
        (df_replaced["DYSPNEA"] == "No") & df_replaced["OPERYR"].between(2021, 2024),
        "DYSPNEA",
    ] = "Unknown(21-24)"

    ########################################################################
    ########################## Create new columns ##########################
    ########################################################################
    print("Creating new cols...")
    ##Lymph Node Surgery
    df_replaced["NOLYMPH"] = (
        (df_replaced["SNLBCPT"] == "0") & (df_replaced["ALNDCPT"] == "0")
    ).astype(int)
    ######## Timing of Surgery ########
    ## Create 3 new colunns
    count_of_ones_mast = df_replaced[mast_cols].astype(str).ne("0").sum(axis=1)
    count_of_ones_recon = df_replaced[recon_cols].astype(str).ne("0").sum(axis=1)
    df_replaced["MASTONLY"] = (
        (count_of_ones_mast >= 1) & (count_of_ones_recon == 0)
    ).astype(int)
    df_replaced["MASTRECON"] = (
        (count_of_ones_mast >= 1) & (count_of_ones_recon >= 1)
    ).astype(int)
    df_replaced["RECONONLY"] = (
        (count_of_ones_mast == 0) & (count_of_ones_recon >= 1)
    ).astype(int)
    ## Create single timing of surgery categorical column
    procedure_cols = ["MASTONLY", "MASTRECON", "RECONONLY"]
    # Ensure mutually exclusive
    assert len(df_replaced[procedure_cols].eq(1).sum(axis=1).value_counts()) == 1
    # Ensure all patients fall into 1 category
    assert df_replaced[procedure_cols].eq(1).sum(axis=1).value_counts().keys() == 1
    # except AssertionError:
    #     print(df_replaced[procedure_cols].eq(1).sum(axis=1).value_counts().keys())
    #     return df_replaced
    df_replaced["SURGTIMINGCPT"] = df_replaced[procedure_cols].idxmax(axis=1)
    df_replaced = df_replaced.drop(columns=procedure_cols, axis=1)
    ######## Surgical Indication (ICD) ########
    # Create single surgical indication (ICD) column
    ## Ensure mutually exclusive
    assert len(df_replaced[surg_ind_cols].eq(1).sum(axis=1).value_counts()) == 1
    assert df_replaced[surg_ind_cols].eq(1).sum(axis=1).value_counts().keys() == 1
    ## Combine
    df_replaced["SURGINDICD"] = df_replaced[surg_ind_cols].idxmax(axis=1)
    df_replaced = df_replaced.drop(columns=surg_ind_cols, axis=1)
    ########################################################################
    ############### Make binary columns all 1/0 ################
    ########################################################################
    ### Replace yes/no with 1/0
    binary_cols = get_feature_lists(df_replaced)["binary_cols"]
    replace_w_binary_cols = [
        col
        for col in binary_cols
        if "Yes" in df_replaced[col].unique() and "No" in df_replaced[col].unique()
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=".*Downcasting behavior.*"
        )
        df_replaced[replace_w_binary_cols] = (
            df_replaced[replace_w_binary_cols]
            .replace({"Yes": 1, "No": 0})
            .infer_objects(copy=False)
            .copy()
        )
        ### CHange others
        df_replaced["ETHNICITY_HISPANIC"] = df_replaced["ETHNICITY_HISPANIC"].replace(
            {"Yes": 1, "noUnknown": 0}
        )
        df_replaced["INOUT"] = df_replaced["INOUT"].replace(
            {"Inpatient": 1, "Outpatient": 0}
        )
        df_replaced["URGENCY"] = df_replaced["URGENCY"].replace(
            {"Urgent": 1, "Elective": 0}
        )
    return df_replaced


##############################################################################
################################## FILTER ####################################
##############################################################################
def create_and_filter_new_cols(
    *_,
    new_col_dict,
    old_df_dict,
    export_dir,
    target_cols,
    target_code_cols,
    filter_cols,
    extra_filtered,
    cpt_flag,
):
    """
    Loops through dict,
    creates new columns based on CPT/ICD codes,
    filters on a given subset of those columns,
    and exports resulting dataframes.

    Parameters
    ----------
    new_col_dict: dict{string:list[string]}
        maps new column names to lists of codes
    old_df_dict: dict{string:list[pd.Dataframe]}
        maps each year to its corresponding (original) data
    export_dir: pathlib.Path
        location of directory where resulting dfs will be exprted
    target_cols: list[string]
        list of columns to subset original df with
        simply used to simplify computation
        should include features+ new ICD cols + new CPT cols
    target_code_cols: list[string]
        list of columns containing CPT/ICD codes to search in
    filter_cols: list[string]
        subset of new_col_dict.keys() (new columns)to filter on
        resulting df will not be 0 for at least one of these columns
    extra_filtered: Boolean
        boolean flag indicating whether or not the data will be extra filtered
    cpt_flag: Boolean
        boolean flag indicating if the call is being made with CPT/ICD codes
    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    if cpt_flag:
        filter_type = "cpt"
    else:
        filter_type = "icd"
    new_df_dict = {}
    total_patients = 0
    ## Deal with export dir
    if extra_filtered:
        export_dir = export_dir / "extra_filtered"

    if export_dir.exists():
        rmtree(export_dir)
    export_dir.mkdir(exist_ok=True, parents=True)
    ## Loop through original dict of data files
    for file_name, file in old_df_dict.items():
        print(f"Working on {file_name}...")
        print(f"\t Initial number of patients: {len(file)}")
        ###########################################################
        ##################### Extract Cols ########################
        ###########################################################
        ## Make all columns upper case
        file.columns = file.columns.str.upper()
        # Subset df and col lists to match current df (can differ from year-year in NSQIP)
        # subset df on all cols
        df_sub = file[file.columns.intersection(target_cols)].copy()
        # get relevant code (ICD/CPT) cols
        target_code_cols_sub = df_sub.columns.intersection(target_code_cols)
        # Ensure code (ICD/CPT) cols are string in df
        df_sub[target_code_cols_sub] = df_sub[target_code_cols_sub].astype("string")
        ## Create new columns
        df_w_new_cols = extract_cols(
            df_sub,
            new_col_dict,
            target_code_cols_sub,
            cpt_flag=cpt_flag,
        )
        ###########################################################
        ######################## Filter ###########################
        ###########################################################
        # df_filtered = df_w_new_cols[df_w_new_cols[filter_cols].eq(1).any(axis=1)]
        # convert to string first
        df_filtered = df_w_new_cols[
            df_w_new_cols[filter_cols].astype(str).ne("0").any(axis=1)
        ]
        total_patients += len(df_filtered)  # add total patients
        print(f"\t Remaining: {len(df_filtered)}")
        new_df_dict[file_name] = df_filtered
        # Export
        export_path = export_dir / f"{file_name}_{filter_type}.parquet"
        df_filtered.to_parquet(export_path)
    print("*" * 30)
    print("*" * 30)
    print("*" * 30)
    print(f"TOTAL remaining patients post-{filter_type} filtering: {total_patients}")
    return new_df_dict


def extract_cols(import_df, new_cols_dict, target_cols_list, cpt_flag):
    """
    Extract binary indicator columns based on code matches.
    Used for CPT/ICD column generation.
    If CPT, count occurrences of codes. If ICD, simply track if any code occurs.

    Parameters
    -----------
    import_df: pandas dataframe
        raw tabular dataframe containing all necessary columns containing codes
    new_cols_dict: dict{<string>: list[<string>]}
        maps new column names to lists of codes
    target_cols_list: list[<string>]
        list of columns containing CPT/ICD codes to search in
    cpt_flag: boolean
        if True, use exact CPT matching (except for otherCPT); if False (ICD), use prefix matching
    """
    df = import_df.copy()

    ## Normalize columns (make string and upper case; also make NA empty '')
    for col in target_cols_list:
        df[col] = df[col].fillna("")
        if cpt_flag:
            try:
                df[col] = df[col].astype(float).astype(int).astype(str)
            except (ValueError, TypeError):
                df[col] = df[col].astype(str)
        else:
            df[col] = df[col].astype(str).str.upper()
    ## Check for matches
    for new_col, target_codes in new_cols_dict.items():
        # Exact matches (normalize in process)
        exact_codes = [
            str(code).upper()
            for code, match_type in target_codes
            if match_type == "exact"
        ]
        # Prefix matches (normalize in process)
        prefix_codes = [
            str(code).upper()
            for code, match_type in target_codes
            if match_type == "prefix"
        ]

        ################## CPT ##################
        def count_cpt(row, exact_codes, prefix_codes):
            count = 0
            for val in row:
                val = str(val).upper()
                if val in exact_codes:
                    count += 1
                elif any(val.startswith(prefix) for prefix in prefix_codes):
                    count += 1
            return count

        if cpt_flag:
            if new_col.upper() == "NPWTCPT":
                df[new_col] = (
                    df[target_cols_list]
                    .apply(
                        lambda col: (
                            col.isin(exact_codes)
                            | col.str.startswith(tuple(prefix_codes), na=False)
                        )
                    )
                    .any(axis=1)
                    .astype(int)
                )
            else:
                ## Count number of occurances
                df["count"] = df[target_cols_list].apply(
                    lambda row: count_cpt(row, exact_codes, prefix_codes), axis=1
                )
                ## Aggregate count into categorical
                df[new_col] = df["count"].apply(lambda x: "2+" if x >= 2 else str(x))
                df = df.drop(columns=["count"])
        ################## ICD (+ npwtCPT)##################
        else:
            df[new_col] = (
                df[target_cols_list]
                .apply(
                    lambda col: (
                        col.isin(exact_codes)
                        | col.str.startswith(tuple(prefix_codes), na=False)
                    )
                )
                .any(axis=1)
                .astype(int)
            )

    return df

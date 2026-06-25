from shutil import rmtree
import numpy as np
import pandas as pd
import warnings
from src.data_utils import get_feature_lists


##############################################################################
################################### CLEAN ####################################
##############################################################################
def get_code_cols(df, include_cpt=False):
    """
    Return the names of CPT/ICD code columns.

    Identifies columns holding diagnosis (``PODIAG``) or procedure (``CPT``)
    codes, excluding the curated single-purpose CPT flag columns in ``skip_cols``
    and the bare ``CPT`` column.

    Parameters
    ----------
    df : pd.DataFrame
        NSQIP df to inspect
    include_cpt : bool
        If True:
            keep generic ``CPT`` code columns (still excluding
            ``skip_cols``)
        If False:
            the skip-list filter is only applied within
            the ``CPT`` branch.

    Returns
    -------
    list[str]
        Names of the matching code columns.
    """
    skip_cols = [
        "SNLBCPT",
        "ALNDCPT",
        "PARTIALCPT",
        "SUBSIMPLECPT",
        "RADICALCPT",
        "MODIFIEDRADICALCPT",
        "PROCANATCPT",
        "IMMEDIATECPT",
        "DELAYEDCPT",
        "TEINSERTIONCPT",
        "TEEXPANDERCPT",
        "FREECPT",
        "LATCPT",
        "SINTRAMCPT",
        "SINTRAMSUPERCPT",
        "BITRAMCPT",
        "MASTOCPT",
        "BREASTREDCPT",
        "FATGRAFTCPT",
        "ADJTISTRANSCPT",
        "AUGPROSIMPCPT",
        "OTHERRECONTECHCPT",
        "REVRECBREASTCPT",
        "NPWTCPT",
    ]
    if include_cpt:
        code_cols = [
            col
            for col in df
            if (
                ("PODIAG" in col or "CPT" in col)
                and col != "CPT"
                and col.upper() not in skip_cols
            )
        ]
    else:
        code_cols = [
            col
            for col in df
            if ("PODIAG" in col or "CPT" in col and col.upper() not in skip_cols)
        ]
    return code_cols


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


############################################
########### Year-specific helpers ##########
############################################
def clean_08_10(df, include_cpt):
    df_w_codes = df.copy()
    ######################################
    ################ 2008 ################
    ######################################
    # {'READ', 'DISCHDEST', 'UNPLREAD'}
    df_w_codes.rename(
        columns={
            "EMERGNCY": "URGENCY",
            "RETURNOR": "UNPLNREOP",
        },
        inplace=True,
    )
    ## Readmission
    df_w_codes["READ"] = "Unknown_08_10"
    ## Unplanned Readmission
    df_w_codes["UNPLNREAD"] = "Unknown_08_10"
    ## Discharge Dest
    df_w_codes["DISCHDEST"] = "Unknown_08_10"
    df_w_codes.reset_index(drop=True, inplace=True)
    # No code df
    drop_cols = get_code_cols(df_w_codes, include_cpt)
    df_no_codes = df_w_codes.drop(drop_cols, axis=1)
    return df_w_codes, df_no_codes


def clean_11(df, include_cpt):
    df_w_codes = df.copy()
    df_w_codes.rename(
        columns={
            "EMERGNCY": "URGENCY",
            "RETURNOR": "UNPLNREOP",
            "READMISSION": "READ",
            "UNPLANREADMISSION": "UNPLNREAD",
        },
        inplace=True,
    )
    df_w_codes.reset_index(drop=True, inplace=True)
    ## DROP codes
    drop_cols = get_code_cols(df_w_codes, include_cpt)
    df_no_codes = df_w_codes.drop(drop_cols, axis=1)
    return df_w_codes, df_no_codes


def clean_12_20(df, include_cpt, year):
    """
    Clean for 12-20
    - Slight differences in combining cols for 12-14 and 15-20 cohorts
    - Note that this is eventually called for 21-24
    - 21-24 do not have BUT won't raise error EMERGNCY col
        - clean_21 renames `CASETYPE` --> `URGENCY
        - clean_22_24 renames `CASETYPE` --> `URGENCY`,
    - ONLY years 15-21 need to rename `BLEEDIS`--> `BLEEDDIS`, but will not raise error
    """
    df_w_codes = df.copy()
    df_w_codes.rename(
        columns={
            "EMERGNCY": "URGENCY",
            "BLEEDIS": "BLEEDDIS",
        },
        inplace=True,
    )
    ## Unplanned ReOp
    unplanned_reop_cols = [
        "RETURNOR",
        "REOPERATION1",
        "REOPERATION2",
        "REOPERATION3",
    ]
    df_w_codes["UNPLNREOP"] = df_w_codes[unplanned_reop_cols].apply(
        combine_columns, axis=1
    )
    df_w_codes.drop(unplanned_reop_cols, axis=1, inplace=True)
    ## Unplanned Readmission
    unplanned_read_cols = [
        "UNPLANNEDREADMISSION1",
        "UNPLANNEDREADMISSION2",
        "UNPLANNEDREADMISSION3",
        "UNPLANNEDREADMISSION4",
        "UNPLANNEDREADMISSION5",
    ]

    ## Readmission
    read_cols = [
        "READMISSION1",
        "READMISSION2",
        "READMISSION3",
        "READMISSION4",
        "READMISSION5",
    ]
    if year in range(2011, 2015):  # 2011-2014
        unplanned_read_cols.append("UNPLANREADMISSION")
        read_cols.append("READMISSION")
    # Unplanned Readmission
    df_w_codes["UNPLNREAD"] = df_w_codes[unplanned_read_cols].apply(
        combine_columns, axis=1
    )
    df_w_codes.drop(unplanned_read_cols, axis=1, inplace=True)
    # Readmission
    df_w_codes["READ"] = df_w_codes[read_cols].apply(combine_columns, axis=1)
    df_w_codes.drop(read_cols, axis=1, inplace=True)

    df_w_codes.reset_index(drop=True, inplace=True)
    drop_cols = get_code_cols(df_w_codes, include_cpt)
    df_no_codes = df_w_codes.drop(drop_cols, axis=1)
    return df_w_codes, df_no_codes


def clean_22_24(df, include_cpt):
    """
    Same as 15-20 w/ addition of:
    1) CASETYPE instead of EMERGNCY
    """
    df_w_codes = df.copy()

    df_w_codes.rename(columns={"CASETYPE": "URGENCY"}, inplace=True)
    # 2) adding missing cols
    df_w_codes["WTLOSS"] = "Unknown_21_24"
    df_w_codes["WNDINF"] = "Unknown_21_24"
    df_w_codes["DYSPNEA"] = "Unknown_21_24"
    # can just put arbitrary year bc not in 2012-2014
    return clean_12_20(df_w_codes, include_cpt, year=2022)


def clean_21(df, include_cpt):
    """
    Same as 22-24 w/ addition of missing cols:
    1) RENAINSF
    2) RENAFAIL
    """
    ## Same as  w/ addition of adding missing cols
    df_w_codes = df.copy()
    ## Add missing cols
    df_w_codes["RENAINSF"] = "Unknown_21"
    df_w_codes["RENAFAIL"] = "Unknown_21"
    return clean_22_24(df_w_codes, include_cpt)


def merge_dfs(data_dict, include_cpt, expcted_rows=69, verbose=False):
    """
    Merges NSQIP dataframes from 2008-2024, normalizing values to append vertically

    Parameters
    ----------
    data_dict: dict
        Dictionary mapping NSQIP file name to pandas df
    """
    w_codes_dict = {}  # fill w/ cleaned data (including code cols)
    no_codes_dict = {}  # fill w/ cleaned data (excluding code cols)
    for year in range(2008, 2025):  # 2008-2024
        yr_str = str(year)[-2:]  # last 2 digits
        nsqip_str = f"NSQIP_{yr_str}_cpt"
        if year in range(2008, 2011):  # 2008-2010
            df_w_codes, df_no_codes = clean_08_10(data_dict[nsqip_str], include_cpt)
        elif year == 2011:
            df_w_codes, df_no_codes = clean_11(data_dict[nsqip_str], include_cpt)
        elif year in range(2012, 2021):  # 2012-2020
            df_w_codes, df_no_codes = clean_12_20(
                data_dict[nsqip_str], include_cpt, year=year
            )
        elif year == 2021:
            df_w_codes, df_no_codes = clean_21(data_dict[nsqip_str], include_cpt)
        elif year in range(2022, 2025):  # 2022-2024
            df_w_codes, df_no_codes = clean_22_24(data_dict[nsqip_str], include_cpt)
        w_codes_dict[yr_str] = df_w_codes
        no_codes_dict[yr_str] = df_no_codes
        if verbose:
            print(f"{yr_str}...")
            print(df_no_codes.shape)
        else:
            print(f"{len(no_codes_dict)}/{len(data_dict)}")
    ##################################################
    ########### ENSURE we did things right ###########
    ##################################################
    ###### ENSURE we did things right #####
    try:  ### Right number of dfs
        assert len(w_codes_dict) == len(data_dict)
        assert len(w_codes_dict) == len(no_codes_dict)
    except AssertionError:
        print("Dicts do not match in size...")
        print(f" New length w/ codes: {len(w_codes_dict)}")
        print(f" New length w/o codes: {len(no_codes_dict)}")
        print(f" OG length: {len(data_dict)}")
        raise AssertionError
    for year1, df1 in no_codes_dict.items():
        if df1.shape[1] != expcted_rows:
            raise ValueError(
                f"Expected {expcted_rows} rows, got {df1.shape[1]} instead"
            )
        for year2, df2 in no_codes_dict.items():
            if year1 == year2:  # no need to compare the same dataset
                continue
            cols_1 = set(df1.columns)
            cols_2 = set(df2.columns)
            try:
                assert cols_2 - cols_1 == set()
                assert cols_1 - cols_2 == set()
            except AssertionError:
                print(f"In {year1} but not {year2}: {cols_1-cols_2}")
                print(f"In {year2} but not {year1}: {cols_2-cols_1}")
                raise AssertionError("Columns do not match in dfs!")
    ##### Combine
    combined_df_no_codes = pd.concat(no_codes_dict.values(), ignore_index=True)
    combined_df_w_codes = pd.concat(w_codes_dict.values(), ignore_index=True)
    print(f"Combined Shape No Codes: {combined_df_no_codes.shape}")
    print(f"Combined Shape With Codes: {combined_df_w_codes.shape}")
    ## Clean
    combined_df_no_codes["UNPLNREOP"] = (
        combined_df_no_codes["UNPLNREOP"].astype(str).apply(lambda x: x.strip())
    )
    return combined_df_no_codes, combined_df_w_codes


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
